import copy
import numpy as np
import torch

from torch.distributions import Normal, kl_divergence
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.ac import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# From ikostrikov's impl
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

        flat_params = torch.cat(params)
    return flat_params


# From ikostrikov's impl
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grads(loss, params, create_graph=False, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph)
    flat_grads = torch.cat([grad.view(-1) for grad in grads])
    return flat_grads


# TODO : ADD TYPES AND DESCRIPTION


class TRPO(A2C):
    """
    The sample-gathering and training algorithm.
    Based on TODO: Cite

    Implementation is based on
    - https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py # noqa E501
    - https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py
        - https://github.com/ajlangley/trpo-pytorch
    - https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py

    Some alterations have been made to the algorithms so it could be fitted to the
    tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_size: int = 256,
        action_std: float = 0.5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.99,
        entropy_loss_coeff: float = 0.01,
        delta: float = 0.01,
        max_backtracks: int = 10,
        backtrack_coeff: float = 0.05,
        n_update: int = 100,
        K_epochs: int = 1,
        batch_size: int = 10000,
        gm_seeding: bool = False,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_size: int
            Width of the RNN
        lr: float
            Learning rate for optimizer
        betas: float
            Beta parameter for Adam optimizer
        gamma: float
            Gamma parameter future reward discounting
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        actor_loss_coeff: float
            Loss coefficient on actor (policy) loss.
            Should sum to 1 with other loss coefficients
        critic_loss_coeff: float
            Loss coefficient on critic (value function estimator) loss
            Should sum to 1 with other loss coefficients
        entropy_loss_coeff: float,
            Loss coefficient on policy entropy
            Should sum to 1 with other loss coefficients
        gm_seeding: bool
            If seeding from GM, don't "go back"
        device: torch.device,
            Device to use for processing (CPU or GPU)
        load_pretrained: str
            Path to pretrained model
        """
        super(TRPO, self).__init__(
            input_size,
            action_size,
            hidden_size,
            action_std,
            lr,
            gamma,
            lmbda,
            entropy_loss_coeff,
            n_update,
            batch_size,
            gm_seeding,
            rng,
            device
        )

        self.on_policy = True

        # Declare policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_size, action_std,
        ).to(device)

        # Initialize teacher policy for imitation learning
        self.teacher = copy.deepcopy(self.policy)
        # Optimizer for actor

        # Note the optimizer is ran on the target network's params
        # TRPO: TRPO uses LGBFS optimization for the value function.
        # Kinda special to TRPO
        # self.optimizer = torch.optim.LBFGS(
        #     self.policy.critic.parameters(), lr=lr, max_iter=25)

        self.optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TRPO Specific parameters
        self.lmbda = lmbda
        self.entropy_loss_coeff = entropy_loss_coeff
        self.max_backtracks = max_backtracks

        self.backtrack_coeff = backtrack_coeff
        self.damping = 0.01
        self.delta = delta

        self.n_update = n_update
        self.K_epochs = K_epochs

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, batch_size, n_update, self.gamma)

    def update(
        self,
        replay_buffer
    ) -> Tuple[float, float]:
        """
        Policy update function, where we want to maximize the probability of
        good actions and minimize the probability of bad actions

        The general idea is to compare the current policy and the target
        policies. To do so, the "ratio" is calculated by comparing the
        probabilities of actions for both policies. The ratio is then
        multiplied by the "advantage", which is how better than average
        the policy performs.

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

        TRPO adds a twist to this where, since the advantage estimation is done
        with your (potentially bad) networks, a "pessimistic view" is used
        where gains will be clamped, so that high gradients (for very probable
        or with a high-amplitude advantage) are tamed. This is to prevent your
        network from diverging too much in the early stages

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions

        Returns
        -------
        running_actor_loss: float
            Average policy loss over all gradient steps
        running_critic_loss: float
            Average critic loss over all gradient steps
        """
        running_actor_loss = 0
        running_critic_loss = 0

        # Sample replay buffer
        state, action, returns, advantage, old_prob, old_mu, old_std = \
            replay_buffer.sample()

        def get_kl():

            def kl(mu, std):
                return kl_divergence(
                    Normal(old_mu.detach(), old_std.detach()),
                    Normal(mu, std)).mean()

            return kl

        def get_loss():
            def loss(policy):
                _, logprob, entropy, mu, std = policy.evaluate(
                    state,
                    action)

                ratio = torch.exp(logprob - old_prob)
                surrogate_policy_loss_1 = ratio * advantage
                # TRPO "pessimistic" policy loss
                actor_loss = surrogate_policy_loss_1.mean()
                # Entropy "loss" to promote entropy in the policy
                entropy_loss = entropy.mean()
                actor_loss += self.entropy_loss_coeff * entropy_loss
                return actor_loss, mu, std

            return loss

        def get_hessian(kl):
            """ Compute Hx, in a flattened version
            x is the grad of the actor loss
            """

            flat_grad_kl = get_flat_grads(
                kl, self.policy.actor.parameters(), create_graph=True)

            def Hx(x):

                kl_v = (flat_grad_kl @ x.clone())
                flat_grad_grad_kl = get_flat_grads(
                    kl_v, self.policy.actor.parameters())

                return flat_grad_grad_kl.detach() + (self.damping * x)

            return Hx

        def compute_conjugate_gradients(b, Hx, nsteps=10):
            """ Compute conjugate gradient of the actor loss gradient
            https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm # noqaE501
            """
            x = torch.zeros(b.size(), device=device)
            p = b.clone()
            r = b.clone()  # - Ax, but Ax = 0 with x = 0
            rr = torch.dot(r, r)
            for i in range(nsteps):
                Ap = Hx(p)
                alpha = rr / (torch.dot(p, Ap) + 1e-8)
                x += alpha * p
                r -= alpha * Ap
                rr_p = torch.dot(r, r)
                if rr_p < 1e-10:
                    break
                p = r + (rr_p / rr) * p
                rr = rr_p
            return x

        def get_step(g, Hx, delta):
            return torch.sqrt(2 * delta / torch.matmul(g, Hx(g)))

        def linesearch(g, step, kl, old_params, old_loss):

            step_size = 1. / self.backtrack_coeff  # to start backtrack at 1.

            for i in np.arange(self.max_backtracks):
                step_size *= self.backtrack_coeff
                new_params = old_params + (step * step_size)
                set_flat_params_to(self.policy.actor, new_params)
                with torch.no_grad():
                    pi_loss, mu, std = compute_loss(self.policy)
                    kl_mean = kl(mu, std)
                expected_improve = expected * step_size
                actual_improvement = pi_loss - old_loss
                ratio = actual_improvement / expected_improve

                # set_flat_params_to(self.policy.actor, old_params)
                kl_cond = kl_mean <= self.delta
                ratio_cond = ratio > 0.1
                improve_cond = actual_improvement > 0.
                if kl_cond and ratio_cond and improve_cond:
                    # print('Found suitable step', step_size)
                    # print('Improv', ratio)
                    # print('KL', kl_mean)
                    return pi_loss, step_size
            print('Linesearch failed', ratio, kl_mean)
            return old_loss, 0

        compute_loss = get_loss()

        actor_loss, old_mu, old_std = compute_loss(self.policy)
        kl = get_kl()
        kl_mean = kl(old_mu, old_std)
        loss_grad = get_flat_grads(
            actor_loss, self.policy.actor.parameters())

        Hx = get_hessian(kl_mean)

        # OpenAI baseline update
        step = compute_conjugate_gradients(loss_grad, Hx)
        max_step_coeff = (2 * self.delta / (step @ Hx(step)))**(0.5)
        max_trpo_step = max_step_coeff * step

        # shs = 0.5 * torch.matmul(g, Hx(g))
        # lm = torch.sqrt(shs / self.delta)
        # max_step = g / lm

        expected = loss_grad @  max_trpo_step

        old_params = get_flat_params_from(self.policy.actor)
        actor_loss, step_size = linesearch(
            step, max_trpo_step, kl, old_params, actor_loss)

        # print('Setting', max_step * step_size)
        set_flat_params_to(
            self.policy.actor, old_params + (max_trpo_step * step_size))

        for _ in range(self.K_epochs):

            # def critic_step():

            #     # V_pi'(s) and pi'(a|s)
            #     v_s, *_ = self.policy.evaluate(
            #         state,
            #         action)

            #     # TRPO critic loss
            #     critic_loss = ((returns - v_s) ** 2).mean()

            #     # Critic gradient step
            #     self.optimizer.zero_grad()
            #     critic_loss.backward()
            #     return critic_loss
            # self.optimizer.step(critic_step)
            v_s, *_ = self.policy.evaluate(
                state,
                action)

            # TRPO critic loss
            critic_loss = ((returns - v_s) ** 2).mean()

            # Critic gradient step
            self.optimizer.zero_grad()
            critic_loss.backward()

            self.optimizer.step()
            running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        # Keep track of losses
        running_actor_loss += actor_loss.mean().cpu().detach().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss, \
            running_critic_loss
