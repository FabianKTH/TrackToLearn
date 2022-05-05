import copy
import numpy as np
import torch

from torch import nn
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.ac import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO : ADD TYPES AND DESCRIPTION
class PPO(A2C):
    """
    The sample-gathering and training algorithm.
    Based on
        John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford:
            “Proximal Policy Optimization Algorithms”, 2017;
            http://arxiv.org/abs/1707.06347 arXiv:1707.06347

    Implementation is based on
    - https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py # noqa E501
    - https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py
    - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py # noqa E501

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
        K_epochs: int = 80,
        n_update: int = 10,
        eps_clip: float = 0.01,
        entropy_loss_coeff: float = 0.01,
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
        n_update: int
            How many steps to gather data before training
        K_epochs: int
            How many epochs to run the optimizer using the current samples
            PPO allows for many training runs on the same samples
        eps_clip: float
            Clipping parameter for PPO
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
        super(PPO, self).__init__(
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

        # Note the optimizer is ran on the target network's params
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)
        # self.critic_optimizer = torch.optim.Adam(
        #     self.policy.critic.parameters(), lr=lr)

        # PPO Specific parameters
        self.n_update = n_update
        self.K_epochs = K_epochs
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.entropy_loss_coeff = entropy_loss_coeff

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, batch_size, n_update, self.gamma, self.lmbda)

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

        PPO adds a twist to this where, since the advantage estimation is done
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
        state, action, returns, advantage, old_prob, *_ = \
            replay_buffer.sample()

        # PPO allows for multiple gradient steps on the same data
        for _ in range(self.K_epochs):

            # V_pi'(s) and pi'(a|s)
            v, logprob, entropy, *_ = self.policy.evaluate(
                state,
                action)

            # Ratio between probabilities of action according to policy and
            # target policies
            assert logprob.size() == old_prob.size(), \
                '{}, {}'.format(logprob.size(), old_prob.size())
            ratio = torch.exp(logprob - old_prob)

            # Surrogate policy loss
            assert ratio.size() == advantage.size(), \
                '{}, {}'.format(ratio.size(), advantage.size())

            # Finding V Loss:
            assert returns.size() == v.size(), \
                '{}, {}'.format(returns.size(), v.size())

            surrogate_policy_loss_1 = ratio * advantage
            surrogate_policy_loss_2 = torch.clamp(
                ratio,
                1-self.eps_clip,
                1+self.eps_clip) * advantage

            # PPO "pessimistic" policy loss
            actor_loss = -(torch.min(
                surrogate_policy_loss_1,
                surrogate_policy_loss_2)).mean() + \
                -self.entropy_loss_coeff * entropy.mean()

            # AC Critic loss
            critic_loss = ((v - returns) ** 2).mean()

            self.optimizer.zero_grad()
            ((critic_loss * 0.5) + actor_loss).backward()

            # Gradient step
            nn.utils.clip_grad_norm_(self.policy.parameters(),
                                     0.5)
            self.optimizer.step()

            # Keep track of losses
            running_actor_loss += actor_loss.mean().cpu().detach().numpy()
            running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss / self.K_epochs, \
            running_critic_loss / self.K_epochs
