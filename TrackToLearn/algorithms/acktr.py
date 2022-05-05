import copy
import numpy as np
import torch

from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.ac import ActorCritic
from TrackToLearn.algorithms.optim import KFACOptimizer
from TrackToLearn.algorithms.shared.replay import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO : ADD TYPES AND DESCRIPTION
class ACKTR(A2C):
    """
    The sample-gathering and training algorithm.

        Wu, Y., Mansimov, E., Liao, S., Grosse, R., & Ba, J. (2017).
        Scalable trust-region method for deep reinforcement learning using
        kronecker-factored approximation. arXiv preprint arXiv:1708.05144.

    Implementation is based on
     - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py # noqa E501
     - https://github.com/alecwangcq/KFAC-Pytorch/blob/master/optimizers/kfac.py

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
        entropy_loss_coeff: float = 0.0001,
        delta: float = 0.001,
        n_update: int = 100,
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
        gm_seeding: bool
            If seeding from GM, don't "go back"
        device: torch.device,
            Device to use for processing (CPU or GPU)
        load_pretrained: str
            Path to pretrained model
        """
        super(ACKTR, self).__init__(
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
        self.optimizer = KFACOptimizer(
            self.policy, lr=lr, kl_clip=delta)

        # GAE Parameter
        self.lmbda = lmbda

        self.entropy_loss_coeff = entropy_loss_coeff

        self.n_update = n_update

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

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

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
        state, action, returns, advantage, *_ = \
            replay_buffer.sample()

        v, log_prob, entropy, *_ = self.policy.evaluate(state, action)

        # Surrogate policy loss
        assert log_prob.size() == advantage.size(), \
            '{}, {}'.format(log_prob.size(), advantage.size())

        # Finding V Loss:
        assert returns.size() == v.size(), \
            '{}, {}'.format(returns.size(), v.size())

        # ACKTR policy loss
        actor_loss = -(log_prob * advantage).mean() + \
            -self.entropy_loss_coeff * entropy.mean()

        # ACKTR critic loss
        critic_loss = ((v - returns) ** 2).mean()

        if self.optimizer.steps % self.optimizer.Ts == 0:
            self.policy.zero_grad()
            pg_fisher_loss = -log_prob.mean()

            noisy_v = v + torch.randn(v.size(), device=device)
            vf_fisher_loss = -(v - noisy_v.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss

            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        # Gradient step
        self.optimizer.zero_grad()
        ((critic_loss * 0.5) + actor_loss).backward()
        self.optimizer.step()

        # Keep track of losses
        running_actor_loss += actor_loss.mean().cpu().detach().numpy()
        running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss, running_critic_loss
