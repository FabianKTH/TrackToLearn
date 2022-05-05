import copy
import numpy as np
import torch

from torch import nn
from typing import Tuple

from TrackToLearn.algorithms.vpg import VPG
from TrackToLearn.algorithms.shared.ac import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO : ADD TYPES AND DESCRIPTION
class A2C(VPG):
    """
    The sample-gathering and training algorithm.

    Implementation is based on TODO

    Some alterations have been made to the algorithms so it could be fitted to
    the tractography problem.

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
        n_update: int = 5,
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
        print(input_size, action_size, hidden_size, action_std, lr, gamma, entropy_loss_coeff, n_update, batch_size, gm_seeding, lmbda)

        super(A2C, self).__init__(
            input_size,
            action_size,
            hidden_size,
            action_std,
            lr,
            gamma,
            entropy_loss_coeff,
            n_update,
            batch_size,
            gm_seeding,
            rng,
            device,
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

        # GAE Parameter
        self.gamma = gamma
        self.lr = lr
        self.lmbda = lmbda
        self.entropy_loss_coeff = entropy_loss_coeff
        self.n_update = n_update

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, batch_size, self.n_update, self.gamma, self.lmbda)

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
        assert log_prob.size() == returns.size(), \
            '{}, {}'.format(log_prob.size(), returns.size())

        # VPG policy loss
        actor_loss = -(log_prob * advantage).mean() + \
            -self.entropy_loss_coeff * entropy.mean()

        # AC Critic loss
        critic_loss = (advantage ** 2).mean()

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

        return running_actor_loss, running_critic_loss
