import copy
import numpy as np
import torch

from os.path import join as pjoin
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.ac import ActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDN(nn.Module):
    """
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super(SDN, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, samples):
        advantage = self._forward(states, actions).squeeze(-1)
        advantage_samples = torch.cat([
            self._forward(states, action).squeeze(-1)
            for action in samples], dim=-1)
        return advantage, advantage_samples.mean()

    def _forward(self, states, actions):
        """ Interestingly, the SDN computes the advantage directly.
        """
        p = torch.relu(self.l1(torch.cat([states, actions], dim=-1)))
        p = torch.relu(self.l2(p))
        advantage = self.l3(p)

        return advantage


class ACERAgent(ActorCritic):
    """ Actor-Critic module that handles both actions and values
    Actors and critics here don't share a body but do share a loss
    function. Therefore they are both in the same module
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_std: float = 0.5,
    ):
        super(ACERAgent, self).__init__(
            state_dim, action_dim, hidden_dim, action_std)
        self.n = 5

        self.sdn = SDN(
            state_dim, action_dim, hidden_dim,
        ).to(device)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        values, action_logprob, entropy, mu, std = \
            super().evaluate(state, action)

        samples = [self.act(state, True) for _ in range(self.n)]

        advantage, advantage_mean = self.sdn(
            state, action, samples)

        # Equation 13, q_tilde
        q = values + advantage - advantage_mean

        return values, action_logprob, q, entropy, mu, std

    def get_evaluation(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)

        v, prob, q, entropy, mu, std = self.evaluate(state, action)

        return (
            v.cpu().data.numpy(),
            prob.cpu().data.numpy(),
            q.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict, sdn_state_dict = state_dict
        super().load_state_dict((actor_state_dict, critic_state_dict))
        self.sdn.load_state_dict(sdn_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        actor_state_dict, critic_state_dict = super().state_dict()
        return (actor_state_dict, critic_state_dict,
                self.sdn.state_dict())

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        super().save(path, filename)
        torch.save(
            self.sdn.state_dict(), pjoin(path, filename + "_sdn.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        super().load(path, filename)
        self.sdn.load_state_dict(
            torch.load(pjoin(path, filename + '_sdn.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()
        self.sdn.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()
        self.sdn.train()


# TODO : ADD TYPES AND DESCRIPTION
class ACER(A2C):
    """
    The sample-gathering and training algorithm.
    TODO: Cite



    Implementation was inspired by
    - https://github.com/dchetelat/acer # noqa E501
    - https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/acer.py

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

        super(ACER, self).__init__(
            input_size,
            action_size,
            hidden_size,
            action_std,
            lr,
            gamma,
            batch_size,
            gm_seeding,
            rng,
            device,
        )

        self.on_policy = True

        # Declare policy
        self.policy = ACERAgent(
            input_size, action_size, hidden_size,  # action_std,
        ).to(device)

        # "Average" network
        self.average = copy.deepcopy(self.policy)

        # Optimizer for actor
        # Note the optimizer is ran on the target network's params
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=lr)

        # GAE Parameter
        self.lmbda = lmbda

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size)

        self.tau = 0.005
        self.k = 2

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
        c = 10
        delta = 0.1

        running_actor_loss = 0
        running_critic_loss = 0

        # TODO: Average

        # alpha = 0.995
        # k = 50
        # c = 5

        # torch.cuda.empty_cache()

        return running_actor_loss, running_critic_loss
