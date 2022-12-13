import copy
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nibabel.streamlines import Tractogram

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer
from TrackToLearn.algorithms.shared.utils import add_to_means
from TrackToLearn.environments.env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on
        Scott Fujimoto, Herke van Hoof, David Meger
        "Addressing Function Approximation Error in
        Actor-Critic Methods", 2018;
        https://arxiv.org/abs/1802.09477 arXiv:1802.09477

    Implementation is based on
    - https://github.com/sfujim/TD3

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_dims: str = '',
        lr: float = 3e-4,
        gamma: float = 0.99,
        action_std: float = 0.35,
        batch_size: int = 2048,
        interface_seeding: bool = False,
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
            Width of the model
        action_std: float
            Standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        batch_size: int
            Batch size for replay buffer sampling
        interface_seeding: bool
            If seeding from GM, don't "go back"
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(TD3, self).__init__(input_size,
                                  action_size,
                                  action_std,
                                  lr,
                                  gamma,
                                  batch_size,
                                  interface_seeding,
                                  rng,
                                  device)

        # Initialize main policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # TD3 requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TD3-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 0
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005
        self.noise_clip = 1

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size)

        self.rng = rng

    def _episode(
        self,
        initial_state: np.ndarray,
        env: BaseEnv,
    ) -> Tuple[Tractogram, float, float, float, int]:
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done
        Gather transitions and train on them according to the RL algorithm's
        rules.

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        tractogram: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        actor_loss: float
            Policty gradient loss of actor
        critic_loss: float
            MSE loss of critic
        episode_length: int
            Length of episode aka how many transitions were gathered
        """

        running_reward = 0
        state = initial_state
        tractogram = None
        done = False

        episode_length = 0

        running_losses = defaultdict(list)

        while not np.all(done):

            # Select action according to policy + noise for exploration
            a, h = self.policy.select_action(np.array(state))
            action = (
                a + self.rng.normal(
                    0, self.max_action * self.action_std,
                    size=a.shape)
            ).clip(-self.max_action, self.max_action)

            self.t += action.shape[0]
            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = done

            # Store data in replay buffer
            # WARNING: This is a bit of a trick and I'm not entirely sure this
            # is legal. This is effectively adding to the replay buffer as if
            # I had n agents gathering transitions instead of a single one.
            # This is not mentionned in the SAC paper. PPO2 does use multiple
            # learners, though.
            # I'm keeping it since since it reaaaally speeds up training with
            # no visible costs
            self.replay_buffer.add(
                state, action,
                next_state, reward[..., None],
                done_bool[..., None])

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:
                losses = self.update(
                    self.replay_buffer)
                running_losses = add_to_means(running_losses, losses)
            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, h, _ = env.harvest(next_state, h)

            # Keeping track of episode length
            episode_length += 1

        tractogram = env.get_streamlines()
        return (
            tractogram,
            running_reward,
            running_losses,
            episode_length)

    def update(
        self,
        replay_buffer: OffPolicyReplayBuffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """
        TODO: Add motivation behind TD3 update ("pessimistic" two-critic
        update, policy that implicitely maximizes the q-function, etc.)

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions
        batch_size: int
            Batch size to sample the memory

        Returns
        -------
        running_actor_loss: float
            Average policy loss over all gradient steps
        running_critic_loss: float
            Average critic loss over all gradient steps
        """
        self.total_it += 1

        # Sample replay buffer
        # TODO: Make batch size parametrizable
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select next action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * (self.action_std * 2)
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.target.actor(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value for s'
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.policy.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_Q1, target_Q.detach()).mean()
        loss_q2 = F.mse_loss(current_Q2, target_Q.detach()).mean()

        critic_loss = loss_q1 + loss_q2

        losses = {
            'actor_loss': 0.0,
            'critic_loss': critic_loss.item(),
            'q1': current_Q1.mean().item(),
            'q2': current_Q2.mean().item(),
            'q1_loss': loss_q1.item(),
            'q2_loss': loss_q2.item(),
        }

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss -Q(s,a)
            actor_loss = -self.policy.critic.get_q1(
                state, self.policy.act(state)).mean()

            losses['actor_loss'] = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.policy.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.policy.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return losses
