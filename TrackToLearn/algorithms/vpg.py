import copy
import numpy as np
import torch

from nibabel.streamlines import Tractogram

from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.ac import PolicyGradient
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
# from TrackToLearn.algorithms.shared.utils import harvest_states, stack_states
from TrackToLearn.environments.env import BaseEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO : ADD TYPES AND DESCRIPTION
class VPG(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on TODO

    Ratio and clipping were removed from PPO to obtain VPG.

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
        gm_seeding: bool
            If seeding from GM, don't "go back"
        device: torch.device,
            Device to use for processing (CPU or GPU)
        load_pretrained: str
            Path to pretrained model
        """

        super(VPG, self).__init__(
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

        self.entropy_loss_coeff = entropy_loss_coeff

        # Declare policy
        self.policy = PolicyGradient(
            input_size, action_size, hidden_size, action_std,
        ).to(device)

        # Initialize teacher policy for imitation learning
        self.teacher = copy.deepcopy(self.policy)

        # Optimizer for actor
        # Note the optimizer is ran on the target network's params
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, batch_size, n_update, self.gamma, lmbda=0.)

        self.n_update = n_update

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
        running_actor_loss = 0
        running_critic_loss = 0

        episode_length = 0
        indices = np.asarray(range(state.shape[0]))

        while not np.all(done):

            for _ in range(self.n_update):
                # Select action according to policy
                # Noise is already added by the policy
                action = self.policy.select_action(
                    np.array(state), stochastic=True)

                v, prob, _, mu, std = self.policy.get_evaluation(
                    np.array(state),
                    action)

                self.t += 1
                # Perform action
                next_state, reward, done, _ = env.step(action)

                vp, *_ = self.policy.get_evaluation(
                    np.array(next_state),
                    action)

                # Set next state as current state
                running_reward += sum(reward)

                # Store data in replay buffer
                self.replay_buffer.add(
                    indices, state, action, next_state,
                    reward, done, v, vp, prob, mu, std)

                # "Harvesting" here means removing "done" trajectories
                # from state as well as removing the associated streamlines
                state, idx = env.harvest(next_state)

                indices = indices[idx]

                # Keeping track of episode length
                episode_length += 1

                if np.all(done):
                    break

            # Train agent after collecting sufficient data
            # Heuristic to prevent small batches from crashing
            if len(self.replay_buffer) >= 10:
                actor_loss, critic_loss = self.update(
                    self.replay_buffer)
            else:
                actor_loss, critic_loss = 0, 0
            self.replay_buffer.clear_memory()

            running_actor_loss += actor_loss
            running_critic_loss += critic_loss

        tractogram = env.get_streamlines()

        return (
            tractogram,
            running_reward,
            running_actor_loss,
            running_critic_loss,
            episode_length)

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
        """

        running_actor_loss = 0

        # Sample replay buffer
        state, action, returns, *_ = \
            replay_buffer.sample()

        log_prob, entropy, *_ = self.policy.evaluate(state, action)

        # Surrogate policy loss
        assert log_prob.size() == returns.size(), \
            '{}, {}'.format(log_prob.size(), returns.size())

        # VPG policy loss
        actor_loss = -(log_prob * returns).mean() + \
            -self.entropy_loss_coeff * entropy.mean()

        # Gradient step
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Keep track of losses
        running_actor_loss += actor_loss.mean().cpu().detach().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss, 0
