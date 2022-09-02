import copy
from os.path import join as pjoin
from typing import Tuple

import numpy as np
import torch
from nibabel.streamlines import Tractogram
# from spharmnet.core.models import DirectMaxSPHARMNet
# spharm-net imports
from spharmnet.core.layers import ISHT, SHT
from spharmnet.core.models import Down, Final
from spharmnet.lib.io import read_mesh
from spharmnet.lib.sphere import spharm_real, vertex_area
from torch import nn

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """ Replay buffer to store transitions. Implemented in a "ring-buffer"
    fashion. Efficiency could probably be improved

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_size: [int, int], action_dim: int, max_size=int(1e6)
    ):
        """
        Parameters:
        -----------
        state_size: int
            Size of states
        action_dim: int
            Size of actions
        max_size: int
            Number of transitions to store
        """
        self.device = device
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Buffers "filled with zeros"

        # TODO: switch to 2D

        self.state = np.zeros([self.max_size] + state_size, dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros_like(self.state, dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        """

        ind = (np.arange(0, len(state)) + self.ptr) % self.max_size

        self.state[ind] = state
        self.action[ind] = action
        self.next_state[ind] = next_state
        self.reward[ind] = reward
        self.not_done[ind] = 1. - done

        self.ptr = (self.ptr + len(ind)) % self.max_size
        self.size = min(self.size + len(ind), self.max_size)

    def sample(
        self,
        batch_size=1024
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Off-policy sampling. Will sample min(batch_size, self.size)
        transitions in an unordered way. This removes the ability to do
        GAE and reward discounting after the transitions are sampled

        Parameters:
        -----------
        batch_size: int
            Number of transitions to sample

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ns: torch.Tensor
            Sampled s'
        r: torch.Tensor
            Sampled non-discounted rewards
        d: torch.Tensor
            Sampled 1-done flags
        """

        ind = np.random.randint(0, self.size, size=int(batch_size))

        s = torch.as_tensor(
            self.state[ind], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(
            self.action[ind], dtype=torch.float32, device=self.device)
        ns = \
            torch.as_tensor(
                self.next_state[ind], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(
            self.reward[ind], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(
            self.not_done[ind], dtype=torch.float32, device=self.device)

        return s, a, ns, r, d

    def clear_memory(self):
        """ Reset the buffer
        """
        self.ptr = 0
        self.size = 0

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass


class SPH2VEC(nn.Module):
    """
    SPH2VEC
    reorders sph entries at positions 1-3 and discards first one (constant w.r.t. direction)

    """
    def __init__(self):
        super().__init__()
        self.idxmap = torch.tensor([2, 0, 1], device=device)

    def forward(self, x):
        idx = self.idxmap.expand(x[..., 1:].shape)
        v = torch.gather(x[..., 1:], 2, idx)

        return v


class PadToLmax(nn.Module):
    """
    PadToLmax
    appends zeros to last dimesion in order to extend a sph coeff vector up to l_max

    """
    def __init__(self, l_in = 1, l_out: int = 6):
        super().__init__()
        no_sphin = (l_in + 1)**2
        no_sphout = (l_out + 1)**2

        self.pad = [0, no_sphout - no_sphin]

    def forward(self, x):
        return nn.functional.pad(x, self.pad, value=0.)


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        sphere: str,
        in_ch: int = 4,
        C: int = 8,
        L: int = 6,
        D: int = None,
        interval: int = 5,
        threads: int = 1,
        verbose: bool = False
    ):
        """
        DirectMaxSPHARNNet.
        TODO description

        Parameters
        __________
        sphere : str
            Sphere mesh file. VTK (ASCII) < v4.0 or FreeSurfer format.
        in_ch : int
            # of input geometric features.
        C : int
            # of channels in the entry layer (see the paper for details).
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        D : int
            Depth of encoding/decoding levels (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        threads: int
            # of CPU threads for basis reconstruction. Useful if the unit sphere has dense tesselation.

        Notes
        _____
        In channel shape  : [batch, in_ch, no_sph_coords]
        Out channel shape : [batch, 4]
        """

        super().__init__()

        if D is None:
            D = int(np.log(C))  # downsample L until we reach L = 1

        L_in = L
        self.down = []

        ch_inc = 2
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)

        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        Y_inv = Y.T

        # first: transform sph harm to spherical signal
        self.in_ = ISHT(Y)

        # first layer: keep L-size but increase channels
        if verbose:
            print("Down {}\t| C:{} -> {}\t| L:{}".format(0, in_ch, out_ch, L))
        self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True))

        # encoding
        for i in range(D):
            L = L_in >> i+1  # downsample, L_new = L+old/2
            in_ch = out_ch
            out_ch *= ch_inc
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))

        if verbose:
            print("Final\t| C:{} -> {}\t| L:{}".format(out_ch, 1, L))
        self.final = Final(Y, Y_inv, area, out_ch, 1, L, interval)

        # lastly: lift signal back into harmonics domain
        self.out = SHT(L=1, Y_inv=Y_inv, area=area)

        # sph to direction vector
        self.out_dir = SPH2VEC()

        # note: why is this required? ...
        self.down = nn.ModuleList(self.down)

    def forward(self, x):
        x = self.in_(x)
        for l_idx, layer in enumerate(self.down):
            # print(f'l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            x = layer(x)
        x = self.final(x)
        x = self.out(x)

        # sph with L=1 to direction
        # x = self.out_dir(x)

        return x, None  # None here just to fit outer call structure TODO: fix


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. SAC uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
            self,
            sphere: str,
            in_ch: int = 5,
            C: int = 8,
            L: int = 6,
            D: int = None,
            interval: int = 5,
            threads: int = 1,
            verbose: bool = False
            ):
        """
        DirectMaxSPHARNNet.
        TODO description

        Parameters
        __________
        sphere : str
            Sphere mesh file. VTK (ASCII) < v4.0 or FreeSurfer format.
        in_ch : int
            # of input geometric features.
        C : int
            # of channels in the entry layer (see the paper for details).
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        D : int
            Depth of encoding/decoding levels (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        threads: int
            # of CPU threads for basis reconstruction. Useful if the unit sphere has dense tesselation.

        Notes
        _____
        In channel shape  : [batch, in_ch, no_sph_coords]
        Out channel shape : [batch, 1]
        """

        super().__init__()

        if D is None:
            D = int(np.log(C))  # downsample L until we reach L = 1

        L_in = L

        self.q1_down = []
        self.q2_down = []

        ch_inc = 2
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)

        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        Y_inv = Y.T

        # first: transform sph harm to spherical signal
        self.in_ = ISHT(Y)

        # pad action (action is expected to be of size 4 for sph vec for l=1)
        self.pad_action = PadToLmax(l_in=1, l_out=L)

        # encoding
        for i in range(D):
            L = L_in >> i+1  # downsample, L_new = L+old/2

            if verbose:
                print("(q1,q2)_down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))

            self.q1_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))
            self.q2_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))
            in_ch = out_ch
            out_ch *= ch_inc

        if verbose:
            print("(q1,q2)_final\t| C:{} -> {}\t| L:{}".format(in_ch, 1, 0))
        self.q1_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)
        self.q2_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)

        # lastly: lift signal back into harmonics domain
        self.out = SHT(L=0, Y_inv=Y_inv, area=area)

        # note: why is this required? ...
        self.q1_down = nn.ModuleList(self.q1_down)
        self.q2_down = nn.ModuleList(self.q2_down)

    def forward(self, state, action):
        # isht of state
        state = self.in_(state)

        # pad action
        action = self.pad_action(action)

        # combine state and action to input
        q1 = torch.cat([state, action], 1)
        q2 = torch.cat([state, action], 1)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.q1_final(q1)

        # q2
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q2) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q2 = layer(q2)
        q2 = self.q2_final(q2)

        return q1, q2

    def Q1(self, state, action):
        # isht of state
        state = self.in_(state)

        # pad action
        action = self.pad_action(action)

        # combine state and action to input
        q1 = torch.cat([state, action], 1)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.q1_final(q1)

        return q1


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ActorCritic:
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        sphere: str,
        in_ch: int,
        C: int,
        L: int,
        D: int,
        interval: int,
        threads: int,
        verbose: bool,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            sphere: str
                Path of a vtk mesh file
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity
            sphere: str
                Path of a vtk mesh file
        """
        self.actor = Actor(
            sphere,
            in_ch,
            C,
            L,
            D,
            interval,
            threads,
            verbose,
        ).to(device)

        self.critic = Critic(
            sphere,
            in_ch,
            C,
            L,
            D,
            interval,
            threads,
            verbose,
        ).to(device)

    def act(
        self,
        state: torch.Tensor,
        stochastic: bool = True,
    ) -> (torch.Tensor, torch.Tensor):
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        a, logprob = self.actor(state, stochastic)
        return a, logprob

    def select_action(self, state: np.array, stochastic=True) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            deterministic: bool
                Return deterministic action (at test time)

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action, _ = self.act(state, stochastic)

        return action.cpu().data.numpy()

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

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
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

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
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth')))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()


class SPHSAC(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on
    TODO: Cite
    Implementation is based on Spinning Up's and rlkit

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        sphere: str,
        in_ch: int,
        C: int,
        L: int,
        D: int,
        interval: int,
        threads: int,
        verbose: bool,
        action_size: int = 3,
        hidden_size: int = 256,
        hidden_layers: int = 3,
        action_std: float = 0.35,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        batch_size: int = 2048,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        sphere: str
            Path of a vtk mesh file
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
        alpha: float
            Temperature parameter
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(SPHSAC, self).__init__(
            input_size,
            action_size,
            hidden_size,
            action_std,
            lr,
            gamma,
            batch_size,
            rng,
            device,
        )

        # Initialize main policy
        self.policy = ActorCritic(
            sphere,
            in_ch,
            C,
            L,
            D,
            interval,
            threads,
            verbose,
            )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005
        self.noise_clip = 0.5
        self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size)

        # Fabi spherical cnn parameters
        self.sphere = sphere
        self.in_ch    = in_ch
        self.C        = C
        self.L        = L
        self.D        = D
        self.interval = interval
        self.threads  = threads
        self.verbose  = verbose

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
        actor_loss = 0
        critic_loss = 0

        episode_length = 0

        while not np.all(done):

            # Select action according to policy + noise for exploration
            action = self.policy.select_action(
                np.array(state), stochastic=True)

            self.t += action.shape[0]
            # Perform action

            # import ipdb; ipdb.set_trace()

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
            # TODO: Add monitors so that losses are properly tracked
            if self.t >= self.start_timesteps:
                actor_loss, critic_loss = self.update(
                    self.replay_buffer)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            new_tractogram, state, _ = env.harvest(next_state)

            # Add streamlines to the lot
            if len(new_tractogram.streamlines) > 0:
                if tractogram is None:
                    tractogram = new_tractogram
                else:
                    tractogram += new_tractogram

            # Keeping track of episode length
            episode_length += 1

        return (
            tractogram,
            running_reward,
            actor_loss,
            critic_loss,
            episode_length)

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """
        TODO: Add motivation behind SAC update ("pessimistic" two-critic
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

        running_actor_loss = 0
        running_critic_loss = 0

        self.total_it += 1

        # Sample replay buffer
        # TODO: Make batch size parametrizable
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_next_action = self.policy.act(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            backup = reward + self.gamma * not_done * \
                (target_Q - self.alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.policy.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = ((current_Q1 - backup)**2).mean()
        loss_q2 = ((current_Q2 - backup)**2).mean()
        critic_loss = loss_q1 + loss_q2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, logp_pi = self.policy.act(state)
        q1, q2 = self.policy.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized policy loss
        actor_loss = (self.alpha * logp_pi - q_pi).mean()

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

        running_actor_loss = actor_loss.detach().cpu().numpy()

        running_critic_loss = critic_loss.detach().cpu().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss, running_critic_loss
