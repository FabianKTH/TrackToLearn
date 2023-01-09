import copy
from collections import defaultdict
from os.path import join as pjoin
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nibabel.streamlines import Tractogram
# spharm-net imports
from spharmnet.core.layers import ISHT, SHT, SHConv
from spharmnet.core.models import Down, Final
from spharmnet.lib.io import read_mesh
from spharmnet.lib.sphere import spharm_real, vertex_area
from torch.optim.lr_scheduler import ExponentialLR

from torch import nn

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.utils import add_to_means
from TrackToLearn.algorithms.direction_so3 import PropDirGetter, OdfDirGetter
from TrackToLearn.environments.env import BaseEnv
# from TrackToLearn.fabi_utils.communication import IbafServer
from TrackToLearn.so3_utils.rotation_utils import SPH2VEC, dirs_to_sph_channels
from TrackToLearn.so3_utils.utils import PadToLmax, nocoeff_from_l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_spharm_basis(L, sphere, threads):
    v, f = read_mesh(sphere)
    v = v.astype(float)
    area = vertex_area(v, f)
    Y = spharm_real(v, L, threads)
    area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
    Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
    Y_inv = Y.T
    return Y, Y_inv, area


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
        try:
            self.next_state[ind] = next_state
        except:
            pass
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


class ConvBlock(nn.Module):
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True, is_final=False):
        super().__init__()
        self.is_final = is_final
        self.isht = ISHT(Y)

        self.shconv = nn.Sequential(
            SHConv(in_ch, out_ch, L, interval),
            ISHT(Y),
        )
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=False) if fullband else None
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False)
        self.sht = SHT(L, Y_inv, area)

    def forward(self, x):
        x1 = self.shconv(x)
        if self.impulse is not None:
            x_isht = self.isht(x)
            x2 = self.impulse(x_isht)
            x = x1 + x2
        else:
            x = x1
        x = self.bn(x)

        if not self.is_final:
            x = F.relu(x)

        x = self.sht(x)

        return x


class Actor(nn.Module):

    def __init__(
            self,
            sphere: str,
            in_ch: int = 4,
            C: int = 32,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):

        super().__init__()

        self.verbose = False # verbose
        self.down = []
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        Y, Y_inv, area = init_spharm_basis(L, sphere, threads)
        self.isht = ISHT(Y)

        # first: transform sph harm to spherical signal
        # self.isht = ISHT(Y)
        # self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True))
        self.down.append(ConvBlock(Y, Y_inv, area, in_ch, 32, L, interval, fullband=False, is_final=False))
        # self.down.append(Down(Y, Y_inv, area, 16, 32, L, interval, fullband=False))
        self.final = ConvBlock(Y, Y_inv, area, 32, 1, L, interval, fullband=True, is_final=True)

        self.sht = SHT(L=L, Y_inv=Y_inv, area=area)  # TODO L=1 ??
        # lift signal back into harmonics domain
        self.down = nn.ModuleList(self.down)


    def forward(self, x):
        # x = self.isht(x)

        for l_idx, layer in enumerate(self.down):
            if self.verbose:
                print(f'l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            x = layer(x)
        x = self.final(x)
        # x = self.sht(x)
        x = x[:, 0] # [batchsize, 0, no_sh]

        return x


class SumActor(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__()
        
        # For debugging and testing only
        # sums all channels and returns that as action

    def forward(self, x):
        return x.sum(axis=1)

class FirstChannelActor(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        # For debugging and testing only
        # sums all channels and returns that as action

    def forward(self, x):
        return x[:, 0]

class MulActor(nn.Module):

    def __init__(
            self,
            sphere: str,
            in_ch: int = 4,
            C: int = 32,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):

        super().__init__()

        self.verbose = False # verbose
        self.down = []

        Y, Y_inv, area = init_spharm_basis(L, sphere, threads)
        self.isht = ISHT(Y)
        self.sht = SHT(L=L, Y_inv=Y_inv, area=area)

    def forward(self, x):
        x = self.isht(x)
        ac = x[:, -1]
        ac[ac < 0] = 0.
        ac[ac > 0] = 1.
        x = torch.mul(x[:, 0], x[:, -1])
        x = self.sht(x)

        return x



class CriticNN(nn.Module):
    """
    TD3
    """

    def __init__(
            self,
            sphere: str,
            in_ch: int = 3,
            C: int = 8,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):
        super().__init__()

        # self.pad_action = PadToLmax(l_in=1, l_out=L)
        self.hidden_layers = 3
        hidden_dim = 16
        self.in_ch = in_ch + 1
        self.L = L
        # input_dim = self.in_ch * 9  # TODO (why 9??)
        self.input_dim = self.in_ch * nocoeff_from_l(L)  # TODO (why 9??)

        # Q1 architecture
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(self.input_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        # action = self.pad_action(action)[:, None, :]
        q1 = torch.cat([state, action], 1)
        # q1 = q1[..., :nocoeff_from_l(self.L)]
        q1 = q1.reshape([-1, self.input_dim])

        q2 = torch.cat([state, action], 1)
        # q2 = q2[..., :nocoeff_from_l(L)]
        q2 = q2.reshape([-1, self.input_dim])

        # import ipdb; ipdb.set_trace()

        q1 = torch.relu(self.l1(q1))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(q2))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def get_q1(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        # action = self.pad_action(action)[:, None, :]
        q1 = torch.cat([state, action], 1)
        # q1 = q1[..., :9]
        q1 = q1.reshape([-1, self.input_dim])

        q1 = torch.relu(self.l1(q1))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class CriticFinal(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        x = x.view([-1, self.in_features])
        x = self.layer(x)

        return x


class Critic(nn.Module):

    def __init__(
            self,
            sphere: str,
            in_ch: int = 3,
            C: int = 8,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):
        super(Critic, self).__init__()

        # init spharm stuff
        Y, Y_inv, area = init_spharm_basis(L, sphere, threads)
        # self.isht = ISHT(Y)
        # self.sht = SHT(L=L, Y_inv=Y_inv, area=area)

        in_ch += 1 # account for action channel

        # Q1 architecture
        self.q1_conv = []
        self.q1_conv.append(ConvBlock(Y, Y_inv, area, in_ch, 32, L, interval, fullband=False, is_final=False))
        self.q1_conv.append(CriticFinal(32 * (L + 1) ** 2, 1))
        self.q1_conv = nn.ModuleList(self.q1_conv)

        # Q2 architecture
        self.q2_conv = []
        self.q2_conv.append(ConvBlock(Y, Y_inv, area, in_ch, 32, L, interval, fullband=False, is_final=False))
        self.q2_conv.append(CriticFinal(32 * (L + 1) ** 2, 1))
        self.q2_conv = nn.ModuleList(self.q2_conv)

    def forward(self, state, action):

        # Q1
        q1 = torch.cat([state, action], 1)
        for l_idx, layer in enumerate(self.q1_conv):
            q1 = layer(q1)

        # Q2
        q2 = torch.cat([state, action], 1)
        for l_idx, layer in enumerate(self.q2_conv):
            q2 = layer(q2)

        return q1, q2

    def get_q1(self, state, action):

        # Q1
        q1 = torch.cat([state, action], 1)
        for l_idx, layer in enumerate(self.q1_conv):
            q1 = layer(q1)

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

        # self.actor = Actor(sphere, in_ch, C, L, D, interval, threads, verbose).to(device)
        self.actor = Actor(sphere, in_ch, C, L, D, interval, threads, verbose).to(device)
        self.critic = CriticNN(sphere, in_ch, C, L, D, interval, threads, verbose).to(device)
        self.sph2vec = SPH2VEC().to(device)
        # how to extract direction
        sphere_vertices, _ = read_mesh(sphere)
        self.dirgetter = OdfDirGetter(sphere=sphere_vertices)  # TODO: use same sphere as self.sphere

        Y, Y_inv, area = init_spharm_basis(L, sphere, threads)
        self.isht = ISHT(Y)


    def act(
            self,
            state: torch.Tensor,
            ) -> (torch.Tensor, torch.Tensor):

        a = self.actor(state)


        # transform sph vec to cartesian direction
        # TODO: modular spherical pmf function to direction
        # but not here

        # a = self.sph2vec(a)
        # a = a.view([-1, 3])

        return a

    def select_action(self, state: np.array, stochastic=True, return_action=False) -> Tuple[np.ndarray, Any]:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = self.act(state)
        action_odf = self.isht(action).cpu().data.numpy()

        # Action to direction
        direction = self.action_to_dir(action_odf)

        if return_action:
            return direction, action.cpu().data.numpy()
        else:
            return direction, None

    def action_to_dir(self, action):
        direction = self.dirgetter.eval(action.astype(np.double))

        return direction

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
        """ Returns state dicts, so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
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

    """

    def __init__(self,
                 input_size: [int, int],
                 sphere: str,
                 in_ch: int,
                 C: int,
                 L: int,
                 D: int,
                 interval: int,
                 threads: int,
                 verbose: bool,
                 action_size: int = 3,
                 action_std: float = 0.35,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 batch_size: int = 2048,
                 rng: np.random.RandomState = None,
                 device: torch.device = "cuda:0",
                 interface_seeding: bool = 'False'):
        """
        Parameters
        ----------
        input_size: [int, int]
            Input size for the model
        sphere: str
            Path of a vtk mesh file
        action_size: int
            Output size for the actor
        action_std: float
            Standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(SPHSAC, self).__init__(input_size, action_size, action_std, lr, gamma, batch_size, interface_seeding, rng,
                                     device)

        # Initialize main policy
        self.policy = ActorCritic(sphere, in_ch, C, L, D, interval, threads, verbose)

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # SAC requires a different model for actors and critics
        # TODO
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)
        # self.actor_sheduler = ExponentialLR(self.actor_optimizer, gamma=.95)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TD3-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005
        self.noise_clip = 1

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            state_size=[in_ch, nocoeff_from_l(L)], 
            action_dim=nocoeff_from_l(L)
            )

        # Fabi spherical cnn parameters
        self.sphere = sphere
        self.in_ch = in_ch
        self.C = C
        self.L = L
        self.D = D
        self.interval = interval
        self.threads = threads
        self.verbose = verbose
        # self.sph2vec = SPH2VEC()
        self.pad_lmax = PadToLmax(l_in=1, l_out=L)

        self.rng = rng
        self.interface_seeding = interface_seeding

    def _episode(
            self,
            initial_state: np.ndarray,
            env: BaseEnv,
            ) -> Tuple[Any, float, float, float, int]:
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

        running_reward = 0.
        state = initial_state
        # tractogram = None
        done = False
        # actor_loss = 0
        # critic_loss = 0

        episode_length = 0

        running_losses = defaultdict(list)

        while not np.all(done):

            # Select action according to policy + noise for exploration
            # a, h = self.policy.select_action(np.array(state))
            # a, h = self.policy.actor(state).cpu().nunpy(), None

            direction, a = self.policy.select_action(state, return_action=True)

            # TODO: this is strange. we need a re-sampling on the sphere here
            # TODO: re-add noise here.
            action = a

            direction = (
                    direction + self.rng.normal(
                0, self.max_action * self.action_std,
                size=direction.shape)
            ).clip(-self.max_action, self.max_action)

            self.t += action.shape[0]
            
            # Perform action

            next_state, reward, done, _ = env.step(direction)
            done_bool = done

            self.replay_buffer.add(
                state, action, next_state,
                reward[..., None], done_bool[..., None])

            running_reward += sum(reward)

            # IbafServer.provide_msg({'text': f'running_reward {running_reward}'})

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:
                losses = self.update(
                    self.replay_buffer)
                running_losses = add_to_means(running_losses, losses)
            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, h, _ = env.harvest(next_state, None)  # h not used here

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
            replay_buffer: ReplayBuffer,
            batch_size: int = 2 ** 7  # was 2**12 before
            ) -> Dict[str, float]:  # Tuple[float, float]:
        """

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
            # TODO: add noise again
            next_action = self.target.actor(next_state)

            """
            next_action = self.target.actor(next_state)[:, 0]

            noise = torch.randn_like(next_action) * torch.tensor((self.action_std * 2), device=device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            # zero out degree 0 entry in noise (irrelevant)
            noise[..., 0] = 0.

            next_action = (next_action + noise
                           ).clamp(-self.max_action, self.max_action)
            """

            # Compute the target Q value for s'
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action[:, None])
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.tensor(reward + not_done * self.gamma * target_Q)

        # Get current Q estimates for s
        current_Q1, current_Q2 = self.policy.critic(
            state, action[:, None])

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
            currrent_action = self.policy.actor(state)

            actor_loss = -self.policy.critic.get_q1(state, currrent_action[:, None]).mean()

            losses['actor_loss'] = actor_loss.item()

            # Optimize the actor
            # TODO
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

        torch.cuda.empty_cache()

        return losses
