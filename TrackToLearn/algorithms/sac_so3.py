import copy
from collections import defaultdict
from os.path import join as pjoin
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nibabel.streamlines import Tractogram
# spharm-net imports
from spharmnet.core.layers import ISHT, SHT
from spharmnet.core.models import Down, Final
from spharmnet.lib.io import read_mesh
from spharmnet.lib.sphere import spharm_real, vertex_area
from torch import nn

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.utils import add_to_means
from TrackToLearn.environments.env import BaseEnv
# from TrackToLearn.fabi_utils.communication import IbafServer
from TrackToLearn.so3_utils.rotation_utils import SPH2VEC, dirs_to_sph_channels
from TrackToLearn.so3_utils.utils import PadToLmax

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


class Actor__(nn.Module):
    """ Dummy actor for testing that only extracts the maximum from the first input channel

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
        super().__init__()

        #  spharm boilerplate
        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T

        self.v = torch.as_tensor(v, device=device)

        # transforms
        self.isht = ISHT(Y)
        self.sht = SHT(L=1, Y_inv=Y_inv, area=area)

        self.sph2vec = SPH2VEC()

    def forward(self, x):
        bs, ch, l = x.size()

        """
        s0 = self.isht(x[:, None, 0])
        s1 = self.isht(x[:, None, 3])
        s = s0 + 0.01 * s1
        y = self.sht(s)
        """

        s = self.isht(x[:, 0])
        # torch.argsort(s, dim=-1, descending=True)[:, :2]

        dirs = self.sph2vec(x[:, 3, None, :4])[:, 0]  # ugly ... but ok

        amax = torch.argmax(s, dim=-1)
        vecs = self.v[amax]

        # check if aligned with direction, else, invert (assumes antipodal symm)
        dots = torch.bmm(vecs[:, None, :].float(), dirs[..., None].float()).view(
            [bs])  # dot product between vec and dir

        signs = torch.sign(dots)
        signs[signs == 0] = 1
        vecs *= signs[..., None].expand(-1, 3)  # also not pretty :)

        y = dirs_to_sph_channels(vecs[:, None])
        # y = self.sht(s)

        return y[:, None]


class Actor__(nn.Module):
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
            interval: int = 1,
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
            print("FreqConvBlock {}\t| C:{} -> {}\t| L:{}".format(0, in_ch, out_ch, L))
        self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True))

        # encoding
        for i in range(D):
            L = L_in >> i + 1  # downsample, L_new = L+old/2
            in_ch = out_ch
            out_ch *= ch_inc
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))

        if verbose:
            print("Final\t| C:{} -> {}\t| L:{}".format(out_ch, 1, L))
        self.final = Final(Y, Y_inv, area, out_ch, 1, L, interval)

        # lift signal back into harmonics domain
        self.out = SHT(L=1, Y_inv=Y_inv, area=area)

        # lastly: mask 0 entry (irrelevant for direction) and normalize
        self.out_mask = torch.tensor([0., 1., 1., 1.], requires_grad=False, device=device)

        # sph to direction vector (currently not used.)
        # self.out_dir = SPH2VEC()

        # note: why is this required? ...
        self.down = nn.ModuleList(self.down)

    def forward(self, x):  # 'stochastic' not implemented/used
        x = self.in_(x)
        for l_idx, layer in enumerate(self.down):
            # print(f'l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            x = layer(x)
        x = self.final(x)
        x = self.out(x)
        x = self.out_mask * x
        x = F.normalize(x, dim=-1)
        # sph with L=1 to direction
        # x = self.out_dir(x)
        # x = torch.squeeze(x)

        return x


class Actor(nn.Module):

    def __init__(
            self,
            sphere: str,
            in_ch: int = 4,
            C: int = 64,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):

        super().__init__()

        # dummy actor
        # self.max_actor = MaxActor(sphere, in_ch, C, L, D, interval, threads, verbose)  TODO
        self.pad_action = PadToLmax(l_in=1, l_out=L)
        #

        self.down = []
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T

        # first: transform sph harm to spherical signal
        self.isht = ISHT(Y)
        self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True))
        self.final = Final(Y, Y_inv, area, out_ch, 1, L, interval)

        # lift signal back into harmonics domain
        self.sht = SHT(L=1, Y_inv=Y_inv, area=area)  # TODO L=1 ??

        # lastly: mask 0 entry (irrelevant for direction) and normalize
        self.out_mask = torch.tensor([0., 1., 1., 1.], requires_grad=False, device=device)

        # sph to direction vector (currently not used.)
        # self.sph2vec = SPH2VEC()

        # note: why is this required? ...
        self.down = nn.ModuleList(self.down)

    def forward(self, x):
        # x = self.max_actor(x)
        # x = self.pad_action(x)
        x = self.isht(x)

        for l_idx, layer in enumerate(self.down):
            # print(f'l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            # import ipdb; ipdb.set_trace()
            x = layer(x)
        x = self.final(x)
        x = self.sht(x)
        x = self.out_mask * x
        x = F.normalize(x, dim=-1)

        return x


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


class CriticSph(nn.Module):
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
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):
        """
        DirectMaxSPHARNNet.

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

        in_ch += 1  # account for the additional dimension for the action channel
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)

        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        Y_inv = Y.T

        # first: transform sph harm to spherical signal
        self.isht = ISHT(Y)

        # pad action (action is expected to be of size 4 for sph vec for l=1)
        self.pad_action = PadToLmax(l_in=1, l_out=L)

        # encoding
        for i in range(D):
            L = L_in >> i + 1  # downsample, L_new = L+old/2

            if verbose:
                print("(q1,q2)_down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))

            self.q1_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))
            self.q2_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))
            in_ch = out_ch
            out_ch *= ch_inc

        if verbose:
            print("(q1,q2)_final\t| C:{} -> {}\t| L:{}".format(in_ch, 1, 0))

        # lift signal back into harmonics domain
        self.sth = SHT(L=L, Y_inv=Y_inv, area=area)

        # self.q1_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)
        self.q1_final = CriticFinal(in_ch * (L + 1) ** 2, 1)
        # self.q2_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)
        self.q2_final = CriticFinal(in_ch * (L + 1) ** 2, 1)

        # note: why is this required? ...
        self.q1_down = nn.ModuleList(self.q1_down)
        self.q2_down = nn.ModuleList(self.q2_down)

    def forward(self, state, action):
        assert action.shape[-1] == 4

        # isht of state
        # state = self.in_(state)

        # pad action
        action = self.pad_action(action)[:, None, :]

        # combine state and action to input
        q1 = torch.cat([state, action], 1)
        q2 = torch.cat([state, action], 1)

        # isht of the input channels
        q1 = self.isht(q1)
        q2 = self.isht(q2)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.sth(q1)
        q1 = self.q1_final(q1)

        # q2
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q2) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q2 = layer(q2)
        q2 = self.sth(q2)
        q2 = self.q2_final(q2)

        return q1, q2

    def get_q1(self, state, action):
        assert action.shape[-1] == 4

        # pad action
        action = self.pad_action(action)[:, None, :]

        # combine state and action to input
        q1 = torch.cat([state, action], 1)

        # isth of input
        q1 = self.isht(q1)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.sth(q1)
        q1 = self.q1_final(q1)

        return q1


class Critic__(nn.Module):
    """
    sandbox testing version of the critic
    """

    def __init__(
            self,
            sphere: str,
            in_ch: int = 5,
            C: int = 8,
            L: int = 6,
            D: int = None,
            interval: int = 1,
            threads: int = 1,
            verbose: bool = False
            ):

        super().__init__()

        L_in = L

        self.q1_down = []
        self.q2_down = []

        ch_inc = 2

        in_ch += 1  # account for the additional dimension for the action channel
        out_ch = C  # Note: this is the running size of one layers output, not the net output

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T

        # first: transform sph harm to spherical signal
        self.isht = ISHT(Y)

        # pad action (action is expected to be of size 4 for sph vec for l=1)
        self.pad_action = PadToLmax(l_in=1, l_out=L)

        # encoding
        L = L_in >> 1  # downsample, L_new = L+old/2

        print("(q1,q2)_down | C:{} -> {}\t| L:{}".format(in_ch, out_ch, L))

        self.q1_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))
        self.q2_down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=False))

        if verbose:
            print("(q1,q2)_final\t| C:{} -> {}\t| L:{}".format(in_ch, 1, 0))

        # lift signal back into harmonics domain
        self.sth = SHT(L=L, Y_inv=Y_inv, area=area)

        # self.q1_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)
        self.q1_final = CriticFinal(in_ch * (L + 1) ** 2, 1)
        # self.q2_final = Final(Y, Y_inv, area, in_ch, 1, 0, interval)
        self.q2_final = CriticFinal(in_ch * (L + 1) ** 2, 1)

        # note: why is this required? ...
        self.q1_down = nn.ModuleList(self.q1_down)
        self.q2_down = nn.ModuleList(self.q2_down)

    def forward(self, state, action):
        assert action.shape[-1] == 4

        # pad action
        action = self.pad_action(action)[:, None, :]

        # combine state and action to input
        q1 = torch.cat([state, action], 1)
        q2 = torch.cat([state, action], 1)

        # isht of the input channels
        q1 = self.isht(q1)
        q2 = self.isht(q2)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.sth(q1)
        q1 = self.q1_final(q1)

        # q2
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q2) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q2 = layer(q2)
        q2 = self.sth(q2)
        q2 = self.q2_final(q2)

        return q1, q2

    def get_q1(self, state, action):
        assert action.shape[-1] == 4

        # pad action
        action = self.pad_action(action)[:, None, :]

        # combine state and action to input
        q1 = torch.cat([state, action], 1)

        # isth of input
        q1 = self.isht(q1)

        # q1
        for l_idx, layer in enumerate(self.q1_down):
            # print(f'(q1) l_idx {l_idx}, x.shape {x.shape}, x.device {x.device}')
            q1 = layer(q1)
        q1 = self.sth(q1)
        q1 = self.q1_final(q1)

        return q1


class Critic(nn.Module):
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
        super(Critic, self).__init__()

        self.pad_action = PadToLmax(l_in=1, l_out=L)
        self.hidden_layers = 3
        hidden_dim = 16
        self.in_ch = in_ch + 1
        input_dim = self.in_ch * 9  # TODO (why 9??)

        # Q1 architecture
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        action = self.pad_action(action)[:, None, :]
        q1 = torch.cat([state, action], 1)
        q1 = q1[..., :9]
        q1 = q1.reshape([-1, self.in_ch * 9])

        q2 = torch.cat([state, action], 1)
        q2 = q2[..., :9]
        q2 = q2.reshape([-1, self.in_ch * 9])

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
        action = self.pad_action(action)[:, None, :]
        q1 = torch.cat([state, action], 1)
        q1 = q1[..., :9]
        q1 = q1.reshape([-1, self.in_ch * 9])

        q1 = torch.relu(self.l1(q1))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

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

        self.actor = Actor(sphere, in_ch, C, L, D, interval, threads, verbose).to(device)
        self.critic = Critic(sphere, in_ch, C, L, D, interval, threads, verbose).to(device)
        self.sph2vec = SPH2VEC().to(device)

    def act(
            self,
            state: torch.Tensor,
            ) -> (torch.Tensor, torch.Tensor):

        a = self.actor(state)

        # transform sph vec to cartesian direction
        a = self.sph2vec(a)
        a = a.view([-1, 3])

        return a

    def select_action(self, state: np.array, stochastic=True) -> Tuple[np.ndarray, Any]:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = self.act(state)

        return action.cpu().data.numpy(), None

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
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

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
            [in_ch, (L + 1) ** 2], action_size)

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
            a, h = self.policy.select_action(np.array(state))

            # TODO: this is strange. we need a re-sampling on the sphere here

            action = (
                    a + self.rng.normal(
                0, self.max_action * self.action_std,
                size=a.shape)
            ).clip(-self.max_action, self.max_action)

            self.t += action.shape[0]
            # Perform action
            next_state, reward, done, _ = env.step(action)
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
            next_action = self.target.actor(next_state)[:, 0]

            noise = torch.randn_like(next_action) * torch.tensor((self.action_std * 2), device=device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            # zero out degree 0 entry in noise (irrelevant)
            noise[..., 0] = 0.

            next_action = (next_action + noise
                           ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value for s'
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.tensor(reward + not_done * self.gamma * target_Q)

        # Get current Q estimates for s
        current_Q1, current_Q2 = self.policy.critic(
            state, dirs_to_sph_channels(action[:, None]))

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
            currrent_action = self.policy.actor(state)[:, 0]

            actor_loss = -self.policy.critic.get_q1(state, currrent_action).mean()

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

        torch.cuda.empty_cache()

        return losses
