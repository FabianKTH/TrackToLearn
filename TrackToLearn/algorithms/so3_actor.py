import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.direction import Sphere

# spharm-net imports
from spharmnet.core.layers import ISHT, SHT, SHConv
from spharmnet.core.models import Down, Final
from spharmnet.lib.io import read_mesh
from spharmnet.lib.sphere import spharm_real, vertex_area

# user imports
from TrackToLearn.algorithms.so3_helper import _init_antipod_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = -10 # (2)
LOG_STD_MIN = -20 # (-20)


class SO3Actor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
    ):

        super(SO3Actor, self).__init__()

        self.l_max = 6
        self._antipod_dict = _init_antipod_dict(l_max=self.l_max)
        self.antipod_idx = self._tt(np.array(self._antipod_dict[self.l_max]),
                                    dtype=torch.int64)

        self.callcount = 0 # TODO remove, this is really dirty

        self._init_spharm_basis() # inits self.Y, self.Y_inv, self.area
        self._init_spharm_conv() # inits self.isht, self.sht


    @staticmethod
    def _tt(x, dtype=torch.float32):
        """
        tt: to torch, shorthand for converting numpy to torch
        """

        return torch.from_numpy(x).to(device=device, dtype=dtype)

    def _init_spharm_basis(self):
        sphere_dir = '/fabi_project/sphere/ico4.vtk' # TODO: dont code so hard (ico_low2.vtk)

        v, f = read_mesh(sphere_dir)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, self.l_max, threads=1)
        self.area = self._tt(area)
        self.B_spha = self._tt(Y)
        self.v = self._tt(v)
        self.invB_spha = self.B_spha.T
        self.sphere = Sphere(xyz=v)
        self._init_dipy_basis()


    def _init_dipy_basis(self):
        B_desc, invB_desc = sh_to_sf_matrix(self.sphere, sh_order=self.l_max,
                                            basis_type='descoteaux07', full_basis=True,
                                            legacy=False, return_inv=True, smooth=0)

        B_tour, invB_tour = sh_to_sf_matrix(self.sphere, sh_order=self.l_max,
                                            basis_type='tournier07', full_basis=True,
                                            legacy=False, return_inv=True, smooth=0)

        self.B_desc, self.invB_desc = self._tt(B_desc), self._tt(invB_desc)
        self.B_tour, self.invB_tour = self._tt(B_tour), self._tt(invB_tour)


    def _init_spharm_conv(self):

        """
        self.in_ch = 2 # 7 + 4 # 1 center + 6 neighbours + 4 prev directions
        self.isht = ISHT(self.B_spha)
        self.sht = SHT(L=self.l_max, Y_inv=self.invB_spha, area=self.area)
        
        self.down = []
        self.down.append(ConvBlock(self.B_spha, self.invB_spha, self.area, self.in_ch, 16,
                                   self.l_max, interval=1, fullband=False, is_final=False))

        # note, 2 output channels, one for mu, one for log_std
        self.final = ConvBlock(self.B_spha, self.invB_spha, self.area, 16, 2, self.l_max,
                               interval=1, fullband=True, is_final=True)

        self.down = nn.ModuleList(self.down)
        self.output_activation = nn.Tanh()
        """


        # below, new
        self.isht = ISHT(self.B_spha)
        self.sht = SHT(L=self.l_max, Y_inv=self.invB_spha, area=self.area)
        self.down = []
        self.down.append(Down(self.B_spha, self.invB_spha, self.area, 2, 32, L=self.l_max,
                              interval=1, fullband=True))
        self.final = Final(self.B_spha, self.invB_spha, self.area, 2, 2, L=self.l_max,
                           interval=1)
        self.down = nn.ModuleList(self.down)
        self.output_activation = nn.Tanh()


    # @staticmethod
    def _reformat_state(self, state):
        """
        takes state from original TTL and converts it to a spherical harmonics representation
        """
        sh_end = 7 * 29 # 7 neigbourhood_size * (28 shcoeff + 1 mask)
        sh_part, dir_part = state[:, :sh_end], state[:, sh_end:]

        # sh_part
        sh_part = sh_part.reshape([-1, 7, 29])
        sh_part = self._remove_mask(sh_part)
        sh_part = self._pad_antipod(sh_part)
        # sh_part = self._change_basis(sh_part, conversion='descoteaux->spharm')
        sh_part = self._sph_to_sp(sh_part, sph_basis='descoteaux')

        # dir_part
        dir_part = dir_part.reshape([-1, 4, 3]) # 4 directions, 3 point coefficients
        dir_part = self._dirs_to_shsignal(dir_part, self.l_max)
        dir_part = self._sph_to_sp(dir_part, sph_basis='spharm')

        # concat channels
        out = torch.cat([sh_part, dir_part], 1)

        # normalize (TODO: spharm normalization instead of vector norm)
        # out = torch.nn.functional.normalize(out, dim=-1)

        return out


    @staticmethod
    def _remove_mask(signal):
        """
        removes the tracking mask part from the sh-signal
        """
        return signal[..., :-1]


    def _dirs_to_shsignal(self, dirs, l_max=6):
        """
        converts direction vectors to sh-signals with maximum at direction
        """
        assert dirs.shape[-1] == 3

        N, no_channels, P = dirs.shape

        sh_ = torch.zeros(size=[N, no_channels, self.nocoeff_from_l(l_max)], device=device)
        sh_[..., 1] = dirs[..., 1]
        sh_[..., 2] = dirs[..., 2]
        sh_[..., 3] = dirs[..., 0]

        return sh_


    @staticmethod
    def nocoeff_from_l(l_max):
        return (l_max + 1) ** 2


    def _pad_antipod(self, signal):
        """
        takes antipodal spharm signal and fills up 0 to shape of full spectrum spharm signal
        # TODO check and test
        """

        N, no_channels, no_coeff = signal.shape

        # zero-padding for all even degree sph harm (antipodal -> podal)
        # l_max = -1.5 + np.sqrt(0.25 + 2 * no_coeff)  # from no_sph = (l + 1)(l/2 + 1)
        # antipod_idx = self._antipod_idx # self._antipod_dict[int(l_max)]
        new_no_coeff = len(self.antipod_idx)
        idx_expanded = self.antipod_idx.expand([N, no_channels, new_no_coeff])
        signal = torch.nn.functional.pad(signal, (0, new_no_coeff - no_coeff))
        signal = torch.gather(signal.view([-1, new_no_coeff]),
                              1,
                              idx_expanded.view([-1, new_no_coeff])
                              ).view(N, no_channels, new_no_coeff)

        return signal


    def _change_basis(self, signal, conversion='tournier->descoteaux'):
        if conversion == 'tournier->descoteaux':
            return torch.matmul(torch.matmul(signal, self.B_tour), self.invB_desc)
        elif conversion == 'descoteaux->tournier':
            return torch.matmul(torch.matmul(signal, self.B_desc), self.invB_tour)
        elif conversion == 'tournier->spharm':
            # return torch.matmul(torch.matmul(signal, self.B_tour), self.invB_spha)
            return self.sht(torch.matmul(signal, self.B_tour))
        elif conversion == 'descoteaux->spharm':
            # return torch.matmul(torch.matmul(signal, self.B_desc), self.invB_spha)
            return self.sht(torch.matmul(signal, self.B_desc))
        elif conversion == 'spharm->tournier':
            # return torch.matmul(torch.matmul(signal, self.B_spha), self.invB_tour)
            return torch.matmul(self.isht(signal), self.invB_tour)
        elif conversion == 'spharm->descoteaux':
            # return torch.matmul(torch.matmul(signal, self.B_spha), self.invB_desc)
            return torch.matmul(self.isht(signal), self.invB_desc)
        else:
            return ValueError


    def _sph_to_sp(self, signal, sph_basis='tournier'):
        if sph_basis == 'tournier':
            return torch.matmul(signal, self.B_tour)
        elif sph_basis == 'descoteaux':
            return torch.matmul(signal, self.B_desc)
        elif sph_basis == 'spharm':
            # return torch.matmul(signal, self.B_harm)
            return self.isht(signal)
        else:
            return ValueError


    def _sp_to_sph(self, signal, sph_basis='tournier'):
        if sph_basis == 'tournier':
            return torch.matmul(signal, self.invB_tour)
        elif sph_basis == 'descoteaux':
            return torch.matmul(signal, self.invB_desc)
        elif sph_basis == 'spharm':
            return self.sht(signal)
        else:
            return ValueError


    def _get_direction(self, signal, is_sp_signal=False):
        """
        expects output sh-signals, samples/extracts directions.
        """

        # check if signal is already in spherical domain (not spharm)
        if not is_sp_signal:
            odf = self.isht(signal)
        else:
            odf = signal

        peak_idx = torch.argmax(odf, dim=-1)

        peak_dir = self.v[peak_idx]

        # import ipdb; ipdb.set_trace()

        return peak_dir


    def _so3_conv_net(self, signal, test=False):

        # TODO: remove this!!
        # if self.callcount < 5000:
        #     self.callcount += 1
        #     test=True
        # if self.callcount == 4999:
        #     print('[!!] network starts')

        x = signal

        if not test:
            x = torch.cat([x[:, 0, None], x[:, -1, None]], 1) # extract only first and last

            for l_idx, layer in enumerate(self.down):
                x = layer(x)

            x = self.final(x)

        else:
            # next line dummy
            x = self.isht(x)
            dir_mask = (x[:, -1, None] > -.05).int()
            x = x[:, 0, None] * dir_mask
            x = self.sht(x)
            x = torch.cat([x, x], 1)

        return x


    def _so3_conv_net2(self, signal, test=False):
        x = signal

        if test:
            dir_mask = (x[:, -1, None] > -.05).int()
            x = x[:, 0, None] * dir_mask
            # x = self.sht(x)
            x = torch.cat([x, x], 1)

        else:
            x = torch.cat([x[:, 0, None], x[:, -1, None]], 1) # extract only first and last
            # for l_idx, layer in enumerate(self.down):
            #     x = layer(x)
            x = self.final(x)

        return x


    def forward(
        self,
        state: torch.Tensor,
        stochastic: bool,
        with_logprob: bool = False,
    ) -> (torch.Tensor, torch.Tensor):

        # extract state data
        state = self._reformat_state(state)

        # convert state to spharm basis
        # state = self._change_basis(state, 'descoteaux->spharm')

        # below: test, TODO remove
        # stochastic=False
        # state = self._change_basis(state, 'tournier->spharm') # should be descoteaux->spharm

        p = self._so3_conv_net2(state, test=False)
        p = self._get_direction(p, is_sp_signal=True)

        mu = p[:, 0]
        log_std = p[:, 1]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = self.output_activation(pi_action)

        return pi_action, logp_pi


    def logprob(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:

        p = self._so3_conv_net(state)
        mu = p[:, 0]
        log_std = p[:, 1]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - action -
                       F.softplus(-2*action))).sum(axis=1)

        return logp_pi


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

