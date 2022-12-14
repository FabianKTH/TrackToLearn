import h5py
from os.path import join

import numpy as np
import torch
from TrackToLearn.so3_utils.utils import (so3_format_state,
                                          so3_test_formatter)
import nibabel as nib
from fabi_scripts.evaluate_state_formatter import load_volume_from_dset, get_sampling_coords
from TrackToLearn.so3_utils.shsample import ShBasisSeqTorch

from spharmnet.core.layers import ISHT, SHT, SHConv
from spharmnet.lib.io import read_mesh
from spharmnet.lib.sphere import spharm_real, vertex_area


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Convolve(torch.nn.Module):

    def __init__(self, l_max=6, l_out=2, return_sph=True):
        super(Convolve, self).__init__()

        Y, Y_inv, area, v = self._init_y(l_max)
        self.return_sph = return_sph
        self.v = torch.as_tensor(v, device=device)

        self.isht0 = ISHT(Y)
        self.sht0 =  SHT(l_out, Y_inv, area)
        self.conv =  SHConv(2, 2, l_out, interval=1)
        self.isht1 = ISHT(Y)
        self.sht1 =  SHT(l_out, Y_inv, area)


    @staticmethod
    def _init_y(l_max):
        sphere = '/fabi_project/sphere/ico_low2.vtk'
        #  spharm boilerplate
        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, l_max, threads=1)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T
        return Y, Y_inv, area, v

    def forward(self, x):
        # return self.shconv(x)
        x = self.isht0(x)
        x = self.sht0(x)
        x = self.conv(x)
        x = self.isht1(x)

        if self.return_sph:
            x = self.sht1(x)

        return x


class GetPeaks(torch.nn.Module):

    def __init__(self, l_max=6, no_peaks=5):
        super(GetPeaks, self).__init__()

        self.no_peaks = no_peaks
        Y, Y_inv, area, v = self._init_y(l_max)
        self.v = torch.as_tensor(v, device=device)

        self.isht = ISHT(Y)

    @staticmethod
    def _init_y(l_max):
        sphere = '/fabi_project/sphere/ico_low2.vtk'
        #  spharm boilerplate
        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, l_max, threads=1)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        Y_inv = Y.T
        return Y, Y_inv, area, v

    def forward(self, x):
        x = self.isht(x)

        sorted_, idx_ = torch.sort(x, dim=1, descending=True)


        import ipdb; ipdb.set_trace()

        return x


if __name__ == '__main__':
    dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    sub_id = 'fibercup'
    out_folder = '/fabi_project/data/scratch/shconv_checks'

    dset = join(dset_folder,'raw_tournier_basis',sub_id, f'{sub_id}.hdf5')
    no_subsampling = 2 # 2
    n_signal = 1
    n_dirs = 4

    vol, aff = load_volume_from_dset(dset, sub_id)
    nib.save(nib.Nifti1Image(vol[..., :-1], aff), join(out_folder, 'input.nii.gz'))

    dimx, dimy, dimz = vol.shape[:-1]
    coords = get_sampling_coords(vol, no_subsampling)

    # FORMATTER CALL
    states = so3_format_state(coords, torch.from_numpy(vol).to(device),
                              None, None, n_signal, n_dirs, device)
    # states = so3_test_formatter(coords, torch.from_numpy(vol).to(device),
    #                           None, None, n_signal, n_dirs, device)


    # WRITE INPUT CHANNELS OUT
    for idx in range(states.shape[1]):
        state = states[:, idx]

        peak_op = GetPeaks()
        peak_op(torch.tensor(state, device=device))

        # fabi = ShBasisSeqTorch(l_max=6)
        # pts = fabi.sample_multi(torch.tensor(state, device=device))
        # np_out = pts.reshape(dimx * no_subsampling - 1,
        #                      dimy * no_subsampling - 1,
        #                      dimz * no_subsampling - 1,
        #                      2).detach().cpu().numpy()
        # np.save('../../data/scratch/shconv_checks/sampled_dirs.npy', np_out)

        state = state.reshape(dimx * no_subsampling - 1,
                              dimy * no_subsampling - 1,
                              dimz * no_subsampling - 1, 49)
        state = state[..., np.r_[0, 4:9, 16:25, 36:49]]

        img = nib.Nifti1Image(state, aff)
        nib.save(img, join(out_folder, f'input-channel_{idx}.nii.gz'))

    # CONVOLUTION
    vonco = Convolve().to(device)
    states = vonco(torch.tensor(states, device=device)).detach().cpu().numpy()

    # WRITE OUTPUT CHANNELS OUT
    for idx in range(states.shape[1]):
        state = states[:, idx]
        state = state.reshape(dimx * no_subsampling - 1,
                              dimy * no_subsampling - 1,
                              dimz * no_subsampling - 1, 9)
        # state = state[..., np.r_[0, 4:9, 16:25, 36:49]]
        state = state[..., np.r_[0, 4:9]]

        img = nib.Nifti1Image(state, aff)
        nib.save(img, join(out_folder, f'shconvout-channel_{idx}.nii.gz'))