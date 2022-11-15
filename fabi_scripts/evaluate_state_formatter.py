import h5py
from os.path import join

import numpy as np
import torch

from TrackToLearn.datasets.utils import SubjectData
from TrackToLearn.environments.so3_utils.utils import (so3_format_state,
                                                       so3_test_formatter)
import nibabel as nib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_volume_from_dset(dataset_file, split_id):
    with h5py.File(
            dataset_file, 'r'
            ) as hdf_file:
        if split_id not in ['training', 'validation', 'testing']:
            split_set = hdf_file
            subject = split_id
        else:
            # import ipdb; ipdb.set_trace()
            split_set = hdf_file[split_id]
            subjects = list(split_set.keys())
            subject = subjects[0]
        tracto_data = SubjectData.from_hdf_subject(
            split_set, subject)
        tracto_data.input_dv.subject_id = subject
    input_volume = tracto_data.input_dv.data

    return input_volume.astype(np.float32), tracto_data.input_dv.affine_vox2rasmm


def get_sampling_coords(volume, no_subs):
    _x, _y, _z, _ = volume.shape
    __x = np.linspace(0, _x-1, _x*no_subs -1)
    __y = np.linspace(0, _y-1, _y*no_subs -1)
    __z = np.linspace(0, _z-1, _z*no_subs -1)

    # __x = np.linspace(0, _x-1, _x*no_subs )
    # __y = np.linspace(0, _y-1, _y*no_subs )
    # __z = np.linspace(0, _z-1, _z*no_subs )

    gx, gy, gz = np.meshgrid(__x, __y, __z, indexing='ij')
    c_list = np.stack([gx, gy, gz]).reshape(3, -1)
    c_list = np.swapaxes(c_list, 1, 0)

    return c_list[:, np.newaxis].astype(np.float32)


if __name__ == '__main__':
    dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    sub_id = 'fibercup'
    out_folder = '/fabi_project/data/scratch/state_checks'

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

    for idx in range(states.shape[1]):
        state = states[:, idx]
        state = state.reshape(dimx * no_subsampling - 1,
                              dimy * no_subsampling - 1,
                              dimz * no_subsampling - 1, 49)
        state = state[..., np.r_[0, 4:9, 16:25, 36:49]]

        img = nib.Nifti1Image(state, aff)
        nib.save(img, join(out_folder, f'state-channel_{idx}.nii.gz'))