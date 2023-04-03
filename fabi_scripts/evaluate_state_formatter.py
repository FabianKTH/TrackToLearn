import h5py
from os.path import join

import numpy as np
import torch

from TrackToLearn.datasets.utils import SubjectData
from TrackToLearn.environments.utils import format_state
from TrackToLearn.algorithms.so3_actor import SO3Actor
from TrackToLearn.environments.utils import get_neighborhood_directions

import sklearn.preprocessing as preprocessing

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
            split_set = hdf_file[split_id]
            subjects = list(split_set.keys())
            subject = subjects[0]
        # import ipdb; ipdb.set_trace()
        tracto_data = SubjectData.from_hdf_subject(
            split_set, subject)
        tracto_data.input_dv.subject_id = subject
    input_volume = tracto_data.input_dv.data

    return input_volume.astype(np.float32), tracto_data.input_dv.affine_vox2rasmm


def get_sampling_coords(volume, no_subs):
    _x, _y, _z, _ = volume.shape
    # __x = np.linspace(0, _x-1, _x*no_subs -1)
    # __y = np.linspace(0, _y-1, _y*no_subs -1)
    # __z = np.linspace(0, _z-1, _z*no_subs -1)

    __x = np.linspace(0, _x-1, _x*no_subs )
    __y = np.linspace(0, _y-1, _y*no_subs )
    __z = np.linspace(0, _z-1, _z*no_subs )

    gx, gy, gz = np.meshgrid(__x, __y, __z, indexing='ij')
    c_list = np.stack([gx, gy, gz]).reshape(3, -1)
    c_list = np.swapaxes(c_list, 1, 0)

    return c_list[:, np.newaxis].astype(np.float32)


if __name__ == '__main__':
    dset_folder = '/fabi_project/data/ttl_anat_priors/fabi_tests'
    sub_id = 'fibercup'
    out_folder = '/fabi_project/data/scratch/state_checks'

    # dset = join(dset_folder,'raw_tournier_basis',sub_id, f'{sub_id}.hdf5')
    dset = join(dset_folder,'raw',sub_id, f'{sub_id}.hdf5')
    no_subsampling = 1 # 2
    n_signal = 1
    n_dirs = 4

    # init object to acess class methods
    so3_act = SO3Actor(42, 42, 42) # dummy args

    # vol, aff = load_volume_from_dset(dset, sub_id)
    vol, aff = load_volume_from_dset(dset, 'training')

    # vol = vol[:, :vol.shape[1] //2]
    # cx, cy = 12, 17
    # vol = vol[cx-10:cx+10, cy-5:cy+5]

    nib.save(nib.Nifti1Image(vol[..., :-1], aff), join(out_folder, 'input.nii.gz'))

    dimx, dimy, dimz = vol.shape[:-1]

    # test basis conversion on input
    ## first create tensor (without mask)
    # vol_tf = so3_act._tt(vol[..., :-1])
    # vol_tf = vol_tf.reshape([-1, 1, 28])
    # vol_tf_full = so3_act._pad_antipod(vol_tf)
    # vol_tf_tour = so3_act._change_basis(vol_tf_full, 'descoteaux->spharm')

    # test direction from spharm getter
    # dir_tf = so3_act._get_direction(vol_tf_tour)
    # dir_tf = dir_tf.reshape([dimx, dimy, dimz, 3])
    # np.save('/fabi_project/data/scratch/dirgetter_chechs/d1_desc.npy', dir_tf.cpu().numpy())

    # test also spharm_to_direction
    # dir_tf = so3_act._get_direction(vol_tf_tour)
    # sph_dir = so3_act._dirs_to_shsignal(dir_tf)
    # sp_dir = so3_act._sph_to_sp(sph_dir, sph_basis='tournier')
    # sp_dir = sp_dir.reshape([dimx, dimy, dimz, 40962])
    # np.save('/fabi_project/data/scratch/dirgetter_chechs/sp_from_dir.npy', sp_dir.cpu().numpy())

    # import ipdb; ipdb.set_trace()

    # vol_tf_tour = vol_tf_tour.reshape([dimx, dimy, dimz, 49])
    # vol_full = vol_tf_tour.detach().cpu().numpy()
    # vol_tour = vol_full[..., np.r_[0, 4:9, 16:25, 36:49]]

    # nib.save(nib.Nifti1Image(vol_tour, aff), join(out_folder, 'input_newbasis2.nii.gz'))
    # import ipdb; ipdb.set_trace()
    # nib.save(nib.Nifti1Image(vol[..., :-1], np.eye(4)), join(out_folder, 'input2.nii.gz'))

    coords = get_sampling_coords(vol, no_subsampling)

    # some directions
    x, y = np.meshgrid(np.linspace(0, dimx-1, dimx), np.linspace(0, dimy-1, dimy))

    u = -(y-dimx//2)/np.sqrt((x)**2 + (y)**2)
    v = (x-dimy//2)/np.sqrt((x)**2 + (y)**2)

    u = np.repeat(u[:, :, np.newaxis], 3, axis=2)
    v = np.repeat(v[:, :, np.newaxis], 3, axis=2)

    dirs = np.stack([u, v, np.zeros_like(u)], -1).reshape([-1, 3])[:, None]
    dirs[dirs == np.inf] = 0.
    dirs[dirs == -np.inf] = 0.
    dirs = preprocessing.normalize(dirs[:, 0])[:, None]

    import ipdb; ipdb.set_trace()

    # go 1 step in uniform direction to test last_direction
    streamlines = np.concatenate((coords,
                                  coords+dirs,
                                  coords+2*dirs,
                                  coords+3*dirs,
                                  coords+4*dirs
                                  ), axis=1).astype(np.float32)
    neighborhood_directions = torch.tensor(
        get_neighborhood_directions(radius=0.25), dtype=torch.float16).to(device)

    # FORMATTER CALL
    states = format_state(streamlines, so3_act._tt(vol),
                              0.25, neighborhood_directions, n_signal, n_dirs, device)
    states = so3_act._tt(states)

    # REFORMAT
    states = so3_act._reformat_state(states)
    # states = so3_act._change_basis(states.float(), 'descoteaux->spharm')
    states = torch.reshape(states, [dimx, dimy, dimz, 11, -1])

    # states = so3_act._change_basis(states.float(), 'descoteaux->spharm')
    import ipdb; ipdb.set_trace()

    net_out = so3_act._so3_conv_net(states.float())
    dirs = so3_act._get_direction(net_out)

    states = net_out # hack, remove

    import ipdb; ipdb.set_trace()

    states = states.detach().cpu().numpy()

    for idx in range(states.shape[1]):
        state = states[:, idx]
        state = state.reshape(dimx * no_subsampling - 1,
                              dimy * no_subsampling - 1,
                              dimz * no_subsampling - 1, 49)

        np.save(join(out_folder, f'fullsig-channel_{idx}.npy'), state)

        state = state[..., np.r_[0, 4:9, 16:25, 36:49]]

        img = nib.Nifti1Image(state, aff)
        nib.save(img, join(out_folder, f'state4-channel_{idx}.nii.gz'))