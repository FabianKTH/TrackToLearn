from enum import Enum

import numpy as np
import torch
from scipy.ndimage.interpolation import map_coordinates
from torch.distributions.normal import Normal

from TrackToLearn.environments.rotation_utils import dirs_to_sph_channels


def _antipod_lmax(l_max):
    assert not l_max % 2, f'l_max {l_max} not even!'

    idx_list = list()
    m = (l_max + 1) * (l_max // 2 + 1)
    n = (l_max + 1) ** 2

    even_idx = list(range(m))
    odd_idx = list(range(m, n))

    # print(f'l_max: {l_max}')
    # print(f'm: {m}')

    for l in range(l_max + 1):
        k = 2 * l + 1
        # n = (l)**2

        # print(idx_list)

        if not l % 2:
            # even
            # l_even += 1
            # idx_list.extend(list(range( int(n), int(n+k) )))
            for _ in range(k):
                idx_list.append(even_idx.pop(0))
        else:
            # l_odd += 1
            # idx_list.extend(list(range( int(n+m), int(n+m+k) )))
            for _ in range(k):
                idx_list.append(odd_idx.pop(0))

    return idx_list


def _init_antipod_dict(l_max=8):
    assert not l_max % 2, f'l_max {l_max} not even!'

    antipod_dict = dict()
    for l in range(0, l_max + 1, 2):
        antipod_dict[l] = _antipod_lmax(l)

    return antipod_dict


# globals that hold indice mappings etc
B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=np.float)

idx = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], dtype=np.float)

antipod_dict = _init_antipod_dict(8)  # initialize the dict to have it pre-cached here


def get_sph_channels(
        segments,
        data_volume,
        previous_dirs,
        no_channels=3,
        neighb_cube_dim=5,
        device=torch.device("cuda")):
    N, H, P = segments.shape
    # t_ = torch.arange(0, neighb_cube_dim)
    t_ = torch.arange(0, neighb_cube_dim) - (neighb_cube_dim-1)/2
    ring_radii = torch.arange(0, neighb_cube_dim, neighb_cube_dim / no_channels).to(device)

    neighb_indices = torch.moveaxis(torch.stack(torch.meshgrid(t_, t_, t_)), 0, -1).view(-1, 3).to(
        device)
    no_neighb = neighb_indices.size()[0]

    flat_centers = np.concatenate(segments, axis=0)
    centers = torch.as_tensor(flat_centers).to(device)
    no_centers = centers.shape[0]
    centers = torch.repeat_interleave(centers, no_neighb, dim=0)
    coords = centers + neighb_indices.repeat(no_centers, 1)
    distances = torch.norm(centers - coords.round(), dim=1)  # TODO: check if rounding right here.

    no_sph_coeff = data_volume.shape[-1]
    ring_radii = torch.repeat_interleave(ring_radii, coords.size()[0])

    # try:
    dist = Normal(ring_radii, 1.)
    weights = torch.exp(dist.log_prob(distances.repeat(no_channels)))
    # except ValueError:
    #     print('ValueError in distrib')

    lower = torch.as_tensor([0, 0, 0]).to(device)
    upper = (torch.as_tensor(data_volume.shape[:-1]) - 1).to(device)
    coords_clipped = torch.min(torch.max(coords, lower), upper)

    # trick: set the weights of all clipped points to zero (to prevent double counting)
    weight_mask = torch.all(coords_clipped == coords, dim=1).repeat(no_channels)
    weights = torch.where(weight_mask, weights, 0.)
    coords_clipped = coords_clipped.round().long()  # TODO: round? should match better to convention...
    data = data_volume[coords_clipped[:, 0], coords_clipped[:, 1], coords_clipped[:, 2]]

    # scale according to weights
    scaled = data.repeat(no_channels, 1) * weights[:, None]

    scaled = scaled.view(no_neighb, N, no_channels, no_sph_coeff)

    # scaled = sort_interleaved(scaled, no_channels, N)

    coeff_channels = torch.sum(scaled, dim=0)

    coeff_channels = assemble_channels(coeff_channels, previous_dirs, N, no_channels, no_sph_coeff, device)

    return coeff_channels


def assemble_channels(coeff_channels, previous_dirs, N, no_channels, no_sph_coeff, device):
    """
    does all the re-ordering and adds directional channel
    """

    # zero-padding for all even degree sph harm (podal <-> antipodal)
    l_max = -1.5 + np.sqrt(0.25 + 2 * no_sph_coeff)  # from no_sph = (l + 1)(l/2 + 1)
    antipod_idx = antipod_dict[int(l_max)]
    new_no_coeff = len(antipod_idx)
    idx_expanded = torch.tensor(antipod_idx).to(device).expand([N, no_channels, new_no_coeff])
    coeff_channels = torch.nn.functional.pad(coeff_channels, (0, new_no_coeff - no_sph_coeff))
    coeff_channels = torch.gather(coeff_channels.view([-1, new_no_coeff]),
                                  1,
                                  idx_expanded.view([-1, new_no_coeff])
                                  ).view(N, no_channels, new_no_coeff)

    # add also the additional directional channel
    dir_channel = dirs_to_sph_channels(previous_dirs)

    # pad also the directional component
    dir_channel = torch.nn.functional.pad(dir_channel, (0, new_no_coeff - dir_channel.size(-1)))

    # clear nans (e.g. from first iteration)
    dir_channel = torch.nan_to_num(dir_channel)

    if torch.any(torch.abs(dir_channel) > 1000.):
        print('in err')

    # combine channels to form input
    coeff_channels = torch.cat([coeff_channels, dir_channel[:, None]], dim=1)

    return coeff_channels


def sort_interleaved(scaled, no_channels, N):
    # split into individual channels
    scaled = torch.stack(
        torch.split(scaled, no_channels))  # check if no_channels TODO: propably slow exec. better reshape
    # split into individual batches
    scaled = torch.stack(torch.split(scaled, N))  # check if no_channels
    return scaled


# Flags enum
class StoppingFlags(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop
    """
    STOPPING_MASK = int('00000001', 2)
    STOPPING_LENGTH = int('00000010', 2)
    STOPPING_CURVATURE = int('00000100', 2)
    STOPPING_TARGET = int('00001000', 2)
    STOPPING_LOOP = int('00010000', 2)


def get_sh(
        segments,
        data_volume,
        add_neighborhood_vox,
        neighborhood_directions,
        history,
        device
        ) -> torch.Tensor:
    """ Get the sh coefficients at the end of streamlines
    """

    N, H, P = segments.shape
    flat_coords = np.concatenate(segments, axis=0)

    coords = torch.as_tensor(flat_coords).to(device)
    n_coords = coords.shape[0]

    if add_neighborhood_vox:
        # Extend the coords array with the neighborhood coordinates
        coords = torch.repeat_interleave(coords,
                                         neighborhood_directions.size()[0], dim=0)  # before: axis instead dim

        coords[:, :3] += \
            neighborhood_directions.repeat(n_coords, 1)

        # Evaluate signal as if all coords were independent
        partial_signal = torch_trilinear_interpolation(
            data_volume, coords)

        # Reshape signal into (n_coords, new_feature_size)
        new_feature_size = partial_signal.size()[-1] * neighborhood_directions.size()[0]
    else:
        partial_signal = torch_trilinear_interpolation(
            data_volume,
            coords).type(torch.float32)
        new_feature_size = partial_signal.size()[-1]

    signal = torch.reshape(partial_signal, (N, history * new_feature_size))

    assert len(signal.size()) == 2, signal.size()

    return signal


def torch_trilinear_interpolation(
        volume: torch.Tensor,
        coords: torch.Tensor,
        ) -> torch.Tensor:
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords : torch.Tensor with shape (N,3)
        The coordinates where to interpolate

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    # Get device, and make sure volume and coords are using the same one
    assert volume.device == coords.device, "volume on device: {}; " \
                                           "coords on device: {}".format(volume.device, coords.device)
    coords = coords.type(torch.float32)
    volume = volume.type(torch.float32)

    device = volume.device
    local_idx = idx[:]

    # Send data to device
    idx_torch = torch.as_tensor(local_idx, dtype=torch.float, device=device)
    B1_torch = torch.as_tensor(B1, dtype=torch.float, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.dim() == 3:
        # torch needs indices to be cast to long
        indices_unclipped = (
                coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0]).to(device)
        upper = (torch.as_tensor(volume.shape) - 1).to(device)
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ].reshape((coords.shape[0], -1)).t()

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy, dy * dz, dx * dz, dx * dy * dz], dim=0)
        output = torch.sum(P * torch.mm(B1_torch.t(), Q1), dim=0)

        return output

    if volume.dim() == 4:
        # 8 coordinates of the corners of the cube, for each input coordinate
        indices_unclipped = torch.floor(
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0], device=device)
        upper = torch.as_tensor(volume.shape[:3], device=device) - 1
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[indices[:, 0], indices[:, 1], indices[:, 2], :].reshape(
            (coords.shape[0], 8, volume.shape[-1]))

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy, dy * dz, dx * dz, dx * dy * dz], dim=0)
        output = torch.sum(
            P * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output.type(torch.float32)

    raise ValueError(
        "There was a problem with the volume's number of dimensions!")


def interpolate_volume_at_coordinates(
        volume: np.ndarray,
        coords: np.ndarray,
        mode: str = 'nearest',
        order: int = 1
        ) -> np.ndarray:
    """ Evaluates a 3D or 4D volume data at the given coordinates using trilinear
    interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’).
        Default is ‘nearest’.
        ('constant' uses 0.0 as a points outside the boundary)
    order : int, optional
        Order of interpolation

    Returns
    -------
    output : 2D array
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=order, mode=mode)

    if volume.ndim == 4:
        D = volume.shape[-1]
        values_4d = np.zeros((coords.shape[0], D))
        for i in range(volume.shape[-1]):
            values_4d[:, i] = map_coordinates(
                volume[..., i], coords.T, order=order, mode=mode)
        return values_4d


def get_neighborhood_directions(
        radius: float
        ) -> np.ndarray:
    """ Returns predefined neighborhood directions at exactly `radius` length
        For now: Use the 6 main axes as neighbors directions, plus (0,0,0)
        to keep current position

    Parameters
    ----------
    radius : float
        Distance to neighbors

    Returns
    -------
    directions : `numpy.ndarray` with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space
    """
    axes = np.identity(3)
    directions = np.concatenate(([[0, 0, 0]], axes, -axes)) * radius
    return directions


def is_inside_mask(
        streamlines: np.ndarray,
        mask: np.ndarray,
        affine_vox2mask: np.ndarray = None,
        threshold: float = 0.
        ):
    """ Checks which streamlines have their last coordinates inside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
        NOTE: The mask coordinates can be in a different space than the
        streamlines coordinates if an affine is provided.
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is inside the mask
        or not.
    """
    # Get last streamlines coordinates
    indices_vox = streamlines[:, -1, :]

    mask_values = interpolate_volume_at_coordinates(
        mask, indices_vox, mode='constant')
    inside = mask_values >= threshold

    return inside


def is_outside_mask(
        streamlines: np.ndarray,
        mask: np.ndarray,
        affine_vox2mask: np.ndarray = None,
        threshold: float = 0.
        ):
    """ Checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
        NOTE: The mask coordinates can be in a different space than the
        streamlines coordinates if an affine is provided.
        Tranformation that aligns streamlines on top of `mask`.
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is outside the
        mask or not.
    """

    # Get last streamlines coordinates
    indices_vox = streamlines[:, -1, :]

    mask_values = interpolate_volume_at_coordinates(
        mask, indices_vox, mode='constant')
    outside = mask_values < threshold

    return outside


def is_too_long(streamlines: np.ndarray, max_nb_steps: int):
    """ Checks whether streamlines have exceeded the maximum number of steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_nb_steps : int
        Maximum number of steps a streamline can have

    Returns
    -------
    too_long : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too long or not
    """
    return (np.full(len(streamlines), True)
            if streamlines.shape[1] >= max_nb_steps
            else np.full(len(streamlines), False))


def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """ Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
        each other.

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian
    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        return np.zeros(len(streamlines), dtype=np.uint8)

    # Compute vectors for the last and before last streamline segments
    u = streamlines[:, -1] - streamlines[:, -2]
    v = streamlines[:, -2] - streamlines[:, -3]

    # Normalize vectors
    u /= np.sqrt(np.sum(u ** 2, axis=1, keepdims=True))
    v /= np.sqrt(np.sum(v ** 2, axis=1, keepdims=True))

    # Compute angles
    cos_theta = np.sum(u * v, axis=1).clip(-1., 1.)
    angles = np.arccos(cos_theta)

    return angles > max_theta_rad


def winding(nxyz: np.ndarray) -> np.ndarray:
    """ Project lines to best fitting planes. Calculate
    the cummulative signed angle between each segment for each line
    and their previous one

    Adapted from dipy.tracking.metrics.winding to allow multiple
    lines that have the same length

    Parameters
    ------------
    nxyz : np.ndarray of shape (N, M, 3)
        Array representing x,y,z of M points in N tracts.

    Returns
    ---------
    a : np.ndarray
        Total turning angle in degrees for all N tracts.
    """

    directions = np.diff(nxyz, axis=1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    thetas = np.einsum(
        'ijk,ijk->ij', directions[:, :-1], directions[:, 1:]).clip(-1., 1.)
    shape = thetas.shape
    rads = np.arccos(thetas.flatten())
    turns = np.sum(np.reshape(rads, shape), axis=-1)
    return np.rad2deg(turns)

    # # This is causing a major slowdown :(
    # U, s, V = np.linalg.svd(nxyz-np.mean(nxyz, axis=1, keepdims=True), 0)

    # Up = U[:, :, 0:2]
    # # Has to be a better way than iterare over all tracts
    # diags = np.stack([np.diag(sp[0:2]) for sp in s], axis=0)
    # proj = np.einsum('ijk,ilk->ijk', Up, diags)

    # v0 = proj[:, :-1]
    # v1 = proj[:, 1:]
    # v = np.einsum('ijk,ijk->ij', v0, v1) / (
    #     np.linalg.norm(v0, axis=-1, keepdims=True)[..., 0] *
    #     np.linalg.norm(v1, axis=-1, keepdims=True)[..., 0])
    # np.clip(v, -1, 1, out=v)
    # shape = v.shape
    # rads = np.arccos(v.flatten())
    # turns = np.sum(np.reshape(rads, shape), axis=-1)

    # return np.rad2deg(turns)


def is_looping(streamlines: np.ndarray, loop_threshold: float):
    """ Checks whether streamlines are looping

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    loop_threshold: float
        Maximum angle in degrees for the whole streamline

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """

    angles = winding(streamlines)

    return angles > loop_threshold


def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value
    return ((flags.astype(np.uint8) & ref_flag) >>
            np.log2(ref_flag).astype(np.uint8)).astype(bool)


def count_flags(flags, ref_flag):
    """ Counts how many flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value
    return is_flag_set(flags, ref_flag).sum()
