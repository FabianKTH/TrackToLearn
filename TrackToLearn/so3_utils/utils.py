import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from TrackToLearn.so3_utils.rotation_utils import dirs_to_sph_channels
from TrackToLearn.environments.utils import torch_trilinear_interpolation


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

    _antipod_dict = dict()
    for l in range(0, l_max + 1, 2):
        _antipod_dict[l] = _antipod_lmax(l)

    return _antipod_dict


antipod_dict = _init_antipod_dict(8)  # initialize the dict to have it pre-cached here


def so3_format_state(
        streamlines: np.ndarray,
        data_volume,
        add_neighborhood_vox,
        neighborhood_directions,
        n_signal,
        n_dirs,
        device
        ) -> np.ndarray:
    """
    From the last streamlines coordinates, extract the corresponding
    SH coefficients

    Parameters
    ----------
    streamlines: `numpy.ndarry`
        Streamlines from which to get the coordinates
    data_volume:
    n_signal:
    n_dirs:
    device:

    Returns
    -------
    signal: `numpy.ndarray`
        SH coefficients at the coordinates
    """
    # first: disregard mask in last dimension
    data_volume = data_volume[..., :-1]

    N, L, P = streamlines.shape

    segments = streamlines[:, :-(n_signal + 1):-1, :]

    # if L >= 2:
    #     IbafServer.provide_msg({'tract': streamlines})

    previous_dirs = np.zeros((N, n_dirs, P), dtype=np.float32)
    if L > 1:
        dirs = streamlines[:, 1:, :] - streamlines[:, :-1, :]
        previous_dirs[:, :min(dirs.shape[1], n_dirs), :] = \
            dirs[:, :-(n_dirs + 1):-1, :]

    # import ipdb; ipdb.set_trace()

    # fabi call
    inputs = get_sph_channels(
        segments,
        data_volume,
        previous_dirs,
        device=device
        ).cpu().numpy()

    return inputs


def get_sph_channels(
        segments,
        data_volume,
        previous_dirs,
        no_channels=1,
        neighb_cube_dim=3,
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

    ### dummy hack
    """
    lower = torch.as_tensor([0, 0, 0]).to(device)
    upper = (torch.as_tensor(data_volume.shape[:-1]) - 1).to(device)
    centers_clipped = torch.min(torch.max(centers, lower), upper)
    c2 = torch.clone(centers_clipped).long()
    dummy_first_channel = data_volume[
        c2[:, 0],
        c2[:, 1],
        c2[:, 2]
        ]
    """
    ### end

    centers = torch.repeat_interleave(centers, no_neighb, dim=0)
    coords = centers + neighb_indices.repeat(N, 1)
    distances = torch.norm(centers - coords.long(), dim=1)  # TODO: check if rounding right here.

    no_sph_coeff = data_volume.shape[-1]  # TODO, should be actually to big by 1 (for mask)
    ring_radii = torch.repeat_interleave(ring_radii, coords.size()[0])

    # try:
    dist = Normal(ring_radii, .2)
    weights = torch.exp(dist.log_prob(distances.repeat(no_channels)))
    # except ValueError:
    #     print('ValueError in distrib')

    lower = torch.as_tensor([0, 0, 0]).to(device)
    upper = (torch.as_tensor(data_volume.shape[:-1]) - 1).to(device)
    coords_clipped = torch.min(torch.max(coords, lower), upper)

    # trick: set the weights of all clipped points to zero (to prevent double counting)
    weight_mask = torch.all(coords_clipped == coords, dim=1).repeat(no_channels)
    weights = torch.where(weight_mask, weights, torch.Tensor([0.]).to(device))
    # coords_clipped = coords_clipped.round().long()  # TODO: round? should match better to convention...
    coords_clipped = coords_clipped.long()
    data = data_volume[coords_clipped[:, 0], coords_clipped[:, 1], coords_clipped[:, 2]]

    # scale according to weights
    scaled = data.repeat(no_channels, 1) * weights[:, None]
    # scaled = scaled.view(no_neighb, N, no_channels, no_sph_coeff) # !! just wrong
    scaled = scaled.view(no_channels, N, no_neighb, no_sph_coeff)

    # scaled = sort_interleaved(scaled, no_channels, N)
    coeff_channels = torch.sum(scaled, dim=2)  # sum over all neighbours
    coeff_channels = coeff_channels.swapaxes(0, 1)  # no_channels <=> N

    ### dummy hack
    # coeff_channels[:, 0] = dummy_first_channel
    ### end

    coeff_channels = assemble_channels(coeff_channels, previous_dirs, N, no_channels, no_sph_coeff, device)

    # normalize all spherical functions
    coeff_channels = torch.nn.functional.normalize(coeff_channels, dim=-1)

    return coeff_channels

def so3_test_formatter(
        streamlines: np.ndarray,
        data_volume,
        add_neighborhood_vox,
        neighborhood_directions,
        n_signal,
        n_dirs,
        device
        ) -> np.ndarray:
    N, L, P = streamlines.shape

    if N <= 0:
        return []

    segments = streamlines[:, -1, :][:, None, :]
    _, H, _ = segments.shape
    flat_coords = np.reshape(segments, (N * H, P))

    coords = torch.as_tensor(flat_coords, device=device)
    n_coords = coords.shape[0]

    # ! drop the mask in the last dim
    data_volume = data_volume[..., :-1]

    partial_signal = torch_trilinear_interpolation(
        data_volume,
        coords).type(torch.float32)

    previous_dirs = np.zeros((N, n_dirs, P), dtype=np.float32)
    if L > 1:
        dirs = streamlines[:, 1:, :] - streamlines[:, :-1, :]
        previous_dirs[:, :min(dirs.shape[1], n_dirs), :] = \
            dirs[:, :-(n_dirs + 1):-1, :]

    coeff_channels = assemble_channels(partial_signal, previous_dirs, N, no_channels=1,
                                       no_sph_coeff=data_volume.shape[-1], device=device)

    return coeff_channels.cpu().numpy()


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


class PadToLmax(nn.Module):
    """
    PadToLmax
    appends zeros to last dimesion in order to extend a sph coeff vector up to l_max

    """

    def __init__(self, l_in=1, l_out: int = 6):
        super().__init__()
        no_sphin = (l_in + 1) ** 2
        no_sphout = (l_out + 1) ** 2

        self.pad = [0, no_sphout - no_sphin]

    def forward(self, x):
        return nn.functional.pad(x, self.pad, value=0.)
