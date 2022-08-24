import numpy as np
import torch

from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
from pyshtools.shio import SHVectorToCilm, SHCilmToVector
from pyshtools.rotate import SHRotateRealCoef, djpi2
from pyshtools.expand import SHExpandDH

import TrackToLearn.environments.kernels as kernels


# cpy paste from https://github.com/numpy/numpy/issues/5228
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)

    return az, el, r


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)

    return x, y, z


def rotate_sph(x, euler_ang, l_max=6):
    x = SHVectorToCilm(x)
    dj = djpi2(l_max)
    res = SHRotateRealCoef(x, euler_ang, dj, l_max)
    res = SHCilmToVector(res)

    return res


def get_directional_sph(l_max=6, device=torch.device("cuda")):
    # ker = kernels.von_misses_fisher()
    # ker = kernels.half_cosine()
    # sig = SHCilmToVector(SHExpandDH(ker), lmax=l_max)

    sig = torch.zeros(4).to(device)
    sig[1] = 1.

    return sig


def get_rotated_directional_sph(rotmat, l_max=6):
    sph = get_directional_sph(l_max)
    rot_sph = rotate_sph(sph, rotmat, l_max)

    return rot_sph


def rot_from_dir(dir_vecs, device=torch.device('cuda')):
    """
    converts a new direction vector to a rotation matrix between the given
    direction and [0, 1, 0]
    """

    N, P = dir_vecs.shape

    init_ = torch.tensor([0, 1, 0]).repeat(N, 1).to(device)

    # adapted from https://math.stackexchange.com/a/476311
    v = np.cross(init_, dir_vecs)

    if np.all(v == 0):
        return np.eye(3)

    s = np.linalg.norm(v)
    c = np.dot(init_, dir_vecs)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c)/(s**2)
    ret = r

    return ret


def dir_to_sph(dir_vec, l_max=6):
    euler_ang = rot_from_dir(dir_vec)
    res = get_rotated_directional_sph(euler_ang, l_max)

    return res


def dirs_to_sph_channels(dir_vecs, device=torch.device("cuda")):
    """
    expects an array of 3D direction coordinates (in last dimension) and returns
    corresponding spherical coefficients
    """
    assert dir_vecs.shape[-1] == 3

    dir_vecs = torch.tensor(dir_vecs[:, 0]).to(device)  # TODO: bit of a hack

    N, P = dir_vecs.shape
    rmats = rotmats_from_dirs(dir_vecs, device)

    # reorder for sh rotmat (already multiplied with sh [0, 0, 1, 0, ...])
    indices = torch.tensor([3, 6, 0]).to(device)
    sh_l1 = torch.index_select(rmats.view([-1, 9]), 1, indices)

    # fill up with zeros for sh coeff of other orders than 1
    sh = torch.cat([torch.zeros(N, 1).to(device), sh_l1], dim=1)
    # TODO: also append zeros for up to l_max

    return sh


def rotmats_from_dirs(dir_vecs, device=torch.device("cuda")):
    # adapted from https://math.stackexchange.com/a/476311

    N, P = dir_vecs.shape
    # add zero before first coordinate
    d0 = torch.cat([torch.zeros(N, 1).to(device), dir_vecs], dim=1)
    # reorder vectors into matrix structure
    indices = torch.tensor([0, 3, 2, 3, 0, 1, 2, 1, 0]).to(device)
    rmat_vals = torch.index_select(d0, 1, indices).view([-1, 3, 3])
    rmat_signs = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]]).to(device)
    rmats = rmat_vals * rmat_signs
    # cross product
    init_ = torch.tensor([0., 1., 0.]).to(device)
    v = torch.cross(init_.repeat(N, 1), dir_vecs)
    # norm
    s = torch.norm(v, dim=1)
    # dot product (for batch of vecs)
    c = (init_ * dir_vecs).sum(dim=1)
    # calculate rotation matrices
    rotmats = torch.eye(3).to(device) + rmats + (rmats @ rmats) * ((1 - c) / (s ** 2)).view([-1, 1, 1])

    return rotmats
