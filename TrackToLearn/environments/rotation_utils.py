import numpy as np

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


def get_directional_sph(l_max=6):
    # ker = kernels.von_misses_fisher()
    ker = kernels.half_cosine()
    sig = SHCilmToVector(SHExpandDH(ker), lmax=l_max)

    return sig


def get_rotated_directional_sph(euler_ang, l_max=6):
    sph = get_directional_sph(l_max)
    rot_sph = rotate_sph(sph, euler_ang, l_max)

    return rot_sph


def rot_from_dir(dir_vec):
    """
    converts a new direction vector (cartesian) to the euler angle between the given
    direction and [1, 0, 0]
    """
    alpha, beta, _ = cart2sph(*list(dir_vec))
    euler = Rotation.from_euler('xyz', [alpha, -beta, 0]).as_euler('ZYZ')

    return list(euler)


def dir_to_sph(dir_vec, l_max=6):
    euler_ang = rot_from_dir(dir_vec)
    res = get_rotated_directional_sph(euler_ang, l_max)

    return res


def dirs_to_sph_channels(dir_vecs):
    """
    expects an array of 3D direction coordinates (in last dimension) and returns
    corresponding spherical coefficients
    """
    assert dir_vecs.shape[-1] == 3

    # for i_ in range(dir_vecs.shape[0]):
    #     dir_to_sph(dir_vecs[i_, 0])

    res = Parallel(n_jobs=24)(delayed(dir_to_sph)(e_) for e_ in dir_vecs[:, 0])

    # res = np.apply_along_axis(dir_to_sph, 1, dir_vecs[:, 0])


