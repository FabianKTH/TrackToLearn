import numpy as np

cimport cython
cimport numpy as cnp

import numpy as np


cdef class BinaryStoppingCriterion():
    """
    cdef:
        unsigned char[:, :, :] mask
    """

    def __cinit__(self, mask):
        self.mask = (mask > 0).astype('uint8')

    cdef int check_point_c(self, cnp.float_t[:] point):
        cdef:
            unsigned char result
            int err
            int voxel[3]

        voxel[0] = int(point[0])
        voxel[1] = int(point[1])
        voxel[2] = int(point[2])

        if (voxel[0] < 0 or voxel[0] >= self.mask.shape[0]
                or voxel[1] < 0 or voxel[1] >= self.mask.shape[1]
                or voxel[2] < 0 or voxel[2] >= self.mask.shape[2]):
            return 1

        result = self.mask[voxel[0], voxel[1], voxel[2]]
        if result > 0:
            return 1
        else:
            return 0


def is_inside_mask(
    cnp.float_t[:, :, :] streamlines,
    BinaryStoppingCriterion sc,
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
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
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

    N = streamlines.shape[1]
    result = np.zeros((N), dtype=np.uint8)

    for i in range(N):
        result[i] = sc.check_point_c(streamlines[i, -1, :])
    return result


def is_outside_mask(
    cnp.float_t[:, :, :] streamlines,
    BinaryStoppingCriterion sc,
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
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
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

    N = streamlines.shape[1]
    result = np.zeros((N), dtype=np.uint8)

    for i in range(N):
        result[i] = np.abs(sc.check_point_c(streamlines[i, -1, :]) - 1)
    return result

