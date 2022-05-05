import numpy as np

from TrackToLearn.environments.utils import (
    interpolate_volume_at_coordinates,
    is_inside_mask,
    is_too_curvy)
from TrackToLearn.datasets.utils import (
    MRIDataVolume)
from TrackToLearn.utils.utils import (
    normalize_vectors)


def reward_streamlines_step(
    streamlines: np.ndarray,
    peaks: MRIDataVolume,
    exclude: MRIDataVolume,
    target: MRIDataVolume,
    max_nb_steps: float,
    max_angle: float,
    min_nb_steps: float,
    alignment_weighting: float = 0.5,
    straightness_weighting: float = 0.1,
    length_weighting: float = 0.4,
    target_bonus_factor: float = 1.0,
    exclude_penalty_factor: float = -1.0,
    angle_penalty_factor: float = -1.0,
    affine_vox2mask: np.ndarray = None,
) -> list:

    """
    Compute rewards for the last step of the streamlines
    Each reward component is weighted according to a
    coefficient

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    peaks: `MRIDataVolume`
        Volume containing the fODFs peaks
    target_mask: MRIDataVolume
        Mask representing the tracking endpoints
    exclude_mask: MRIDataVolume
        Mask representing the tracking no-go zones
    max_len: `float`
        Maximum lengths for the streamlines (in terms of points)
    max_angle: `float`
        Maximum degrees between two streamline segments
    alignment_weighting: `float`
        Coefficient for how much reward to give to the alignment
        with peaks
    straightness_weighting: `float`
        Coefficient for how much reward to give to the alignment
        with the last streamline segment
    length_weighting: `float`
        Coefficient for how much to reward the streamline for being
        long
    target_bonus_factor: `float`
        Bonus for streamlines reaching the target mask
    exclude_penalty_factor: `float`
        Penalty for streamlines reaching the exclusion mask
    angle_penalty_factor: `float`
        Penalty for looping or too-curvy streamlines
    affine_vox2mask: np.ndarray
        Affine for moving stuff to voxel space

    Returns
    -------
    rewards: `float`
        Reward components weighted by their coefficients as well
        as the penalites
    """

    N = len(streamlines)

    length = reward_length(streamlines, max_nb_steps) \
        if length_weighting > 0. else np.zeros((N), dtype=np.uint8)
    alignment = reward_alignment_with_peaks(
        streamlines, peaks.data, affine_vox2mask) \
        if alignment_weighting > 0 else np.zeros((N), dtype=np.uint8)
    straightness = reward_straightness(streamlines) \
        if straightness_weighting > 0 else np.zeros((N), dtype=np.uint8)

    weights = np.asarray([
        alignment_weighting, straightness_weighting, length_weighting])
    params = np.stack((alignment, straightness, length))
    rewards = np.dot(params.T, weights)

    # Penalize sharp turns
    if angle_penalty_factor > 0.:
        rewards += penalize_sharp_turns(
            streamlines, max_angle, angle_penalty_factor)

    # Penalize streamlines ending in exclusion mask
    if exclude_penalty_factor > 0.:
        rewards += penalize_exclude(
            streamlines,
            exclude.data,
            affine_vox2mask,
            exclude_penalty_factor)

    # Reward streamlines ending in target mask
    if target_bonus_factor > 0.:
        rewards += reward_target(
            streamlines,
            min_nb_steps,
            target.data,
            affine_vox2mask,
            target_bonus_factor)

    return rewards


def reward_target(
    streamlines: np.ndarray,
    min_length: int,
    target: np.ndarray,
    affine_vox2mask: np.ndarray,
    factor: float
):
    """ Reward streamlines if they end up in the GM

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    target: np.ndarray
        Grey matter mask
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    penalty_factor: `float`
        Penalty for streamlines ending in target mask
        Should be >= 0

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    # Get boolean array of streamlines ending in mask * penalty
    reward = \
        is_inside_mask(streamlines, target, affine_vox2mask, 0.5) * \
        factor
    # Mask reward by length
    lengths = np.asarray([streamlines.shape[1]] * len(streamlines))
    reward[lengths <= min_length] = 0.
    return reward


def penalize_exclude(streamlines, exclude, affine_vox2mask, penalty_factor):
    """ Penalize streamlines if they loop

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    exclude: np.ndarray
        CSF matter mask
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    penalty_factor: `float`
        Penalty for streamlines ending in exclusion mask
        Should be <= 0

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return (
        is_inside_mask(streamlines, exclude, affine_vox2mask, 0.5) *
        -penalty_factor)


def penalize_sharp_turns(streamlines, max_angle, penalty_factor):
    """ Penalize streamlines if they curve too much

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_angle: `float`
        Maximum angle between streamline steps
    penalty_factor: `float`
        Penalty for looping or too-curvy streamlines

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return is_too_curvy(streamlines, max_angle) * -penalty_factor


def reward_length(streamlines, max_length):
    """ Reward streamlines according to their length w.r.t the maximum length

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, S, _ = streamlines.shape

    rewards = np.asarray(
        [streamlines.shape[1]] * len(streamlines)) / max_length

    return rewards


def reward_alignment_with_peaks(
    streamlines, peaks, affine_vox2mask
):
    """ Reward streamlines according to the alignment to their corresponding
        fODFs peaks

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, L, _ = streamlines.shape

    if streamlines.shape[1] < 2:
        # Not enough segments to compute curvature
        return np.ones(len(streamlines), dtype=np.uint8)

    X, Y, Z, P = peaks.shape
    idx = streamlines[:, -2].astype(np.int)

    # Get peaks at streamline end
    v = interpolate_volume_at_coordinates(
        peaks, idx, mode='nearest', order=0)
    v = np.reshape(v, (N, 5, P // 5))

    with np.errstate(divide='ignore', invalid='ignore'):
        # # Normalize peaks
        v = normalize_vectors(v)

    # Zero NaNs
    v = np.nan_to_num(v)

    # Get last streamline segments

    dirs = np.diff(streamlines, axis=1)
    u = dirs[:, -1]
    # Normalize segments
    with np.errstate(divide='ignore', invalid='ignore'):
        u = normalize_vectors(u)

    # Zero NaNs
    u = np.nan_to_num(u)

    # Get do product between all peaks and last streamline segments
    dot = np.einsum('ijk,ik->ij', v, u)

    # Get alignment with the most aligned peak
    rewards = np.amax(np.abs(dot), axis=-1)
    # rewards = np.abs(dot)

    factors = np.ones((N))

    # Weight alignment with peaks with alignment to itself
    if streamlines.shape[1] >= 3:
        # Get previous to last segment
        w = dirs[:, -2]

        # # Normalize segments
        with np.errstate(divide='ignore', invalid='ignore'):
            w = normalize_vectors(w)

        # # Zero NaNs
        w = np.nan_to_num(w)

        # Calculate alignment between two segments
        np.einsum('ik,ik->i', u, w, out=factors)

    # Penalize angle with last step
    rewards *= factors

    return rewards


def reward_straightness(streamlines):
    """ Reward streamlines according to its sinuosity

    Distance between start and end of streamline / length

    A perfectly straight line has 1.
    A circle would have 0.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the angle between the last two segments
e   """

    N, S, _ = streamlines.shape

    start = streamlines[:, 0]
    end = streamlines[:, -1]

    step_size = 1.
    reward = np.linalg.norm(end - start, axis=1) / (S * step_size)

    return np.clip(reward + 0.5, 0, 1)
