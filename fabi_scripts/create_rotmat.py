from scipy.spatial.transform import Rotation as R
import numpy as np
import sys

if __name__ == '__main__':
    assert len(sys.argv) == 3 or len(sys.argv) == 5, f'usage: test_utils.py rotmat_file angle <cx> <cy>'

    if len(sys.argv) == 3:
        cx, cy = 0, 0
    else:
        cx, cy = int(sys.argv[3]), int(sys.argv[4])

    rotmat_file = sys.argv[1]
    angle = int(sys.argv[2])

    angle_transvec = np.pi/4 + np.radians(angle)
    transvec = np.sqrt(2) * np.array([cx * np.cos(angle_transvec), cy * np.sin(angle_transvec), 0.]) - \
               np.array([cx, cy, 0.])

    rotmat = R.from_rotvec(angle * np.array([0, 0, 1]), degrees=True).as_matrix()
    rotmat = np.c_[rotmat, transvec]  # append empty transformation vec
    # rotmat = np.c_[rotmat, np.array([-192, 0, 0])]# np.zeros([3, 1])]  # append empty transformation vec
    # rotmat = np.c_[np.eye(3), np.array([0, 0, 0.])]
    # rotmat = np.c_[np.eye(3), transvec]

    np.savetxt(rotmat_file, rotmat, fmt='%1.4f')