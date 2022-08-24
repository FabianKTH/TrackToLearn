import numpy as np  # TODO: PYTORCH, hellow?!


def rotate_coeff(src, rotation, degree):
    R = SHRotate(rotation, degree)

    dest_ = np.empty_like(src)

    for i in range(0, degree):
        print(i)
        start_pos = i * i
        if i == 0:
            dest_[:, start_pos:start_pos + 2 * i + 1] = src[:, start_pos:start_pos + 2 * i + 1] * R[i]
        else:
            dest_[:, start_pos:start_pos + 2 * i + 1] = src[:, start_pos:start_pos + 2 * i + 1] @ R[i]

    return dest_


def SHRotate(r, degree):
    R = [None for _ in range(degree)]

    R[0] = np.array([1])

    R1 = np.empty([3, 3])
    R1[0, 0] = r[1, 1]
    R1[0, 1] = r[2, 1]
    R1[0, 2] = r[0, 1]

    R1[1, 0] = r[1, 2]
    R1[1, 1] = r[2, 2]
    R1[1, 2] = r[0, 2]

    R1[2, 0] = r[1, 0]
    R1[2, 1] = r[2, 0]
    R1[2, 2] = r[0, 0]

    R[1] = R1

    # Calculate each block of the rotation matrix for each subsequent band
    for band in range(2, degree):
        R[band] = np.empty([2 * band + 1, 2 * band + 1])
        for m in range(-band, band + 1):
            for n in range(-band, band + 1):
                # print(M(band, m, n, R))
                R[band][m + band, n + band] = M(band, m, n, R)

    return R


def M(l, m, n, R):
    d = int(m == 0)

    if abs(n) == l:
        denom = (2 * l) * (2 * l - 1)
    else:
        denom = (l * l - n * n)

    u = np.sqrt((l * l - m * m) / denom)

    v = np.sqrt((1 + d) * (l + abs(m) - 1) * (l + abs(m)) / denom) * (1 - 2 * d) * 0.5
    w = np.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d) * (-0.5)

    if u != 0:
        u = u * U(l, m, n, R)

    if v != 0:
        v = v * V(l, m, n, R)

    if w != 0:
        w = w * W(l, m, n, R)

    ret = u + v + w

    return ret


def U(l, m, n, R):
    ret = P(0, l, m, n, R)

    return ret


def P(i, l, a, b, R):
    ri1 = R[1][i + 1, 2]
    rim1 = R[1][i + 1, 0]
    ri0 = R[1][i + 1, 1]

    if b == -l:
        ret = ri1 * R[l-1][a + l - 1, 0] + rim1 * R[l-1][a + l -1, 2 * l - 2]
    elif b == l:
        ret = ri1 * R[l-1][a + l - 1, 2 * l - 2] - rim1 * R[l-1][a + l - 1, 0]
    else:
        ret = ri0 * R[l-1][a + l - 1, b + l - 1]

    return ret


def V(l, m, n, R):
    if m == 0:
        p0 = P(1, l, 1, n, R)
        p1 = P(-1, l, -1, n, R)
        ret = p0 + p1
    elif m > 0:
        d = (m == 1)
        p0 = P(1, l, m - 1, n, R)
        p1 = P(-1, l, -m + 1, n, R)
        ret = p0 * np.sqrt(1 + d) - p1 * (1 - d)
    else:
        d = (m == -1)
        p0 = P(1, l, m + 1, n, R)
        p1 = P(-1, l, -m - 1, n, R)
        ret = p0 * (1 - d) + p1 * np.sqrt(1 + d)

    return ret


def W(l, m, n, R):
    if m == 0:
        raise ValueError('never gets called')
    elif m > 0:
        p0 = P(1, l, m + 1, n, R)
        p1 = P(-1, l, -m - 1, n, R)
        ret = p0 + p1
    else:
        p0 = P(1, l, m - 1, n, R)
        p1 = P(-1, l, -m + 1, n, R)
        ret = p0 - p1

    return ret


if __name__ == '__main__':
    A_ = [[0.521441686053178, 0.0465425697238158, 0.436650064472378],
          [0.928110835083805, 0.905735575528561, 0.553645125546680],
          [0.252965436241309, 0.688749239620436, 0.0892280700122154]]
    A_ = np.array(A_)

    U_, S_, V_ = np.linalg.svd(A_)

    if np.linalg.det(U_) < 0:
        r = -U_
    else:
        r = U_

    degree_ = 3

    coeff = [[0.1605, 0.7565, 0.5547, 0.7359, 0.3366, 0.2326, 0.2862, 0.2621, 0.5542],
             [0.9015, 0.4077, 0.7853, 0.0661, 0.8411, 0.7557, 0.9707, 0.7291, 0.1270],
             [0.1484, 0.0680, 0.9970, 0.0693, 0.7562, 0.6954, 0.1182, 0.4868, 0.1112],
             [0.8404, 0.2772, 0.2613, 0.8368, 0.6310, 0.0693, 0.3615, 0.4567, 0.0551]]

    coeff = np.array(coeff)

    dest = rotate_coeff(coeff, r, degree_)

    print(dest)
