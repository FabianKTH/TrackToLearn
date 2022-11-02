import math as ma

import numpy as np


class Options:
    def __init__(self):
        self.in_dim = 64
        self.sigma = 5
        self.filter_threshold = .1
        self.kappa = 2.7
        self.L = 20
        self.angular_step = 1


def half_cosine():
    """ defines the half cosine kernel"""
    opt = Options()

    phi = np.arange(0, 90, opt.angular_step) * ma.pi / 180

    res = np.tile(np.cos(phi), (int(360 / opt.angular_step) + 1, 1)).T

    # print(res.shape)
    # print(np.zeros((res.shape[0]-1, res.shape[1])).shape)

    # res = np.vstack((res, np.zeros((res.shape[0] +1, res.shape[1]))))
    return res


def von_misses_fisher():
    """ defines the vonMisesFisher kernel"""
    opt = Options()

    phi = np.arange(0, 180, opt.angular_step) * ma.pi / 180
    res = np.tile(np.exp(opt.kappa * np.cos(phi)), (int(360 / opt.angular_step + 1), 1)).T
    res = res / np.sum(res[:, 0])

    return res
