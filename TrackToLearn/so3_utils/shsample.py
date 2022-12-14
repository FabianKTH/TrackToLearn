import numpy as np
import pyshtools as pysh
import torch

from astropy.coordinates import uniform_spherical_random_surface
from sklearn.preprocessing import normalize

from dipy.reconst.shm import order_from_ncoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShBasisSeq:

    def __init__(self, l_max, n=1024, lm_order='pysh'):
        self.l_max = l_max
        self.n = n
        self.lm_order = lm_order
        self.arr = np.zeros([2, self.l_max + 1, self.l_max + 1, 2, self.n])
        self.idx_mapping = list()
        self.populate_array()

    @staticmethod
    def m_sign(m):
        return int(np.sign(m) < 0)

    def populate_array(self):

        for _l in range(self.l_max + 1):

            if self.lm_order == 'pysh':
                for _m in list(range(0, _l + 1)) + list(range(-1, -_l - 1, -1)):
                    self.arr[self.m_sign(_m), _l, np.abs(_m)] = self.pre_sampling(_l, _m)
                    self.idx_mapping.append([self.m_sign(_m), _l, np.abs(_m)])
            elif self.lm_order == 'lm':
                for _m in range(-_l, _l + 1):
                    self.arr[self.m_sign(_m), _l, np.abs(_m)] = self.pre_sampling(_l, _m)
                    self.idx_mapping.append([self.m_sign(_m), _l, np.abs(_m)])
            else:
                raise ValueError('lm_order should be [pysh, lm]')

    @staticmethod
    def sample_sphere(n=32):
        sph_unit = uniform_spherical_random_surface(size=n)
        phi, theta = np.array(sph_unit.lat), np.array(sph_unit.lon)
        phi += np.pi / 2

        return theta, phi

    @staticmethod
    def grid_sphere(n=32):
        d_th = d_ph = int(np.sqrt(n))

        theta = np.linspace(0., 2 * np.pi, d_th)
        # phi = np.linspace(-np.pi/2, np.pi/2, d_ph)
        phi = np.linspace(0., np.pi, d_ph)
        # Create a 2-D meshgrid of (theta, phi) angles.
        theta, phi = np.meshgrid(theta, phi)

        return theta.flatten(), phi.flatten()

    @staticmethod
    def get_sph_harm(l_, m_, theta, phi):
        # Y = pysh.expand.spharm_lm(l_, m_, theta, phi, kind='real', degrees=False)
        Y = pysh.expand.spharm_lm(l_, m_, phi, theta, kind='real', degrees=False)

        return Y

    def pre_sampling(self, l_, m):
        S = np.empty([2, self.n])
        s_idx = 0

        while s_idx < self.n:

            # theta, phi = self.sample_hemisphere(self.n)
            # theta, phi = self.grid_sphere(self.n)
            theta, phi = self.sample_sphere(self.n)
            samples = self.get_sph_harm(l_, m, theta, phi)

            # TODO: next line
            if not l_ == 0:
                samples[samples < 0.] = 0.
                samples = np.abs(samples)

                # normalize samples to [0, 1]
                samples = normalize(samples[None, ...], norm='max').squeeze()

            for s, th, ph in zip(samples, theta, phi):
                if s_idx >= self.n:
                    break

                if not l_ == 0:
                    r1 = np.random.uniform()

                    if r1 > s:
                        # if s < .7:
                        # np.mean(samples):
                        continue

                S[:, s_idx] = np.array((ph, th))
                s_idx += 1

        return S

    @staticmethod
    def spvec_to_prop(sp_vec):
        prop = normalize(np.abs(sp_vec)[None, ...], norm='max').squeeze()
        prop /= prop.sum()

        return prop

    def sample_multi(self, sph_vecs):
        """
        samples_l_m: precomputed filtered sample sequences for each Y_l_m
        sph_vecs: sequence of flattened spharm vectors
        """

        prop_w = [self.spvec_to_prop(sp_vec) for sp_vec in sph_vecs]
        lms = [np.random.choice(len(sph_vecs[0]), p=prop) for prop in prop_w]
        arr_idx = np.array([self.idx_mapping[lm] for lm in lms]).T
        ns = np.random.choice(self.n, size=len(sph_vecs))

        return self.arr[arr_idx[0], arr_idx[1], arr_idx[2], :, ns]


class ShBasisSeqTorch(ShBasisSeq):

    def __init__(self, l_max, n=1048):
        super().__init__(l_max, n, lm_order='lm')

        self.arr_torch = torch.tensor(self.arr, device=device)

    # def populate_array(self):
    #     assert self.lm_order == 'lm'
    #     # super().populate_array()

    @staticmethod
    def spvec_to_prop(sp_vec):
        # ditch negative coefficients
        sp_vec[sp_vec < 0.] = 0.

        # zero mask for division
        zero_mask = (sp_vec.sum(dim=1) != 0.)
        sp_vec[zero_mask] /= sp_vec[zero_mask].sum(dim=1, keepdim=True)

        # set first (isotropic) coeff for all zero vectors to 1.
        sp_vec[torch.logical_not(zero_mask), 0] = 1.

        return sp_vec

    def sample_multi(self, sph_vecs):

        no_vecs = len(sph_vecs)
        no_coeff = len(sph_vecs[0])

        assert order_from_ncoef(no_coeff, full_basis=True) == self.l_max

        prop_n = torch.full(size=(self.n,), fill_value=1./self.n, device=device)
        sph_vecs = self.spvec_to_prop(sph_vecs)
        lms = torch.multinomial(sph_vecs, num_samples=1, replacement=True).squeeze()
        arr_idx = torch.index_select(torch.tensor(self.idx_mapping, device=device),
                                     dim=0,
                                     index=lms).T
        ns = torch.multinomial(prop_n, no_vecs, replacement=True)

        return self.arr_torch[arr_idx[0], arr_idx[1], arr_idx[2], :, ns]


if __name__ == '__main__':
    bla = torch.full((3, 8), 0.).to(device)
    bla[1, 1] = 1.
    bla[0, 2] = 1.
    bla[2, 3] = .1
    bla[2, 7] = .1

    fabi = ShBasisSeqTorch(l_max=2)

    print(fabi.sample_multi(bla))


