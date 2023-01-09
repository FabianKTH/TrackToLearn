import numpy as np

from dipy.direction import DeterministicMaximumDirectionGetter, ClosestPeakDirectionGetter, ProbabilisticDirectionGetter
from dipy.data import default_sphere
from dipy.reconst.recspeed import local_maxima
from dipy.direction import peak_directions, Sphere


class BaseDirGetter:
    pass # TODO with IDE


class PropDirGetter:

    def __init__(self, sphere=default_sphere):
        self.sphere=sphere

    def _get_getter(self, sh_signal):
        getter = ProbabilisticDirectionGetter.from_shcoeff(sh_signal, max_angle=25., sphere=self.sphere)

        return getter


    def eval(self, sh_signal, prev_dirs=None):

        assert sh_signal.ndim == 2  # expect list of sh signals

        no_components = sh_signal.shape[0]
        directions = np.empty([no_components, 3])

        for _ in range(4 - sh_signal.ndim):
            sh_signal = sh_signal[None, ...]

        getter = self._get_getter(sh_signal)

        for c_idx in range(no_components):
            pos = np.array([0., 0., c_idx])
            import ipdb; ipdb.set_trace()
            directions[c_idx] = getter.initial_direction(pos)  # assume here: info from direction already encoded in sh sig
                                                               # else: getter.get_direction(pos, dir)
        return directions


class OdfDirGetter:

    def __init__(self, sphere):
        self.sphere = Sphere(xyz=sphere)
        self.sphere_verts = sphere

    def eval(self, odf_signal):

        no_components = odf_signal.shape[0]
        directions = np.zeros([no_components, 3])

        for c_idx in range(no_components):

            # dir_, val, idx = peak_directions(odf_signal[c_idx], self.sphere, relative_peak_threshold=.5)
            val, idx = local_maxima(odf_signal[c_idx], self.sphere.edges)
            val[val < 0] = 0.
            dir_ = self.sphere_verts[idx]

            # sample with peak values as weights
            # print(f'no peaks: {len(dir_)}')

            if len(dir_) == 0:
                # print('[!] no peak found')
                # no peak found -> sample uniform
                directions[c_idx] = np.random.uniform(0, 1, 3)
            else:
                # print('[:] using peak')
                peak_idx = np.argmax(val)
                directions[c_idx] = dir_[peak_idx]

            """
            if len(dir_) == 0:
                # no peak found -> sample uniform
                directions[c_idx] = np.random.uniform(0, 1, 3)
            else:
                peak_idx = np.random.choice(dir_.shape[0], p=val / val.sum())
                # peak_idx = np.random.choice(dir_.shape[0])
                directions[c_idx] = dir_[peak_idx]
            """

        return directions


if __name__ == '__main__':
    _pd = PropDirGetter()
    _sig = np.zeros([5, 9])
    _sig[0, 1] = _sig[1, 2] = _sig[2, 3] = _sig[4, 1] = _sig[4, 3] = _sig[3, 1] = 1.
    _dirs = _pd.eval(_sig)
        
    import ipdb; ipdb.set_trace()
