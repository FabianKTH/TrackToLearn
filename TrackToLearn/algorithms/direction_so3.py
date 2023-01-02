import numpy as np

from dipy.direction import DeterministicMaximumDirectionGetter, ClosestPeakDirectionGetter, ProbabilisticDirectionGetter
from dipy.data import default_sphere


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
            directions[c_idx] = getter.initial_direction(pos)  # assume here: info from direction already encoded in sh sig
                                                               # else: getter.get_direction(pos, dir)
        return directions
            

if __name__ == '__main__':
    _pd = PropDirGetter()
    _sig = np.zeros([5, 9])
    _sig[0, 1] = _sig[1, 2] = _sig[2, 3] = _sig[4, 1] = _sig[4, 3] = _sig[3, 1] = 1.
    _dirs = _pd.eval(_sig)
        
    import ipdb; ipdb.set_trace()
