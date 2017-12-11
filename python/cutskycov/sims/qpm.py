from nbodykit.binned_statistic import BinnedStatistic
from pyRSD.rsdfit.results import LBFGSResults
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsdfit.parameters import ParameterSet

from collections import defaultdict
from glob import glob
import os
import numpy

def _load(kind, box=None):
    """
    Internal function to load the QPM results.
    """
    d = os.environ['THESIS_DIR']
    d = os.path.join(d, 'boss_dr12_mocks', 'Results', 'QPM', 'nbodykit', 'redshift', kind)

    # the pattern
    box = "*" if box is None else "%04d" %box
    if kind == 'power':
        files = glob(os.path.join(d, f"pkmu_qpm_unscaled_{box}_0.6452_dk005_Nmu100.dat"))
        dims = ['k', 'mu']
        names = ['k', 'mu', 'power', 'modes']
    else:
        files = glob(os.path.join(d, f"poles_qpm_unscaled_{box}_0.6452_dk005_Nmu100.dat"))
        dims = ['k']
        names = ['k', 'power_0', 'power_2', 'power_4', 'modes']

    toret = []
    for f in files:
        d = BinnedStatistic.from_plaintext(dims, f)
        if kind == 'poles':
            for i, name in enumerate(names):
                d.rename_variable('col_%d' %i, name)
        toret.append(d)

    return toret

def load_spectra(kind, box=None, subtract_shot_noise=True, average=True):
    """
    Load the QPM measurement results.

    Parameters
    ----------
    box : int, optional
        return the measurement for a specific box
    subtract_shot_noise : bool, optional
        whether or not to subtract out the shot noise
    average : bool, optional
        whether to average multiple results
    """
    assert kind in ['power', 'poles']

    power = 'power'
    if kind == 'poles':
        power += '_0'

    # load the results
    results = _load(kind, box=box)
    assert len(results) > 0

    # return a single result
    if len(results) == 1:
        r = results[0]
        if subtract_shot_noise:
            r[power].real -= r.attrs['volume'] / r.attrs['N1']
        return r

    # return all of the results, maybe averaged
    else:
        if subtract_shot_noise:
            for r in results:
                r[power].real -= r.attrs['volume'] / r.attrs['N1']

        if not average:
            data = [r.data for r in results]
            data = numpy.asarray(data, dtype=data[0].dtype)
            return data
        else:
            data = results[0].copy()
            for k in results[0].variables:
                data[k] = numpy.asarray([r[k] for r in results]).mean(axis=0)
            return data
