from nbodykit.binned_statistic import BinnedStatistic
from pyRSD.rsdfit.results import LBFGSResults
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsdfit.parameters import ParameterSet

from collections import defaultdict
from glob import glob
import os
import numpy

def _load(box=None, los=None):
    """
    Internal function to load the N-cubic results.
    """
    d = os.environ['THESIS_DIR']
    d = os.path.join(d, 'boss_dr12_mocks', 'Results', 'ChallengeMocks', 'nbodykit', 'power')

    if los is not None:
        assert los in "xyz"

    # the pattern
    box = "*" if box is None else "%d" %box
    los = "*" if los is None else los

    files = glob(os.path.join(d, f"pkmu_challenge_boxN{box}_unscaled_dk005_Nmu100_{los}los.dat"))

    toret = []
    for f in files:
        toret.append(BinnedStatistic.from_plaintext(['k', 'mu'], f))

    return toret

def load_spectra(box=None, los=None, subtract_shot_noise=True, average=True):
    """
    Load the N-cubic measurement results.

    Parameters
    ----------
    box : int, optional
        return the measurement for a specific box
    los : int, optional
        return the measurement for a specific los
    subtract_shot_noise : bool, optional
        whether or not to subtract out the shot noise
    average : bool, optional
        whether to average multiple results
    """
    # load the results
    results = _load(box=box, los=None)
    assert len(results) > 0

    # return a single result
    if len(results) == 1:
        r = results[0]
        if subtract_shot_noise:
            r['power'].real -= r.attrs['volume'] / r.attrs['N1']
        return r

    # return all of the results, maybe averaged
    else:
        if subtract_shot_noise:
            for r in results:
                r['power'].real -= r.attrs['volume'] / r.attrs['N1']

        if not average:
            data = [r.data for r in results]
            data = numpy.asarray(data, dtype=data[0].dtype)
            return data
        else:
            data = results[0].copy()
            for k in results[0].variables:
                data[k] = numpy.asarray([r[k] for r in results]).mean(axis=0)
            return data

def load_fits():
    """
    Load a set of fit results.

    Returns a structued numpy array holding best-fit values for all free
    parameters all mocks.
    """
    # find matches
    pattern = os.path.join(os.environ['RSDFIT_FITS'], 'periodic', 'ChallengeBoxN', 'box*_*los')
    pattern = os.path.join(pattern, 'poles', 'nlopt_gausscov_base_kmax04')
    matches = glob(pattern)
    assert len(matches) > 0

    driver = None
    data = defaultdict(list)
    for f in matches:
        r = sorted(glob(os.path.join(f, '*.npz')), key=os.path.getmtime, reverse=True)
        assert len(r) > 0, "no npz results found in directory '%s'" %os.path.normpath(f)

        th = ParameterSet.from_file(os.path.join(f, 'params.dat'), tags='theory')
        r = LBFGSResults.from_npz(r[0])
        for param in r.free_names:
            data[param].append(r[param])
            th[param].value = r[param]

        if driver is None:
            driver = FittingDriver.from_directory(f, init_model=False)

        # add fsigma8
        if 'f' in r.free_names and 'sigma8_z' in r.free_names:
            data['fsigma8'].append(r['f'] * r['sigma8_z'])

        # the prior to add back
        lnprior = sum(par.lnprior for par in th.free)

        # add the reduced chi2
        red_chi2 = (2*(r.min_chi2 + lnprior)) / driver.dof
        data['red_chi2'].append(red_chi2)

    params = list(data.keys())
    dtype = list(zip(params, ['f8']*len(params)))
    out = numpy.empty(len(matches), dtype=dtype)
    for param in out.dtype.names:
        out[param] = numpy.array(data[param])

    return out

def load_bestfit_model():
    """
    Return a GalaxySpectrum model initialized with the mean of the best-fitting
    theory for all 21 N-series cubic boxes.
    """
    # the model file
    model = os.path.join(os.environ['RSDFIT'], 'data', 'models', 'model_nseries.npy')

    # the directory of box 1
    d = os.path.join(os.environ['RSDFIT_FITS'], 'periodic', 'ChallengeBoxN', 'box1_xlos')
    d = os.path.join(d, 'poles', 'nlopt_gausscov_base_kmax04')

    # the bestfit values
    fits = load_fits()
    print("taking the mean of %d fits..." %len(fits))

    driver = FittingDriver.from_directory(d, model_file=model)
    theta = numpy.array([fits[name].mean() for name in driver.theory.free_names])
    driver.theory.set_free_parameters(theta)

    return driver.theory.model
