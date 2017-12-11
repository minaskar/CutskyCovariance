from nbodykit.lab import ConvolvedFFTPower
from pyRSD.rsdfit.results import LBFGSResults
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsdfit.parameters import ParameterSet

from collections import defaultdict
from glob import glob
import os
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def _load(box=None):
    """
    Internal function to load the N-cutsky results.
    """
    d = os.environ['THESIS_DIR']
    d = os.path.join(d, 'boss_dr12_mocks', 'Results', 'CutskyChallengeMocks', 'nbodykit', 'poles')

    if box is None:
        box = "*"
    else:
        box = "%d" %box

    files = glob(os.path.join(d, f"poles_my_cutskyN{box}_redshift_unscaled_fkp_1e4_dk005_lmax6_interlaced.json"))
    return [ConvolvedFFTPower.load(f) for f in files]

def get_normalization():
    """
    Compute the mean power spectrum normalization ``A`` from the N-cutsky mocks.
    """
    results = _load()
    return numpy.mean([r.attrs['randoms.A'] for r in results])

def load_spectra(box=None, subtract_shot_noise=True, average=True):
    """
    Load the N-cutsky measurement results.

    Parameters
    ----------
    box : int, optional
        return the measurement for a specific box
    subtract_shot_noise : bool, optional
        whether or not to subtract out the shot noise
    average : bool, optional
        whether to average multiple results
    """
    # load the results
    results = _load(box=box)

    # return a single result
    if box is not None:
        assert len(results) == 1
        r = results[0]
        if subtract_shot_noise:
            r.poles['power_0'].real -= r.attrs['shotnoise']
        return r

    # return all of the results, maybe averaged
    else:
        if subtract_shot_noise:
            for r in results:
                r.poles['power_0'].real -= r.attrs['shotnoise']

        if not average:
            data = [r.poles.data for r in results]
            data = numpy.asarray(data, dtype=data[0].dtype)
            return data
        else:
            data = results[0].poles.copy()
            for k in results[0].poles.variables:
                data[k] = numpy.asarray([r.poles[k] for r in results]).mean(axis=0)
            return data

def load_fits():
    """
    Load a set of fit results.

    Returns a structued numpy array holding best-fit values for all free
    parameters all mocks.
    """
    # find matches
    pattern = os.path.join(os.environ['RSDFIT_FITS'], 'cutsky', 'ChallengeMockN', 'fkp_1e4', 'box*')
    pattern = os.path.join(pattern, 'poles', 'nlopt_gausscov_base_high_nbar_unscaled_kmax04')
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
    theory for all 84 N-cutsky cubic boxes.
    """
    # the model file
    model = os.path.join(os.environ['RSDFIT'], 'data', 'models', 'model_nseries.npy')

    # the directory of box 1
    d = os.path.join(os.environ['RSDFIT_FITS'], 'cutsky', 'ChallengeMockN', 'fkp_1e4', 'box1')
    d = os.path.join(d, 'poles', 'nlopt_gausscov_base_high_nbar_unscaled_kmax04')

    # the bestfit values
    fits = load_fits()
    print("taking the mean of %d fits..." %len(fits))

    driver = FittingDriver.from_directory(d, model_file=model)
    theta = numpy.array([fits[name].mean() for name in driver.theory.free_names])
    driver.theory.set_free_parameters(theta)

    return driver.theory.model


from . import Config as BaseConfig

class Config(BaseConfig):

    def __init__(self):

        d = os.environ['THESIS_DIR']
        d = os.path.join(d, 'boss_dr12_mocks', 'Results', 'CutskyChallengeMocks', 'extra', 'nbar')
        self._nbar_file = os.path.join(d, 'nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat')

    @property
    def nbar(self):

        try:
            return self._nbar
        except:
            filename = os.path.join(os.environ['THESIS_DIR'], 'boss_dr12_data', 'cmass', 'Meta', 'cmass-dr12v5-ngc.zsel')
            zsel = numpy.loadtxt(filename)
            nbar_spline = spline(zsel[:,0], zsel[:,1])
            self._nbar = lambda z: 4.17e-4 * nbar_spline(z)
            return self._nbar

    @property
    def cosmo(self):
        from pyRSD import pygcl
        return pygcl.Cosmology("Boss_Nseries.ini")
