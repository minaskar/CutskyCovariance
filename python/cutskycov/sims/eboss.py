from nbodykit.lab import ConvolvedFFTPower
from pyRSD.rsdfit.results import LBFGSResults
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsdfit.parameters import ParameterSet

from collections import defaultdict
from glob import glob
import os
import numpy

def _load(sample, box=None):
    """
    Internal function to load the eBOSS EZ mock results.
    """
    d = os.environ['EBOSS_DIR']
    d = os.path.join(d, 'measurements', 'spectra', 'mocks', 'ezmock', 'v1.8e-fph')

    if box is None:
        box = "*"
    else:
        box = "%04d" %box

    files = glob(os.path.join(d, f"poles_zevoEZmock_v1.8e-fph_QSO-{sample}_{box}-bba5aabfa6.json"))
    return [ConvolvedFFTPower.load(f) for f in files]

def get_normalization(sample):
    """
    Compute the mean power spectrum normalization ``A`` from the N-cutsky mocks.
    """
    results = _load(sample)
    return numpy.mean([r.attrs['randoms.norm'] for r in results])

def load_spectra(sample, box=None, subtract_shot_noise=True, average=True):
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
    assert sample in 'NS'

    # load the results
    results = _load(sample, box=box)

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

def load_fits(sample, params):
    """
    Load a set of ezmock fit results.

    Returns a structued numpy array holding best-fit values for all free
    parameters all mocks.
    """
    # find matches
    pattern = os.path.join(os.environ['EBOSS_FITS'], 'mocks', 'ezmock', 'v1.8e-fph', '0.0001-0.3', params)
    assert os.path.isdir(pattern)
    assert sample in 'NS'
    pattern = os.path.join(pattern, '0.8-2.2', f'QSO-{sample}-*-P0+P2-mock-cov-*')

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

def load_bestfit_model(sample, params):
    """
    Return a GalaxySpectrum model initialized with the mean of the best-fitting
    theory.
    """
    # the directory of box 1
    d = os.path.join(os.environ['EBOSS_FITS'], 'mocks', 'ezmock', 'v1.8e-fph', '0.0001-0.3', params)
    assert os.path.isdir(d)
    assert sample in 'NS'
    d = os.path.join(d, '0.8-2.2', f'QSO-{sample}-0001-P0+P2-mock-cov-*')
    d = glob(d)[0]

    # the bestfit values
    fits = load_fits(sample, params)
    print("taking the mean of %d fits..." %len(fits))

    driver = FittingDriver.from_directory(d)
    theta = numpy.array([fits[name].mean() for name in driver.theory.free_names])
    driver.theory.set_free_parameters(theta)

    return driver.theory.model

from . import Config as BaseConfig

class Config(BaseConfig):

    def __init__(self, sample):

        d = os.environ['EBOSS_DIR']
        d = os.path.join(d, 'data', 'v1.8', 'meta')
        self._nbar_file = os.path.join(d, f'nbar-eboss_v1.8-QSO-{sample}-eboss_v1.8.dat')

    @property
    def cosmo(self):
        from pyRSD.rsd.cosmology import Cosmology

        pars = {}
        pars['H0'] = 67.77
        pars['Neff'] = 3.046
        pars['Ob0'] = 0.048206
        pars['Om0'] = 0.307115
        pars['Tcmb0'] = 2.7255
        pars['m_nu'] = 0.0
        pars['n_s'] = 0.9611
        pars['sigma8'] = 0.8225
        cosmo = Cosmology(flat=True, **pars)
        return cosmo
