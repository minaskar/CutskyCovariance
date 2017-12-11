import numpy
from pyRSD.rsdfit.data import PoleCovarianceMatrix

def compute_mock_covariance(mocks, ells, kmin=0., kmax=0.4):
    """
    Compute the covariance from a set of mocks.
    """
    Pell = numpy.concatenate([mocks['power_%d' %ell].real for ell in ells], axis=-1)
    cov = numpy.cov(Pell, rowvar=False)

    k = mocks['k'].mean(axis=0)
    k_coord = numpy.concatenate([k for i in range(len(ells))])
    ell_coord = numpy.concatenate([numpy.ones(len(k), dtype=int)*ell for ell in ells])

    # create and slice to correct range
    C = PoleCovarianceMatrix(cov, k_coord, ell_coord, verify=False)
    valid_range = slice(kmin, kmax)
    C = C.sel(k1=valid_range, k2=valid_range)

    # store the number of mocks
    C.attrs['Nmock'] = len(mocks)

    return C
