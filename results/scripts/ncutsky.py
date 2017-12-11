import cutskycov as cov
from cutskycov.sims import ncutsky, ncubic

def mock_covariance():
    """
    Save the covariance computed from the 84 mocks
    """
    mocks = ncutsky.load_spectra(average=False)
    C = cov.compute_mock_covariance(mocks, [0,2], kmax=0.4)
    C.to_plaintext("../covariance/ncutsky_mock_cov_02.dat")


def analytic_covariance(kind='theory'):
    """
    Use an evolving model in redshift to compute analytic covariance.
    """
    c = ncutsky.Config()
    kws = {}
    kws['nbar']=  c.nbar
    kws['zmin'] = 0.43
    kws['zmax'] = 0.7
    kws['kmin'] = 0.
    kws['kmax'] = 0.4
    kws['ells'] = [0,2]
    kws['P0_FKP'] = 1e4
    kws['A'] = ncutsky.get_normalization()

    if kind == 'theory':
        m = ncubic.load_bestfit_model()
        m.kmin = 1e-5
        m.kmax = 0.6

        kws['dk'] = 0.005
        C = cov.covariance_from_model(m, **kws)
        C.to_plaintext("../covariance/ncutsky_analytic_cov_from_model_02.dat")

    else:

        Pkmu = ncubic.load_spectra()
        C = cov.covariance_from_data(Pkmu, c.cosmo, **kws)
        C.to_plaintext("../covariance/ncutsky_analytic_cov_from_data_02.dat")


if __name__ == '__main__':

    mock_covariance()
    analytic_covariance(kind='theory')
    analytic_covariance(kind='data')
