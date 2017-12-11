import cutskycov as cov
from cutskycov.sims import eboss
from pyRSD.rsd import QuasarSpectrum
from eboss_qso.measurements import bias_model

def mock_covariance():
    """
    Save the covariance computed from the 1000 EZ mocks
    """
    for sample in ['N', 'S']:
        mocks = eboss.load_spectra(sample, average=False)
        C = cov.compute_mock_covariance(mocks, [0,2], kmax=0.4)
        C.to_plaintext("../covariance/eboss_%s_mock_cov_02.dat" %sample)


def analytic_covariance():
    """
    Use an evolving model in redshift to compute analytic eBOSS covariance.
    """
    def update(model, z):
        model.z = z
        model.f = model.cosmo.f_z(z)
        model.sigma8_z = model.cosmo.Sigma8_z(z)
        model.b1 = bias_model(z)

    for sample in ['N', 'S']:

        c = eboss.Config(sample)
        m = QuasarSpectrum(params=c.cosmo, z=1.52)

        kws = {}
        kws['zmin'] = 0.8
        kws['zmax'] = 2.2
        kws['kmin'] = 0.
        kws['kmax'] = 0.4
        kws['dk'] = 0.005
        kws['ells'] = [0,2]
        kws['P0_FKP'] = 3e4
        kws['A'] = eboss.get_normalization(sample)
        kws['evolve'] = update
        C = cov.covariance_from_model(m, c.nbar, **kws)
        C.to_plaintext("../covariance/eboss_%s_analytic_cov_evolve_02.dat" %sample)

if __name__ == '__main__':

    mock_covariance()
    analytic_covariance()
