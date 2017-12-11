from pyRSD.rsdfit.data import PoleCovarianceMatrix
from astropy import units, constants
import numpy
from scipy.special import legendre
from scipy.integrate import simps

def covariance_from_data(data, cosmo, nbar, zmin, zmax, ells, kmin=0, kmax=0.4,
                            P0_FKP=0., Nz=50, fsky=None, A=None):
    """
    Compute the analytic cutsky covariance from a P(k,mu) as measured from
    cubic periodic simulations.

    Parameters
    ----------
    data : BinnedStatistic
        object behaving like structured numpy array holding ``k``, ``mu``
        ``power``, ``modes``
    cosmo :
        the cosmology object
    nbar : callable
        the function returning number density as a function of redshift
    zmin : float
        minimum redshift of the survey
    zmax : float
        maximum redshift of the survey
    ells : list of int
        the desired multipole numbers to compute
    kmin : float, optional
        trim covariance to this min wavenumber
    kmax : float, optional
        trim covariance to this max wavenumber
    P0_FKP : float, optional
        the FKP ``P0`` value
    Nz : int, optional
        the number of bins when integrating over redshift
    fsky : float, optional
        the sky fraction, either this or ``A`` must be supplied
    A : float, optional
        the normalization, given by :math:`\int dV n^2 w^2`; either this or
        ``fsky`` must be supplied

    Returns
    -------
    C : PoleCovarianceMatrix
        the covariance matrix object holding the multipole covariance
    """
    # both fsky and A cannot be None
    if fsky is None and A is None:
        raise ValueError("either 'fsky' or 'A' must be supplied")

    # nbar is a callable
    assert callable(nbar)

    k        = data['k']
    mus      = data['mu']
    Pkmu     = data['power'].real
    modes_2d = data['modes']

    # 1d k
    k = numpy.nansum(k*modes_2d, axis=-1) / numpy.nansum(modes_2d, axis=-1)

    N1 = Pkmu.shape[0]
    N2 = len(ells)

    # setup redshift binning
    zbins = numpy.linspace(zmin, zmax, Nz+1)
    zcen = 0.5*(zbins[:-1] + zbins[1:])

    # the comoving volume element
    Da = cosmo.Da_z(zcen) * cosmo.h() # in Mpc/h
    dV = 4*numpy.pi*(1+zcen)**2 * Da**2  / cosmo.H_z(zcen) * cosmo.h()
    dV *= constants.c.to(units.km/units.second).value
    dV *= numpy.diff(zbins)

    # compute nbar
    nbar_ = nbar(zcen)

    # weights
    w = 1. / (1 + nbar_*P0_FKP) # FKP weights

    # properly calibrate fsky
    W = ((nbar_*w)**2 * dV).sum()
    if A is not None:
        fsky = A / W
        print('using fsky = ', fsky)
    dV *= fsky
    W *= fsky

    # k-shell volume
    dk = numpy.diff(k).mean()
    Vk  = 4*numpy.pi*k**2*dk

    # initialize the return array
    cov = numpy.zeros((N2, N1)*2)

    # leg weights
    leg = numpy.array([(2*ell+1)*legendre(ell)(mus) for ell in ells])

    # P(k,mu)^2 * L_ell * L_ellprime
    weights = leg[:,None]*leg[None,:]
    power = (Pkmu[...,None] + 1./nbar_)**2
    tobin = weights[...,None] * power[None,...]

    # do the sum over redshift first
    x = ( (w*nbar_)**4 * dV * tobin).sum(axis=-1) / W**2

    # the sum of the weights
    N = numpy.nansum(modes_2d, axis=-1)

    # normalization of covariance
    norm = 2 * (2*numpy.pi)**3 / Vk

    # fill the covariance for each ell, ell_prime
    for i in range(N2):
        for j in range(i, N2):

            t = numpy.nansum(x[i,j,:]*modes_2d, axis=-1) / N
            cov[i,:,j,:] = norm * numpy.diag(t)
            if i != j:
                cov[j,:,i,:] = cov[i,:,j,:]

    # reshape squared power and modes
    cov = cov.reshape((N1*N2,)*2)
    cov = numpy.nan_to_num(cov)

    # the coordinate arrays
    k_coord = numpy.concatenate([k for i in range(len(ells))])
    ell_coord = numpy.concatenate([numpy.ones(len(k), dtype=int)*ell for ell in ells])

    C = PoleCovarianceMatrix(cov, k_coord, ell_coord, verify=False)

    valid_range = slice(kmin, kmax)
    return C.sel(k1=valid_range, k2=valid_range)


def covariance_from_model(model, nbar, zmin, zmax, kmin, kmax, dk, ells,
                            P0_FKP=0., Nmu=100, Nz=50, Nk=50, fsky=None, A=None, evolve=None):
    """
    Compute the analytic cutsky covariance from an analytic P(k,mu) model.

    Parameters
    ----------
    model : QuasarSpectrum
        the model predicting P(k,mu)
    nbar : callable
        the function returning number density as a function of redshift
    zmin : float
        minimum redshift of the survey
    zmax : float
        maximum redshift of the survey
    ells : list of int
        the desired multipole numbers to compute
    kmin : float, optional
        trim covariance to this min wavenumber
    kmax : float, optional
        trim covariance to this max wavenumber
    dk : float
        the width of the wavenumber bins
    P0_FKP : float, optional
        the FKP ``P0`` value
    Nz : int, optional
        the number of bins when integrating over redshift
    fsky : float, optional
        the sky fraction, either this or ``A`` must be supplied
    A : float, optional
        the normalization, given by :math:`\int dV n^2 w^2`; either this or
        ``fsky`` must be supplied
    evolve : callable
        the function that updates the model as a function of redshift

    Returns
    -------
    C : PoleCovarianceMatrix
        the covariance matrix object holding the multipole covariance
    """
    # both fsky and A cannot be None
    if fsky is None and A is None:
        raise ValueError("either 'fsky' or 'A' must be supplied")

    # nbar is a callable
    assert callable(nbar)

    # the cosmology
    cosmo = model.cosmo

    # the finer k binning
    dk_ = dk/Nk
    kedges = numpy.arange(kmin, kmax+0.5*dk_, dk_)
    kcen = 0.5*(kedges[1:] + kedges[:-1])

    # the 2D (k,mu) grid
    mus_ = numpy.linspace(0, 1, Nmu+1)
    _, mus = numpy.meshgrid(kcen, mus_, indexing='ij')

    # the output k grid
    kedges = numpy.arange(kmin, kmax+0.5*dk, dk)
    kout = 0.5*(kedges[1:] + kedges[:-1])

    N1 = len(kout)
    N2 = len(ells)

    # setup redshift binning
    zbins = numpy.linspace(zmin, zmax, Nz+1)
    zcen = 0.5*(zbins[:-1] + zbins[1:])

    # the comoving volume element
    Da = cosmo.Da_z(zcen) # in Mpc/h
    dV = 4*numpy.pi*(1+zcen)**2 * Da**2  / cosmo.H_z(zcen)
    dV *= cosmo.h()**3
    dV *= constants.c.to(units.km/units.second).value
    dV *= numpy.diff(zbins)

    # compute nbar
    nbar_ = nbar(zcen)

    # weights
    w = 1. / (1 + nbar_*P0_FKP) # FKP weights

    # properly calibrate fsky
    W = ((nbar_*w)**2 * dV).sum()
    if A is not None:
        fsky = A / W
        print('using fsky = ', fsky)
    dV *= fsky
    W *= fsky

    # k-shell volume
    Vk  = 4*numpy.pi/3. * numpy.diff(kedges**3)

    # initialize the return array
    cov = numpy.zeros((N2, N1)*2)

    # leg weights
    leg = numpy.array([(2.*ell+1.)*legendre(ell)(mus) for ell in ells])

    # compute Pkmu, optionally evolving in redshift
    if evolve is not None:
        Pkmu = []
        for zi in zcen:
            evolve(model, zi)
            Pkmu.append(model.power(kcen,mus_).values)
        Pkmu = numpy.asarray(Pkmu)
        Pkmu = numpy.moveaxis(Pkmu,0,-1)
    else:
        Pkmu = model.power(kcen,mus_).values
        Pkmu = Pkmu[...,None]

    # P(k,mu)^2 * L_ell * L_ellprime
    weights = leg[:,None]*leg[None,:]
    power = (Pkmu + 1./nbar_)**2
    tobin = weights[...,None] * power[None,...]

    # do the sum over redshift first
    x = ( (w*nbar_)**4 * dV * tobin).sum(axis=-1) / W**2

    # the normalization
    norm = 4*(2*numpy.pi)**4 / (Vk**2)

    # split k into chunks to average
    N_chunks = int(len(kcen)/Nk)
    kcen_split = numpy.split(kcen, N_chunks)

    # fill the covariance for each ell, ell_prime
    for i in range(N2):
        for j in range(i, N2):

            # do the mu integral (mean is okay b/c of [0,1] domain)
            t = numpy.nanmean(x[i,j,:], axis=-1)

            # do the k averaging
            x_split = numpy.split(t, N_chunks)
            t = numpy.array([simps(xx * kk**2, x=kk) for xx,kk in zip(x_split, kcen_split)])

            cov[i,:,j,:] = norm * numpy.diag(t)
            if i != j:
                cov[j,:,i,:] = cov[i,:,j,:]

    # reshape squared power and modes
    cov = cov.reshape((N1*N2,)*2)
    cov = numpy.nan_to_num(cov)

    # the coordinate arrays
    k_coord = numpy.concatenate([kout for i in range(len(ells))])
    ell_coord = numpy.concatenate([numpy.ones(len(kout), dtype=int)*ell for ell in ells])

    return PoleCovarianceMatrix(cov, k_coord, ell_coord, verify=False)
