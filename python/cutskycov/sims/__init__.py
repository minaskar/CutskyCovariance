import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class Config(object):
    """
    A simulation config object.
    """
    def _get_effective_area(self):
        """
        Return the effective area in squared degrees from the relevant nbar file.
        """
        lines = open(self._nbar_file, 'r').readlines()
        return float(lines[1].split()[0])

    def _get_nbar(self):
        """
        Return a spline fit to n(z).
        """
        nbar = numpy.loadtxt(self._nbar_file, skiprows=3)
        return spline(nbar[:,0], nbar[:,3])

    @property
    def fsky(self):
        try:
            return self._fsky
        except AttributeError:
            eff_area = self._get_effective_area()
            self._fsky = eff_area * (numpy.pi/180.)**2 / (4*numpy.pi)
            return self._fsky

    @property
    def nbar(self):
        try:
            return self._nbar
        except AttributeError:
            self._nbar = self._get_nbar()
            return self._nbar
