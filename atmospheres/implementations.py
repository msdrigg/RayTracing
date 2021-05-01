from Vector import spherical_to_cartesian, cartesian_to_spherical, angle_between, unit_vector
from Constants import EARTH_RADIUS
from . import BaseAtmosphere 
from numpy import array, exp, linspace, zeros, cross, sign, repeat, square, maximum, sqrt
from scipy.spatial.transform import Rotation


class ChapmanLayers(BaseAtmosphere):
    """
    This class is an implementation of the chapman layers 
    atmosphere definition. See BaseAtmosphere for documentation on 
    how property definitions used in this class
    """
    def __init__(self, f0, hm, ym, gradient, start_point):
        self._parameters = array([f0, hm, ym])
        # Array-like parameter whose first element is the magnitude of the gradient in a unit of
        #   MHz per degree and whose second element is the parameter for the direction
        #   -1 for north, +1 for south, +2 for east, -2 for west
        self.gradient = gradient

        # Spherical start point
        self.start_point = start_point

    def plasma_frequency(self, coordinate, using_spherical=False):
        if not using_spherical:
            coordinate = cartesian_to_spherical(coordinate).reshape(-1, 3)
        else:
            coordinate = coordinate.reshape(-1, 3)

        h = coordinate[:, 0] - EARTH_RADIUS
        z1 = (h - self._parameters[1])/self._parameters[2]

        multiplier = self._parameters[0]
        if self.gradient is not None:
            multiplier = multiplier + self.gradient[0] * sign(self.gradient[1]) * \
                      (coordinate[:, abs(self.gradient[1])] -
                       cartesian_to_spherical(self.start_point)[abs(self.gradient[1])])
        output = multiplier*exp((1 - z1 - exp(-z1)))

        if len(output) == 1:
            return output[0]
        else:
            return output


class QuasiParabolic(BaseAtmosphere):
    """
    This class is an implementation of the quasi parabolic
    atmosphere definition. See BaseAtmosphere for documentation on 
    how property definitions used in this class
    """
    def __init__(self, f0, hm, ym):
        # Parameters are of the form (Max frequency, height of maximum, and semi-layer thickness.
        self._parameters = array([f0, hm, ym])

    def plasma_frequency(self, coordinate, using_spherical=False):
        if not using_spherical:
            coordinate = cartesian_to_spherical(coordinate).reshape(-1, 3)
        else:
            coordinate = coordinate.reshape(-1, 3)

        multiplier = self._parameters[0]
        output = multiplier*sqrt(maximum(1 - square((coordinate[:, 0] - self._parameters[1]) *
                                 (self._parameters[1] - self._parameters[2]) /
                                 (coordinate[:, 0]*self._parameters[2])), 0))

        if len(output) == 1:
            return output[0]
        else:
            return output
