import numpy as np
from numpy.typing import ArrayLike

from utilities.Constants import EARTH_RADIUS
from utilities.Vector import cartesian_to_spherical
from atmospheres import BaseAtmosphere


class ChapmanLayers(BaseAtmosphere):
    """
    This class is an implementation of the chapman layers
    atmosphere definition. See BaseAtmosphere for documentation on
    how property definitions used in this class

    The plasma frequency is

        f_p = f0(position) * exp( (profile_exponent / 2) * (1 - z - exp(-z)) ),   z = (h - hm) / ym

    so the electron density (which is proportional to f_p^2) follows
    exp(profile_exponent * (1 - z - exp(-z))). profile_exponent = 1 is the
    textbook (beta) Chapman layer and 0.5 the alpha Chapman layer. The default
    of 2 reproduces the profile this code has always used
    (f_p = f0 * exp(1 - z - exp(-z))).
    """
    def __init__(self, f0, hm, ym, gradient, start_point, profile_exponent: float = 2.0):
        """
        :param f0: peak plasma frequency in Hz (at start_point if there is a gradient)
        :param hm: altitude of the peak in meters
        :param ym: scale height in meters
        :param gradient: None, or a tuple (magnitude, direction). Magnitude is the
            horizontal gradient of the peak plasma frequency in Hz per radian.
            Direction is -1 for increasing to the north, +1 to the south,
            +2 to the east and -2 to the west.
        :param start_point: cartesian point where the peak plasma frequency is f0
        :param profile_exponent: see class docstring
        """
        self._parameters = np.array([f0, hm, ym])
        self.gradient = gradient
        self.profile_exponent = profile_exponent

        # Cartesian point at which the peak plasma frequency equals f0
        self.start_point = np.asarray(start_point, dtype=float)
        self._start_spherical = cartesian_to_spherical(self.start_point).reshape(3)

    def peak_frequency(self, coordinate: ArrayLike) -> ArrayLike:
        """Peak plasma frequency (Hz) above each cartesian coordinate, shape (N,)"""
        spherical = cartesian_to_spherical(np.asarray(coordinate, dtype=float)).reshape(-1, 3)
        multiplier = np.full(len(spherical), self._parameters[0])
        if self.gradient is not None:
            index = abs(int(self.gradient[1]))
            delta = spherical[:, index] - self._start_spherical[index]
            if index == 2:
                # Longitude differences have to be wrapped
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
            multiplier = multiplier + self.gradient[0] * np.sign(self.gradient[1]) * delta
        return np.maximum(multiplier, 0)

    def plasma_frequency(self, coordinate: ArrayLike) -> ArrayLike:
        coordinate = np.asarray(coordinate, dtype=float)
        h = np.linalg.norm(coordinate.reshape(-1, 3), axis=1) - EARTH_RADIUS
        z1 = (h - self._parameters[1]) / self._parameters[2]

        # exp(-z1) overflows far below the layer where the density is zero anyway
        z1 = np.maximum(z1, -30)
        output = self.peak_frequency(coordinate) * np.exp(0.5 * self.profile_exponent * (1 - z1 - np.exp(-z1)))

        if coordinate.ndim == 1:
            return output[0]
        else:
            return output


class QuasiParabolic(BaseAtmosphere):
    """
    This class is an implementation of the quasi parabolic
    atmosphere definition (Croft and Hoogasian 1968),

        f_p^2 = f0^2 * [1 - ((r - r_m) / ym)^2 * (r_b / r)^2],   r_m = R_E + hm,  r_b = r_m - ym

    and zero wherever that expression is negative. See BaseAtmosphere for
    documentation on how property definitions used in this class
    """
    def __init__(self, f0, hm, ym):
        """
        :param f0: peak plasma frequency in Hz
        :param hm: altitude of the peak in meters (above the ground, NOT from the earth's centre)
        :param ym: layer semi-thickness in meters
        """
        self._parameters = np.array([f0, hm, ym])

    def plasma_frequency(self, coordinate: ArrayLike) -> ArrayLike:
        coordinate = np.asarray(coordinate, dtype=float)
        r = np.linalg.norm(coordinate.reshape(-1, 3), axis=1)

        f0, hm, ym = self._parameters
        rm = EARTH_RADIUS + hm
        rb = rm - ym
        output = f0 * np.sqrt(np.maximum(1 - np.square((r - rm) * rb / (r * ym)), 0))

        if coordinate.ndim == 1:
            return output[0]
        else:
            return output
