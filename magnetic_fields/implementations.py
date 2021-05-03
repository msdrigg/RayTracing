import numpy as np

from utilities.Constants import B_FIELD, EARTH_RADIUS
from utilities.Vector import cartesian_to_spherical, unit_radius, unit_theta, unit_vector, spherical_to_cartesian
from magnetic_fields import BaseField


class DipoleField(BaseField):
    """
    Implementation of BaseField that follows the dipole model of earth's magnetic field
    """
    def __init__(self):
        self.b_max = B_FIELD
        self.re3 = np.power(EARTH_RADIUS, 3)  # E_Radius^3

    def field_vec(self, position):
        position = cartesian_to_spherical(position).reshape(-1, 3)

        # radii cubed is 1/r^3 for all r
        radii_cubed = np.power(position[:, 0], -3)
        cos_thetas = np.cos(position[:, 1])
        b_vec = -2*self.b_max*radii_cubed.reshape(-1, 1)*self.re3*cos_thetas.reshape(-1, 1)*unit_radius(position) - \
            self.b_max*self.re3*radii_cubed.reshape(-1, 1)*np.sin(position[:, 1]).reshape(-1, 1)*unit_theta(position)
        b_vec = unit_vector(b_vec)

        if len(b_vec) == 1:
            return b_vec[0]
        else:
            return b_vec

    def field_mag(self, position):
        position = cartesian_to_spherical(position).reshape(-1, 3)
        # radii cubed is 1/r^3 for all r
        radii_cubed = np.power(position[:, 0], -3)
        cos_thetas = np.cos(position[:, 1])
        b_mags = self.b_max * self.re3 * radii_cubed * np.sqrt(1 + 3 * np.square(cos_thetas))
        if len(b_mags) == 1:
            return b_mags[0]
        else:
            return b_mags


class ZeroField(BaseField):
    """
    Implementation of Base field that always returns 0
    """
    def field_vec(self, position):
        # Field vector is normalized, so we return 1, 0, 0 as normal vector
        return spherical_to_cartesian(np.array([1, 0, 0]))

    def field_mag(self, position):
        return np.repeat(0, len(position))

    def gyro_frequency(self, position):
        return np.repeat(0, len(position))
