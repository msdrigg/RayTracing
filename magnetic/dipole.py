from utils import constants, coordinates as coords
import numpy as np


def field_vec(heights, z_component):
    # radii cubed is 1/r^3 for all r
    radii_cubed = np.power(heights[:, 0] + coords.EARTH_RADIUS, 3)
    cos_thetas = z_component/heights
    b_vec = -2 * self.b_max * radii_cubed.reshape(-1, 1) * constants.re3 * cos_thetas.reshape(-1, 1) * unit_radius(
        position) - \
            self.b_max * self.re3 * radii_cubed.reshape(-1, 1) * sin(position[:, 1]).reshape(-1, 1) * unit_theta(
        position)
    b_vec = unit_vector(b_vec)

    if len(b_vec) == 1:
        return b_vec[0]
    else:
        return b_vec


def field_mag(self, position, using_spherical=False):
    if not using_spherical:
        position = cartesian_to_spherical(position).reshape(-1, 3)
    # radii cubed is 1/r^3 for all r
    radii_cubed = power(position[:, 0], -3)
    cos_thetas = cos(position[:, 1])
    b_mags = self.b_max * self.re3 * radii_cubed * sqrt(1 + 3 * square(cos_thetas))
    if len(b_mags) == 1:
        return b_mags[0]
    else:
        return b_mags


def gyro_frequency(self, position, using_spherical=False):
    """
    This function returns the gyro frequency at the given point using the class's defined functions for
    b_factor and field_mag
    """
    if using_spherical:
        position = spherical_to_cartesian(position)
    return constants.B_FACTOR * self.field_mag(position)
