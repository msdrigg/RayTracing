"""
Implements the dipole model of earths magnetic field. For reference, these calculations were taken from these sources
https://en.wikipedia.org/wiki/Magnetic_dipole
https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
"""
from core import constants, vector
import numpy as np
from core.constants import STANDARD_MAGNETIC_FIELD_MAXIMUM as B_MAX
from core.constants import EARTH_RADIUS_CUBED
from numpy.typing import *
from typing import Optional
from scipy import linalg


def calculate_gyro_frequency(
        position_vector: ArrayLike,
        norms: Optional[ArrayLike] = None) -> ArrayLike:
    """
    This function calculates the gyro frequency at the provided points
    :param position_vector: This is an array of shape (N, 3) whose rows contain cartesian coordinates
    :param norms: This is an array of shape (N,) whose elements are the norms of the corresponding cartesian points
    This parameter is an optional speed up to reduce the repeated calculating of norms.
    :returns: A vector whose elements are the gyro frequency evaluated at the provided cartesian coordinates
    """
    if norms is None:
        norms = linalg.norm(position_vector, axis=1)

    radii_cubed = np.power(norms, -3)
    cos_thetas = position_vector[:, 2] / norms
    
    field_magnitude = B_MAX * EARTH_RADIUS_CUBED * radii_cubed * np.sqrt(1 + 3 * np.square(cos_thetas))

    return constants.B_FACTOR * field_magnitude


def calculate_magnetic_field_unit_vec(
        position_vector: ArrayLike,
        norms: Optional[ArrayLike] = None) -> ArrayLike:
    """
    This function calculates the unit magnetic field vector (magnitude 1) at different points.
    We split this up with the previous function to optimize it. See the equation for B(r) in
    https://en.wikipedia.org/wiki/Magnetic_dipole for equation
    :param position_vector: This is an array of shape (N, 3) whose rows contain cartesian coordinates
    :param norms: This is an array of shape (N,) whose elements are the norms of the corresponding cartesian points
    This parameter is an optional speed up to reduce the repeated calculating of norms.
    :returns: A (N, 3) array whose rows are the magnetic field vectors in cartesian coordinates
    """
    if norms is None:
        norms = linalg.norm(position_vector, axis=1)

    unit_position_vectors = position_vector / norms

    # We are optimizing here because we know that the unit_magnetic_moment of earth's magnetic field is
    # -z_hat. Because the north-south pole confusion.
    field_vector = -3 * unit_position_vectors * unit_position_vectors[:, 2]
    field_vector[:, 2] -= 1

    return vector.normalize_rows(field_vector)
