"""
Implements a 0 magnetic field.
Because magnetic_field_vec must be normalized, I set all vectors to (0, 0, 1) -- pointing upward in the z axis
"""

import numpy as np
from numpy.typing import *
from typing import Optional


# noinspection PyUnusedLocal
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
    return np.zeros(position_vector.shape[0])


# noinspection PyUnusedLocal
def calculate_magnetic_field_unit_vec(
        position_vector: ArrayLike,
        norms: Optional[ArrayLike] = None) -> ArrayLike:
    """
    This function calculates the unit magnetic field vector (magnitude 1) at different points.
    We split this up with the previous function to optimize it.
    :param position_vector: This is an array of shape (N, 3) whose rows contain cartesian coordinates
    :param norms: This is an array of shape (N,) whose elements are the norms of the corresponding cartesian points
    This parameter is an optional speed up to reduce the repeated calculating of norms.
    :returns: A (N, 3) array whose rows are the magnetic field vectors in cartesian coordinates
    """
    result = np.zeros_like(position_vector)

    # We can't actually return a 0 vector.
    # We need it to be normalized, so we return a constant vector whose only component is a unit z component
    result[:, 2] = 1

    return result
