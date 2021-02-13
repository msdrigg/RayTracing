"""
This class does not define any method bodies, but it does define
the required functions and their call signatures that all implementations of atmosphere must provide
"""
from numpy.typing import *
from typing import Optional


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
    pass


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
    pass

