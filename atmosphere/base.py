"""
This class does not define any method bodies, but it does define
the required functions and their call signatures that all implementations of atmosphere must provide
"""
import typing
from numpy.typing import *


def calculate_plasma_frequency_squared(
        position_vector: ArrayLike,
        norms: ArrayLike = None,
        *atmosphere_params) -> ArrayLike:
    """
    This function calculates the plasma frequency squared of the atmosphere at different points.
    Most derivations of this use E_Density. But plasma_frequency^2 is proportional to e_density.
    So if we use atmosphere_params with maximum_plasma_frequency or (critical_frequency), then we
    can apply the same equation as if we were calculating e_density
    :param position_vector: This is an array of shape (N, 3) whose rows contain cartesian coordinates
    :param norms: This is an array of shape (N,) whose elements are the norms of the corresponding cartesian points
    This parameter is an optional speed up to reduce the repeated calculating of norms.
    :returns: A vector whose elements are the e density evaluated at the provided cartesian coordinates
    """
    pass


def get_qp_parameters(*atmosphere_params) -> typing.Tuple[float]:
    """
    Because we use qp path for initialization, we need a method to calculate the qp parameters from any atmosphere's
    parameters.
    :returns: A tuple
    (atmosphere_height_of_max, atmosphere_base_height, atmosphere_max_gyro_frequency) (all floats)
    """
    pass
