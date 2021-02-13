"""
This file implements the chapman layers atmosphere.
To see the required atmosphere function definitions look at atmosphere.base
"""

from numpy.typing import *
import numpy as np
import typing


def get_qp_parameters(*atmosphere_params: float) -> typing.Tuple[float, ...]:
    """
    Because we use qp path for initialization, we need a method to calculate the qp parameters from any atmosphere's
    parameters.
    :returns: A tuple
    (atmosphere_height_of_max, atmosphere_base_height, atmosphere_max_gyro_frequency) (all floats)
    """
    return atmosphere_params


# noinspection PyUnusedLocal
def calculate_plasma_frequency(position_vector: ArrayLike,
                               norm_vector: ArrayLike,
                               *atmosphere_params) -> np.ndarray:
    """
    Follows the description in atmosphere.base to calculate plasma frequency for given positions
    """
    atmosphere_height_of_max, atmosphere_base_height, max_plasma_frequency = atmosphere_params
    semi_width = atmosphere_height_of_max - atmosphere_base_height

    term_2_numerator = (norm_vector - atmosphere_height_of_max) * atmosphere_base_height
    term_2_denominator = semi_width * norm_vector
    term_2 = np.square(term_2_numerator / term_2_denominator)

    in_atmosphere_plasma_frequency = max_plasma_frequency * (1 - term_2)

    return np.where(
        (atmosphere_base_height < norm_vector) &
        (atmosphere_base_height * atmosphere_height_of_max /
         (atmosphere_base_height - semi_width) > norm_vector),
        in_atmosphere_plasma_frequency,
        0
    )
