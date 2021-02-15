"""
This file implements the chapman layers atmosphere.
To see the required atmosphere function definitions look at atmosphere.base
"""

from numpy.typing import *
import numpy as np
from scipy import linalg


# noinspection PyUnusedLocal
def calculate_plasma_frequency_squared(position_vector: ArrayLike,
                                       norm_vector: ArrayLike,
                                       *atmosphere_params) -> np.ndarray:
    """
    Follows the description in atmosphere.base to calculate plasma frequency for given positions
    """
    if norm_vector is None:
        norm_vector = linalg.norm(position_vector, axis=1)

    atmosphere_height_of_max, semi_width, max_plasma_frequency_squared = atmosphere_params
    atmosphere_base_height = atmosphere_height_of_max - semi_width

    term_2_numerator = (norm_vector - atmosphere_height_of_max) * atmosphere_base_height
    term_2_denominator = semi_width * norm_vector
    term_2 = np.square(term_2_numerator / term_2_denominator)
    in_atmosphere_plasma_frequency = max_plasma_frequency_squared * (1 - term_2)

    return np.maximum(in_atmosphere_plasma_frequency, 0)
