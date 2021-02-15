# noinspection SpellCheckingInspection
"""
This file implements the chapman layers atmosphere.
To see the required atmosphere function definitions look at atmosphere.base

The equations in this file were gotten from
www.uio.no/studier/emner/matnat/fys/nedlagte-emner/FYS3610/h04/undervisningsmateriale/Chapter%204-25August.pdf
https://trs.jpl.nasa.gov/bitstream/handle/2014/40912/08-24.pdf?sequence=1
"""

from numpy.typing import *
import numpy as np
from scipy import linalg


def calculate_plasma_frequency_squared(
        position_vector: ArrayLike,
        norms: ArrayLike = None,
        *chapman_params) -> np.ndarray:
    """
    This function calculates the plasma frequency of the atmosphere at different points.
    This follows
    See atmosphere.base for a more detailed description
    """
    if norms is None:
        norms = linalg.norm(position_vector, axis=1)

    atmosphere_height_of_max, atmosphere_semi_width, maximum_plasma_frequency_squared = chapman_params

    z1 = (norms - atmosphere_height_of_max) / atmosphere_semi_width

    return maximum_plasma_frequency_squared * np.exp((1 - (z1 + np.exp(-z1))) / 2)
