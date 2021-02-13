"""
This file implements the chapman layers atmosphere.
To see the required atmosphere function definitions look at atmosphere.base
"""

from numpy.typing import *
import numpy as np
import typing


def calculate_plasma_frequency_squared(
        position_vector: ArrayLike,
        norms: ArrayLike = None,
        *chapman_params) -> np.ndarray:
    """
    This function calculates the plasma frequency of the atmosphere at different points.
    This follows
    See atmosphere.base for a more detailed description
    """
    # Equation, but with an additional factor of 2. I have adjusted to add this factor of 2 because without it,
    # we are only matching ion_production, not electron density.
    # www.uio.no/studier/emner/matnat/fys/nedlagte-emner/FYS3610/h04/undervisningsmateriale/Chapter%204-25August.pdf
    atmosphere_height_of_max, atmosphere_semi_width, maximum_plasma_frequency_squared = chapman_params

    z1 = (norms - atmosphere_height_of_max) / atmosphere_semi_width

    return maximum_plasma_frequency_squared * np.exp((1 - (z1 + np.exp(-z1))) / 2)


def get_qp_parameters(*atmosphere_params: float) -> typing.Tuple[float, ...]:
    """
    This method calculates qp parameters. See atmosphere.base for a more detailed description
    """
    atmosphere_height_of_max, atmosphere_semi_width, maximum_gyro_frequency = atmosphere_params

    return atmosphere_height_of_max, atmosphere_height_of_max - atmosphere_semi_width, maximum_gyro_frequency