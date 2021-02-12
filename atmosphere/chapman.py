from utils import coordinates as coords
import typing
import numpy as np


def calculate_e_density(heights, *atmosphere_params: float) -> np.ndarray:
    """
    Returns the chapman layers e density given points in path_component format and
    atmosphere_params
    :param heights: Array of heights as measured above the center of the earth
    :param atmosphere_params: a tuple of (atmosphere_height_of_max, atmosphere_semi_width, atmosphere_maximum_e_density)
    :return: Array of e densities given the chapman layers point
    """
    atmosphere_height_of_max, atmosphere_semi_width, \
        atmosphere_maximum_e_density, operating_frequency = atmosphere_params
    z1 = ((heights + coords.EARTH_RADIUS) - atmosphere_height_of_max) / atmosphere_semi_width

    return atmosphere_maximum_e_density * np.exp((1 - z1 - np.exp(-z1)))


def to_qp_params(*chapman_params: float) -> typing.Tuple:
    atmosphere_height_of_max, atmosphere_semi_width, \
        atmosphere_maximum_e_density, operating_frequency = chapman_params

    return atmosphere_height_of_max, atmosphere_height_of_max - atmosphere_semi_width, \
        atmosphere_maximum_e_density, operating_frequency
