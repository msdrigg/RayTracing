import numpy as np
from utils import coordinates as coords
import typing


def to_chapman_params(*qp_params: float) -> typing.Tuple:
    atmosphere_height_of_max, atmosphere_base_height, \
        atmosphere_maximum_e_density, operating_frequency = qp_params

    return atmosphere_height_of_max, atmosphere_height_of_max - atmosphere_base_height, \
        atmosphere_maximum_e_density, operating_frequency


def e_density_helper(heights: np.ndarray, *atmosphere_params: float) -> np.ndarray:
    """
    // NOTE: UNTESTED
    Returns the e density as a numpy array for the given heights. Heights are measured from origin not surface
    :param heights: array of heights as measured from the surface of the earth
    :returns: array of plasma frequency at given heights
    """
    radii = heights + coords.EARTH_RADIUS
    atmosphere_height_of_max, atmosphere_base_height, atmosphere_max_e_density, operating_frequency = atmosphere_params
    semi_width = atmosphere_height_of_max - atmosphere_base_height

    term_2_numerator = (radii - atmosphere_height_of_max) * atmosphere_base_height
    term_2_denominator = semi_width * radii
    term_2 = np.square(term_2_numerator / term_2_denominator)

    in_atmosphere_density = atmosphere_max_e_density * (1 - term_2)

    return np.where(
        (atmosphere_base_height < radii) & (atmosphere_base_height * atmosphere_height_of_max /
                                            (atmosphere_base_height - semi_width) > radii),
        atmosphere_max_e_density * in_atmosphere_density,
        0
    )
