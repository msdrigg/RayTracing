from scipy import interpolate
import numpy as np


def generate_cartesian_callable(
        path_components: np.ndarray,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        degree: int = 2,
        **kwargs) -> interpolate.UnivariateSpline:
    """
    This function generates a python callable f that can be used like f(input_array) -> output_array,
    or f(input_array, n) -> n'th derivative evaluated at input_array.
    :param path_components: An array of size (N, 3) representing the path components defining the path we are
    creating an interpolation of
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param degree: The degree polynomial to fit. For example, degree = 3 for a cubic spline
    :returns: A callable with signature (np.ndarray) -> np.ndarray where the input is a vector of points in
    the interval (0, 1) and the output is an array whose rows are the cartesian vectors corresponding to the input
    points. This callable under the hood is a collection of several interpolate.InterpolatedUnivariateSpline's
    """
    # TODO: Implement this: return a function that when called with an array of points, returns a cartesian array
    poly_fit = interpolate.InterpolatedUnivariateSpline(path_components, output_values, k=degree, **kwargs)
    return poly_fit