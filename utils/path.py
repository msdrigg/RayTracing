from scipy import interpolate
import numpy as np


def generate_callable(input_points: np.ndarray,
                      output_values: np.ndarray,
                      degree: int = 2,
                      **kwargs) -> interpolate.UnivariateSpline:
    """
    This function generates a python callable f that can be used like f(input_array) -> output_array,
    or f(input_array, n) -> n'th derivative evaluated at input_array.
    :param input_points: A 1D array of size (N, ) representing the x values to interpolate against
    :param output_values: An array of size (N, n) representing the y values to interpolate between
    :param degree: The degree polynomial to fit. For example, degree = 3 for a cubic spline
    """
    poly_fit = interpolate.InterpolatedUnivariateSpline(input_points, output_values, k=degree, **kwargs)
    return poly_fit
