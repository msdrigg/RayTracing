"""
Utilities to help create and manipulate paths in 3D space
"""
import functools
from typing import Tuple, Optional, Callable, Sequence, Dict

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from utils import coordinates as coords


def _interpolate_multidimensional(
        spline_tck: Sequence[np.ndarray],
        evaluation_points: np.ndarray,
        der: Optional[int] = 0) -> np.ndarray:
    # noinspection SpellCheckingInspection
    """
        Helper function to interpolate between
        :param spline_tck: The spline coefficients defining the spline.
        See scipy.interpolate.splprep
        :param evaluation_points: A vector of points between 0 and 1 to evaluate the
        spline at
        """
    interpolated = interpolate.splev(
        evaluation_points,
        spline_tck,
        ext=3,
        der=der
    )
    return np.stack(interpolated, axis=-1)


def generate_cartesian_callable(
        path_parameters: Tuple[ArrayLike, ArrayLike],
        path_start_spherical: np.ndarray,
        path_end_spherical: np.ndarray,
        degree: int = 2) -> Callable[[np.ndarray, Optional[Dict[str, int]]], np.ndarray]:
    # noinspection SpellCheckingInspection
    """
        This function generates a python callable f that can be used like f(input_array) -> output_array,
        or f(input_array, n) -> n'th derivative evaluated at input_array.
        :param path_parameters: A tuple of (radial parameters, normal parameters)
        :param path_start_spherical: The path start point in spherical coordinates
        :param path_end_spherical: The path end point in spherical coordinates
        :param degree: The degree polynomial to fit. For example, degree = 3 for a cubic spline
        :returns: A callable with signature (np.ndarray, der=0) -> np.ndarray where the input is a vector of points in
        the interval (0, 1) and the output is an array whose rows are the cartesian vectors corresponding to the input
        points. If der>0, then the derivative is returned evaluated at the evaluation points
        """
    path_components = generate_path_components(
        path_parameters,
        degree=degree
    )
    cartesian_vectors = coords.path_component_to_standard(
        path_components,
        path_start_spherical,
        path_end_spherical,
        to_spherical=False
    )

    # noinspection PyTupleAssignmentBalance
    tck, u = interpolate.splprep(cartesian_vectors.T)

    # noinspection SpellCheckingInspection,PyTypeChecker
    return functools.partial(_interpolate_multidimensional, tck)  # noqa


def generate_path_components(
        path_parameters: Tuple[ArrayLike, ArrayLike],
        degree: Optional[int] = 2) -> np.ndarray:
    # noinspection SpellCheckingInspection
    """
        This function generates a vector of path components that match path_parameters
        :param path_parameters: A tuple of (radial parameters, normal parameters)
        :param degree: The degree of interpolation used
        :returns: A np.ndarray of shape (N, 2) whose rows are (radial parameter, normal_parameter)
        """
    evaluated_points = np.zeros((path_parameters[0].size, 2))

    # First interpolate over the normal parameters (because there are less of them)
    normal_interpolated = interpolate.InterpolatedUnivariateSpline(
        np.linspace(0, 1, path_parameters[1].size),
        path_parameters[1],
        k=degree,
        ext='const'
    )

    evaluated_points[:, 0] = path_parameters[1]
    evaluated_points[:, 1] = normal_interpolated(np.linspace(0, 1, path_parameters[0].size))

    return evaluated_points
