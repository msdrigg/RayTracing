from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import UnivariateSpline, make_interp_spline
from scipy.spatial.transform import Rotation
from scipy import linalg

from utilities import Vector
from utilities import Coordinates
from paths import quasi_parabolic_core
from utilities.Constants import EARTH_RADIUS
from paths import BasePath


class QuasiParabolic(BasePath):
    """
    implementation of the BasePath to follow QuasiParabolic model
    """
    def __init__(
            self, initial_coordinate: ArrayLike,
            final_coordinate: ArrayLike,
            atmosphere_params: ArrayLike,
            operating_frequency: float,
            point_number: Optional[int] = None,
            use_high_ray: Optional[bool] = True,
    ):
        """
        supply initial and final points, atmosphere parameters (f_0, r_m, y_m) and operating_frequency (f)

        :param initial_coordinate : array_like, shape (3,)
            the cartesian coordinates of the path start
        :param final_coordinate : array_like, shape (3,)
            the cartesian coordinates of the path end
        :param atmosphere_params : Tuple, shape (3,)
            the atmosphere parameters as a tuple of
            (max_plasma_frequency (f_0), height_of_max_plasma_frequency(r_m), layer_semi_width(y_m))
        :param operating_frequency: float
            The operating frequency of the ray
        :param point_number : int
            The number of points to interpolate between in the final path
        :param use_high_ray : bool
            Whether or not to use high ray
        """
        super().__init__()

        # Initial and final points are normalized, but poly_fit(0) and poly_fit(1) will return the radial component
        # for the initial and final point
        initial_rad, final_rad = linalg.norm(initial_coordinate), linalg.norm(final_coordinate)
        self.initial_point = initial_coordinate / linalg.norm(initial_coordinate)
        self.final_point = final_coordinate / linalg.norm(final_coordinate)

        self.normal_vec = np.cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec / linalg.norm(self.normal_vec)

        angle_between = Vector.angle_between(initial_coordinate, final_coordinate)

        # Parameters for quasi-parabolic path only depend on atmospheric model and ignore magnetic field

        self._parameters = self.calculate_parameters(atmosphere_params, operating_frequency)
        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(angle_between * EARTH_RADIUS)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value at each point (in km)
        self.points = np.zeros((self.point_number, 2))
        self.points[:, 0] = np.linspace(0, angle_between, self.point_number)
        self.points[:, 1] = np.linspace(initial_rad, final_rad, self.point_number)

        # Real representation of the line will be a poly fit of radius vs angle along great circle
        self._poly_fit = None
        self._total_angle = None
        self.using_high_ray = use_high_ray

        # Spacing (meters of ground range) of the points the analytic path is sampled at
        self.step_size_horizontal = 500.0

    @property
    def adjustable_parameters(self):
        # Parameters are the parameters defining the parabolic atmosphere
        # Examples are upper and lower radius, midpoint, maximum, etc.
        return self._parameters

    @adjustable_parameters.setter
    def adjustable_parameters(self, new_parameters):
        self._parameters = new_parameters
        self.compile_points()

    def __call__(self, fraction, **kwargs):
        # Nu is not allowed as a parameter
        if "nu" in kwargs and kwargs.get("nu") != 0:
            raise (NotImplementedError("Quasi-parabolic class does not have derivative capabilities."))
        if self._poly_fit is None:
            self.compile_points()
        alpha = fraction * self.points[-1, 0]

        rotations = Rotation.from_rotvec(np.outer(alpha, self.normal_vec))
        rotated_vecs = rotations.apply(self.initial_point)
        output_vecs = rotated_vecs * self._poly_fit(alpha).reshape(-1, 1)
        if np.isscalar(fraction):
            return output_vecs[0]
        else:
            return output_vecs

    def compile_points(self):
        fc, rm, rb, ym, f = self._parameters
        total_angle = Vector.angle_between(self.initial_point, self.final_point)

        points_unprocessed = quasi_parabolic_core.get_quasi_parabolic_path(
            total_angle * EARTH_RADIUS, f, rm + EARTH_RADIUS, ym, fc ** 2,
            step_size_horizontal=self.step_size_horizontal
        )
        if self.using_high_ray or len(points_unprocessed) == 1:
            points_unprocessed = points_unprocessed[0]
        else:
            points_unprocessed = points_unprocessed[1]

        points_unprocessed[:, 1] = points_unprocessed[:, 1]
        points_unprocessed[:, 0] = points_unprocessed[:, 0] / EARTH_RADIUS
        self.points = points_unprocessed

        self._total_angle = total_angle

        self._poly_fit = UnivariateSpline(
            points_unprocessed[:, 0], points_unprocessed[:, 1],
            k=3, s=0, ext=0
        )

    @property
    def poly_fit(self):
        if self._poly_fit is None:
            self.compile_points()
        return self._poly_fit

    @staticmethod
    def calculate_parameters(atmosphere_params, wave_frequency):
        f_max = atmosphere_params[0]
        rm = atmosphere_params[1]
        ym = atmosphere_params[2]
        rb = rm - ym
        return np.array([f_max, rm, rb, ym, wave_frequency])

    def __copy__(self):
        atmosphere_params = self._parameters[
            [0, 1, 3]
        ].tolist()
        return QuasiParabolic(
            self.initial_point, self.final_point,
            atmosphere_params,
            self._parameters[4],
            point_number=self.point_number,
            use_high_ray=self.using_high_ray
        )


class GreatCircleDeviation(BasePath):
    """
    implementation of BasePath reflecting the GCD path.

    With u(s) the unit vector that sweeps along the great circle from the start
    point (s = 0) to the end point (s = 1) and n the unit normal of the great
    circle plane, the path is

        r(s) = R(s) * u(s) + b(s) * n

    R(s) is a cubic spline through the radial parameters (meters from the
    earth's centre) and b(s) a cubic spline through the lateral parameters
    (meters of deviation out of the great circle plane). Both splines are
    linear in their knot values, so r(s) is LINEAR in the adjustable
    parameters. The tracer relies on this to build an exact Hessian.
    """
    is_linear_in_parameters = True

    def __init__(
            self,
            radial_parameters: ArrayLike,
            angular_parameters: ArrayLike,
            initial_point: ArrayLike,
            final_point: ArrayLike
    ):
        """
        supply the radial_parameters as a array of (fractional_position, radial_distance) and angular_parameters
        as an array of (fractional_position, lateral_deviation), initial_point in cartesian and final_point in
        cartesian.

        Note: both parameter arrays need to have a value at 0 and at 1 (endpoints of our interval), but these values
        are fixed and aren't included in adjustable_parameters because they don't get optimized.

        :param radial_parameters : array_like, shape (N,2)
            each row of this vector is of the format
            (fractional amount along path from 0 to 1, distance from the earth's centre at this path point in meters)
            these are interpolated between to get path
        :param angular_parameters : array_like, shape (M,2)
            same format as radial_parameters, but parameter values are the lateral deviation from
            the great circle plane in meters (positive along cross(initial_point, final_point))
        :param initial_point : array_like, shape (3,)
            the cartesian coordinates of the path start
        :param final_point : array_like, shape (3,)
            the cartesian coordinates of the path end
        """
        super().__init__()

        # Each parameter is represented by a 2-vector: (position, value)
        # The the start and end points are fixed
        self._radial_parameters = np.array(radial_parameters, dtype=float)
        self._angular_parameters = np.array(angular_parameters, dtype=float)

        # Get the initial and final points from the initializer state
        self.initial_point = np.array(initial_point, dtype=float)
        self.final_point = np.array(final_point, dtype=float)

        # Orthonormal frame of the great circle plane
        self._e1 = Vector.unit_vector(self.initial_point)
        self.normal_vec = Vector.unit_vector(
            np.cross(self.initial_point, self.final_point)
        )
        self._e2 = np.cross(self.normal_vec, self._e1)
        self._total_angle = Vector.angle_between(self.initial_point, self.final_point)

        self._poly_fit_angular = None
        self._poly_fit_radial = None
        self._basis_cache = None

    @staticmethod
    def from_path(
            radial_parameter_locations: ArrayLike,
            angular_parameter_locations: ArrayLike,
            other_path: BasePath
    ):
        """
        helper function to create GCD path given another path and interpolation points.

        :param radial_parameter_locations : array_like, shape (N,)
            Similar format as radial_parameters in the __init__ function, but not including values, only path points
        :param angular_parameter_locations : array_like, shape (N,)
            Similar format as angular_parameters in the __init__ function, but not including values, only path points
        :param other_path : BasePath
            This is the function that will be used to determine the path location.
        """
        radial_parameter_locations = np.asarray(radial_parameter_locations, dtype=float)
        angular_parameter_locations = np.asarray(angular_parameter_locations, dtype=float)

        # Take initial and final points from the path
        initial_point = other_path(0)
        final_point = other_path(1)
        normal_vec = Vector.unit_vector(np.cross(initial_point, final_point))

        # In-plane distance from the earth's centre at the radial locations
        radial_points = other_path(radial_parameter_locations)
        in_plane = radial_points - np.outer(radial_points @ normal_vec, normal_vec)
        radial_params = np.column_stack([
            radial_parameter_locations,
            linalg.norm(in_plane, axis=-1)
        ])

        # Out of plane distance at the lateral locations
        angular_params = np.column_stack([
            angular_parameter_locations,
            other_path(angular_parameter_locations) @ normal_vec
        ])
        # The end points are in the plane by definition
        angular_params[[0, -1], 1] = 0

        return GreatCircleDeviation(
            radial_params, angular_params, initial_point, final_point
        )

    @property
    def radial_points(self):
        return self._radial_parameters

    @property
    def lateral_points(self):
        return self._angular_parameters

    @property
    def radial_parameter_number(self):
        return self._radial_parameters.shape[0] - 2

    @property
    def quadrature_breakpoints(self):
        # The path is a piecewise polynomial between the radial knots, so those are
        # the natural panel boundaries for integrating along it
        return self._radial_parameters[:, 0]

    @property
    def adjustable_parameters(self):
        # We don't vary the first or last parameters (these are the fixed points on the path).
        return np.concatenate((self._radial_parameters[1:-1, 1], self._angular_parameters[1:-1, 1]))

    @adjustable_parameters.setter
    def adjustable_parameters(self, value):
        new_radial_parameters, new_angular_parameters = np.split(
            value, [self._radial_parameters.shape[0] - 2]
        )
        self._radial_parameters[1:-1, 1] = new_radial_parameters
        self._angular_parameters[1:-1, 1] = new_angular_parameters
        self._poly_fit_radial = None
        self._poly_fit_angular = None

    def __copy__(self):
        new_path = GreatCircleDeviation(
            self._radial_parameters,
            self._angular_parameters,
            self.initial_point,
            self.final_point
        )
        # The basis functions only depend on the knot locations, which copies share
        new_path._basis_cache = self._basis_cache
        return new_path

    @staticmethod
    def _spline(parameters: np.ndarray, values: np.ndarray):
        return make_interp_spline(parameters[:, 0], values, k=min(3, len(parameters) - 1))

    @property
    def poly_fit_radial(self):
        if self._poly_fit_radial is None:
            self._poly_fit_radial = self._spline(self._radial_parameters, self._radial_parameters[:, 1])
        return self._poly_fit_radial

    @property
    def poly_fit_angular(self):
        if self._poly_fit_angular is None:
            self._poly_fit_angular = self._spline(self._angular_parameters, self._angular_parameters[:, 1])
        return self._poly_fit_angular

    def _great_circle_frame(self, fraction: np.ndarray):
        """u(s) and its first two derivatives with respect to s, each of shape (N, 3)"""
        angle = fraction * self._total_angle
        cos, sin = np.cos(angle)[:, np.newaxis], np.sin(angle)[:, np.newaxis]
        u = cos * self._e1 + sin * self._e2
        du = self._total_angle * (-sin * self._e1 + cos * self._e2)
        ddu = -self._total_angle ** 2 * u
        return u, du, ddu

    def __call__(self, fraction, nu=0):
        scalar_input = np.isscalar(fraction)
        fraction = np.atleast_1d(np.asarray(fraction, dtype=float))
        u, du, ddu = self._great_circle_frame(fraction)
        radial, lateral = self.poly_fit_radial, self.poly_fit_angular

        if nu == 0:
            point = radial(fraction)[:, np.newaxis] * u
        elif nu == 1:
            point = radial(fraction, 1)[:, np.newaxis] * u + radial(fraction)[:, np.newaxis] * du
        elif nu == 2:
            point = radial(fraction, 2)[:, np.newaxis] * u + 2 * radial(fraction, 1)[:, np.newaxis] * du + \
                radial(fraction)[:, np.newaxis] * ddu
        else:
            raise NotImplementedError("GreatCircleDeviation only provides derivatives up to second order")
        point = point + np.outer(lateral(fraction, nu), self.normal_vec)

        if scalar_input:
            return point[0]
        else:
            return point

    def parameter_jacobian(self, fraction: ArrayLike):
        """
        Exact derivatives of the position and of the path derivative with respect to every adjustable parameter.
        See BasePath.parameter_jacobian
        """
        fraction = np.atleast_1d(np.asarray(fraction, dtype=float))
        if self._basis_cache is None or not np.array_equal(self._basis_cache[0], fraction):
            # Splines are linear in their knot values, so interpolating the identity matrix
            # gives the response of the spline to a unit change of each knot
            radial_basis = self._spline(self._radial_parameters, np.eye(len(self._radial_parameters)))
            lateral_basis = self._spline(self._angular_parameters, np.eye(len(self._angular_parameters)))
            self._basis_cache = (
                fraction.copy(),
                radial_basis(fraction)[:, 1:-1], radial_basis(fraction, 1)[:, 1:-1],
                lateral_basis(fraction)[:, 1:-1], lateral_basis(fraction, 1)[:, 1:-1]
            )
        _, b_radial, db_radial, b_lateral, db_lateral = self._basis_cache
        u, du, _ = self._great_circle_frame(fraction)

        d_position = np.concatenate((
            u[:, :, np.newaxis] * b_radial[:, np.newaxis, :],
            self.normal_vec[np.newaxis, :, np.newaxis] * b_lateral[:, np.newaxis, :]
        ), axis=2)
        d_derivative = np.concatenate((
            u[:, :, np.newaxis] * db_radial[:, np.newaxis, :] + du[:, :, np.newaxis] * b_radial[:, np.newaxis, :],
            self.normal_vec[np.newaxis, :, np.newaxis] * db_lateral[:, np.newaxis, :]
        ), axis=2)
        return d_position, d_derivative
