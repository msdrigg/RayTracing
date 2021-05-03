from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import UnivariateSpline
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
            total_angle * EARTH_RADIUS, f, rm + EARTH_RADIUS, ym, fc ** 2
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
    implementation of BasePath reflecting the GCD path
    """
    def __init__(
            self,
            radial_parameters: ArrayLike,
            angular_parameters: ArrayLike,
            initial_point: ArrayLike,
            final_point: ArrayLike
    ):
        """
        supply the radial_parameters as a array of (fractional_position, radial_distance) and angular_parameters
        as an array of (fractional_position, angular_deviation), initial_point in cartesian and final_point in
        cartesian.

        Note: radial_parameters needs to have a value at 0 and at 1 (endpoints of our interval), but these values
        will be fixed and won't be included in our adjustable_parameters function because these don't get optimized.

        :param radial_parameters : array_like, shape (N,2)
            each row of this vector is of the format
            (fractional amount along path from 0 to 1, height (including earth's radius) at this path point)
            these are interpolated between to get path
        :param angular_parameters : array_like, shape (N,2)
            same format as radial_parameters, but parameter values are in radians
        :param initial_point : array_like, shape (3,)
            the cartesian coordinates of the path start
        :param final_point : array_like, shape (3,)
            the cartesian coordinates of the path end
        """
        super().__init__()

        # Set the position of all parameters by concatenating the lists of radial deviations and angular deviations
        # Position is the angular location along the great circle connecting initial and final points

        # Each parameter is represented by a 2-vector: (position, value)
        # The the start and end points are fixed
        self._radial_parameters = np.copy(radial_parameters)
        self._angular_parameters = np.copy(angular_parameters)

        # Get the initial and final points from the initializer state
        self.initial_point = np.copy(initial_point)
        self.final_point = np.copy(final_point)

        # Get vector normal to path
        self.normal_vec = Vector.unit_vector(
            np.cross(self.initial_point, self.final_point)
        )

        # Declare other private variables
        self._total_angle = None

        self._poly_fit_angular = None
        self._poly_fit_radial = None
        self._poly_fit_cartesian = None

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
        # Take initial and final points from the path
        initial_point = other_path(0)
        final_point = other_path(1)

        # Take radial parameters from the norm of the positions along the path
        radial_parameter_path_components = Coordinates.standard_to_path_component(
            other_path(radial_parameter_locations), initial_point, final_point
        )
        radial_params = np.column_stack([
            radial_parameter_locations,
            radial_parameter_path_components[:, 0]
        ])

        angular_parameter_path_components = Coordinates.standard_to_path_component(
            other_path(angular_parameter_locations), initial_point, final_point
        )
        angular_params = np.column_stack([
            angular_parameter_locations,
            angular_parameter_path_components[:, 2]
        ])

        return GreatCircleDeviation(
            radial_params, angular_params, initial_point, final_point
        )

    @property
    def radial_points(self):
        return self._radial_parameters

    @property
    def adjustable_parameters(self):
        # We don't vary the first or last parameters (these are the fixed points on the path).
        return np.concatenate((self._radial_parameters[1:-1, 1], self._angular_parameters[1:-1, 1]))

    def transform_adjustments(self, indexes: ArrayLike, adjustments: ArrayLike) -> ArrayLike:
        # convert adjustments in angle into radians
        adjustments[indexes >= self._radial_parameters.shape[0] - 2] = \
            adjustments[indexes >= self._radial_parameters.shape[0] - 2] / EARTH_RADIUS
        return adjustments

    @adjustable_parameters.setter
    def adjustable_parameters(self, value):
        new_radial_parameters, new_angular_parameters = np.split(
            value, [self._radial_parameters.shape[0] - 2]
        )
        self._radial_parameters[1:-1, 1] = new_radial_parameters
        self._angular_parameters[1:-1, 1] = new_angular_parameters

        # Redefine the poly_fits that control this method
        self.interpolate_params()

    def __copy__(self):
        return GreatCircleDeviation(
            self._radial_parameters,
            self._angular_parameters,
            self.initial_point,
            self.final_point
        )

    def __call__(self, fraction, nu=0):
        _ = self.poly_fit_cartesian
        point = np.array(list(map(lambda poly_fit: poly_fit(fraction, nu=nu), self._poly_fit_cartesian))).T

        if np.isscalar(fraction):
            return point.flatten()
        else:
            return point

    @property
    def poly_fit_angular(self):
        if self._poly_fit_angular is None:
            self.interpolate_params()
        return self._poly_fit_angular

    @property
    def poly_fit_radial(self):
        if self._poly_fit_radial is None:
            self.interpolate_params(radial=True)
        return self._poly_fit_radial

    @property
    def poly_fit_cartesian(self):
        if self._poly_fit_cartesian is None:
            self.interpolate_params()
        return self._poly_fit_cartesian

    def interpolate_params(self, radial=False, degree=3):
        self._total_angle = Vector.angle_between(self.initial_point, self.final_point)

        self._poly_fit_angular = UnivariateSpline(
            self._angular_parameters[:, 0],
            self._angular_parameters[:, 1],
            k=min(degree, len(self._angular_parameters) - 1),
            s=0,
            ext=0
        )
        if radial:
            self._poly_fit_radial = UnivariateSpline(
                self._radial_parameters[:, 0],
                self._radial_parameters[:, 1],
                k=degree,
                s=0,
                ext=0
            )
        else:
            cartesian_points = np.zeros((len(self._radial_parameters), 3))
            for index in range(len(self._radial_parameters)):
                alpha = self._radial_parameters[index, 0]
                r_1 = Rotation.from_rotvec(self.normal_vec * alpha * self.total_angle)
                v_1 = r_1.apply(Vector.unit_vector(self.initial_point))
                rotation_vec_2 = Vector.unit_vector(np.cross(self.normal_vec, v_1))
                rotation_vec_2 *= self._poly_fit_angular(alpha)
                r_2 = Rotation.from_rotvec(rotation_vec_2)
                v_2 = r_2.apply(v_1)
                v_2 *= self._radial_parameters[index, 1]
                cartesian_points[index] = v_2
            self._poly_fit_cartesian = []
            for index in range(3):
                self._poly_fit_cartesian.append(
                    UnivariateSpline(
                        self._radial_parameters[:, 0],
                        cartesian_points[:, index],
                        k=degree,
                        s=0
                    )
                )
