import copy
from abc import ABC, abstractmethod

import numpy as np
import typing
from numpy.linalg import norm
from numpy import cross, outer, zeros, \
    linspace, concatenate, array, asarray, amax
from numpy.typing import ArrayLike

import Coordinates
import Initialize
import Vector
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation

from Constants import EARTH_RADIUS, TYPE_ABBREVIATION
from matplotlib import pyplot as plt


class Path(ABC):
    def __init__(self):
        self._total_angle = None

    def visualize(self, fig=None, ax=None, show=True, frequency=None, color=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        if frequency is None:
            _frequency = "?"
        else:
            _frequency = int(frequency/1E6)
        ax.set_title(f"3D Ray Trace")
        ax.autoscale(False)
        if color is None:
            _color = 'black'
        else:
            _color = color
        class_type = self.__class__.__name__
        _label = f'{TYPE_ABBREVIATION[class_type]} - {_frequency} MHz'
        angular_distance = linspace(0, 1, 100)
        radii = np.linalg.norm(self(angular_distance), axis=-1)
        radii = (radii - EARTH_RADIUS) / 1000
        km_range = angular_distance * self.total_angle * EARTH_RADIUS / 1000
        max_x = km_range[-1]
        max_y = amax(radii)
        plt.xlim(-max_x*.05, max_x*1.05)
        plt.ylim(-max_y*.05, max_y*1.05)
        ax.plot(km_range, radii, color=_color, label=_label)
        ax.legend()
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")

        if show:
            plt.show()
        else:
            return fig, ax

    # noinspection PyMethodMayBeStatic
    def transform_adjustments(self, indexes: ArrayLike, adjustments: ArrayLike) -> ArrayLike:
        return adjustments

    def adjust_parameters(self, indexes, adjustments):
        adjusted_params = np.copy(self.adjustable_parameters)
        broadcast_indexes, broadcast_adjustments = [
            np.array(x) for x in np.broadcast_arrays(indexes, adjustments)
        ]
        indexes = asarray(indexes)
        adjusted_params[indexes] = adjusted_params[indexes] + \
            self.transform_adjustments(indexes, broadcast_adjustments)

        # Override the behavior of the __copy__ method to change how this works
        new_path = copy.copy(self)
        new_path.adjustable_parameters = adjusted_params
        return new_path

    @property
    def total_angle(self):
        if self._total_angle is None:
            self._total_angle = Vector.angle_between(self(0), self(1))
        return self._total_angle

    @property
    @abstractmethod
    def adjustable_parameters(self) -> np.ndarray:
        raise NotImplementedError("Inheriting classes must override the adjustable_parameters property")

    @adjustable_parameters.setter
    @abstractmethod
    def adjustable_parameters(self, value):
        raise NotImplementedError(
            "Inheriting classes must override the adjustable_parameters property getter and setter"
        )

    @abstractmethod
    def __call__(self, fraction, nu=0):
        raise NotImplementedError("Inheriting classes must override the __call__ method")


class QuasiParabolic(Path):
    # Currently, must initialize with a cartesian vector
    # This class needs less optimization because its just a starting point.
    def __init__(
            self, initial_coordinates: ArrayLike,
            final_coordinates: ArrayLike,
            atmosphere_params: ArrayLike,
            wave_frequency: float,
            degree: typing.Optional[int] = 4,
            point_number: typing.Optional[int] = None,
            use_high_ray: typing.Optional[bool] = True,
    ):
        super().__init__()

        # Initial and final points are normalized, but poly_fit(0) and poly_fit(1) will return the radial component
        # for the initial and final point
        initial_rad, final_rad = norm(initial_coordinates), norm(final_coordinates)
        self.initial_point = initial_coordinates/norm(initial_coordinates)
        self.final_point = final_coordinates/norm(final_coordinates)

        self.degree = degree

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec/norm(self.normal_vec)

        angle_between = Vector.angle_between(initial_coordinates, final_coordinates)

        # Parameters for quasi-parabolic path only depend on atmospheric model and ignore magnetic field

        self._parameters = self.calculate_parameters(atmosphere_params, wave_frequency)
        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(angle_between*EARTH_RADIUS)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value at each point (in km)
        self.points = zeros((self.point_number, 2))
        self.points[:, 0] = linspace(0, angle_between, self.point_number)
        self.points[:, 1] = linspace(initial_rad, final_rad, self.point_number)

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
            raise(NotImplementedError("Quasi-parabolic class does not have derivative capabilities."))
        if self._poly_fit is None:
            self.compile_points()
        alpha = fraction*self.points[-1, 0]

        rotations = Rotation.from_rotvec(outer(alpha, self.normal_vec))
        rotated_vecs = rotations.apply(self.initial_point)
        output_vecs = rotated_vecs*self._poly_fit(alpha).reshape(-1, 1)
        if np.isscalar(fraction):
            return output_vecs[0]
        else:
            return output_vecs

    def compile_points(self):
        fc, rm, rb, ym, f = self._parameters
        total_angle = Vector.angle_between(self.initial_point, self.final_point)

        points_unprocessed = Initialize.get_quasi_parabolic_path(
            total_angle * EARTH_RADIUS, f, rm + EARTH_RADIUS, ym, fc**2
        )
        if self.using_high_ray:
            points_unprocessed = points_unprocessed[0]
        else:
            points_unprocessed = points_unprocessed[1]

        points_unprocessed[:, 1] = points_unprocessed[:, 1]
        points_unprocessed[:, 0] = points_unprocessed[:, 0] / EARTH_RADIUS
        self.points = points_unprocessed
        
        self._total_angle = total_angle

        self._poly_fit = UnivariateSpline(
            points_unprocessed[:, 0], points_unprocessed[:, 1],
            k=self.degree, s=0, ext=0
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
        return array([f_max, rm, rb, ym, wave_frequency])

    def __copy__(self):
        atmosphere_params = self._parameters[
            [0, 1, 3]
        ].tolist()
        return QuasiParabolic(
            self.initial_point, self.final_point,
            atmosphere_params,
            self._parameters[4],
            degree=self.degree,
            point_number=self.point_number,
            use_high_ray=self.using_high_ray
        )


class GreatCircleDeviation(Path):
    def __init__(
            self,
            radial_parameters: ArrayLike,
            angular_parameters: ArrayLike,
            initial_point: ArrayLike,
            final_point: ArrayLike
    ):
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
            cross(self.initial_point, self.final_point)
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
            other_path: Path
    ):
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

    @staticmethod
    def __old_init__(
            radial_deviations: typing.Union[int, ArrayLike],
            angular_deviations: typing.Union[int, ArrayLike],
            **kwargs
    ):
        self = GreatCircleDeviation(np.array([1, 1]), np.array([1, 1]), np.array([1, 2, 3]), np.array([2, 3, 4]))
        # Set the position of all parameters by concatenating the lists of radial deviations and angular deviations
        # Position is the angular location along the great circle connecting initial and final points
        if isinstance(radial_deviations, int) or isinstance(angular_deviations, int):
            radial_deviations = linspace(0, 1, radial_deviations + 2)
            angular_deviations = linspace(0, 1, angular_deviations + 2)
        else:
            new_radii = zeros((len(radial_deviations) + 2))
            new_thetas = zeros((len(angular_deviations) + 2))
            new_radii[1:-1] = radial_deviations
            new_thetas[1:-1] = angular_deviations
            new_radii[-1] = 1
            new_thetas[-1] = 1
            radial_deviations = new_radii
            angular_deviations = new_thetas

        self.radial_param_number, self.angular_param_number = len(radial_deviations), len(angular_deviations)

        # Each parameter is represented by a 2-vector: (position, value)
        # The extra 2 parameters are static parameters which correspond to fixing the start and end points
        self._radial_parameters = zeros((self.radial_param_number, 2))
        self._angular_parameters = zeros((self.angular_param_number, 2))
        self._radial_parameters[:, 0] = radial_deviations
        self._angular_parameters[:, 0] = angular_deviations

        # Set to either the provided value or None
        self.initial_point, self.final_point = None, None
        if 'initial_coordinate' in kwargs and 'final_coordinate' in kwargs:
            self.initial_point, self.final_point = kwargs['initial_coordinate'], kwargs['final_coordinate']

        # We provide 3 ways to initialize the parameters
        # Initialize with full parameters in one array
        # We must provide initial and final point if we choose to use the 'initial_parameters' kwarg
        if "initial_parameters" in kwargs:
            init_params = kwargs["initial_parameters"]
            self._radial_parameters[0, 1] = norm(self.initial_point)
            self._radial_parameters[1:-1, 1] = init_params[:self.radial_param_number - 2]
            self._angular_parameters[1:-1, 1] = init_params[self.radial_param_number - 2:]
            self._radial_parameters[-1, 1] = norm(self.final_point)

        # Right and left are defined by start and end points
        # We must provide initial and final point if we choose to use the 'initial_radial_parameters' kwarg
        elif 'initial_radial_parameters' in kwargs and 'initial_angular_parameters' in kwargs:
            self._radial_parameters[0, 1] = norm(self.initial_point)
            self._radial_parameters[1:-1, 1] = kwargs['initial_radial_parameters']
            self._angular_parameters[1:-1, 1] = kwargs['initial_angular_parameters']
            self._radial_parameters[-1, 1] = norm(self.final_point)

        # Initialize with the QuasiParabolic path
        # We don't need initial and final point provided if we use this path kwarg
        elif 'quasi_parabolic' in kwargs:
            qp_initial = kwargs['quasi_parabolic']
            self.initial_point = qp_initial(0)
            self.final_point = qp_initial(1)
            self.initial_point, self.final_point = qp_initial.initial_point, qp_initial.final_point
            total_angle = Vector.angle_between(self.initial_point, self.final_point)
            self._radial_parameters[:, 1] = qp_initial.poly_fit(radial_deviations*total_angle)
            self._radial_parameters[0, 1] = norm(qp_initial(0))
            self._radial_parameters[-1, 1] = norm(qp_initial(1))

        self.final_point = Vector.unit_vector(self.final_point)
        self.initial_point = Vector.unit_vector(self.initial_point)

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = Vector.unit_vector(self.normal_vec)

        self._total_angle = None

        self._poly_fit_angular = None
        self._poly_fit_radial = None
        self._poly_fit_cartesian = None

        return self

    @property
    def radial_points(self):
        return self._radial_parameters

    @property
    def adjustable_parameters(self):
        # We don't vary the first or last parameters (these are the fixed points on the path).
        return concatenate((self._radial_parameters[1:-1, 1], self._angular_parameters[1:-1, 1]))

    def transform_adjustments(self, indexes: ArrayLike, adjustments: ArrayLike) -> ArrayLike:
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
        point = array(list(map(lambda poly_fit: poly_fit(fraction, nu=nu), self._poly_fit_cartesian))).T

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

    def adjust_parameters(self, indexes, adjustments):
        # TODO: Add some parameters to control the adjustment rate to account for unresponsive parameters
        #  specifically radial params in the case of GCD path
        adjusted_params = self.adjustable_parameters
        indexes = asarray(indexes).flatten()
        copied_adjustments = asarray(adjustments).flatten()
        if len(indexes) != len(copied_adjustments):
            copied_adjustments = np.repeat(copied_adjustments, len(indexes))
        copied_adjustments[indexes >= self._radial_parameters.shape[0] - 2] = \
            copied_adjustments[indexes >= self._radial_parameters.shape[0] - 2]/EARTH_RADIUS
        adjusted_params[indexes] = adjusted_params[indexes] + copied_adjustments
        if self._poly_fit_cartesian is None:
            self.interpolate_params()

        new_path = GreatCircleDeviation.__old_init__(
            self._radial_parameters.shape[0] - 2,
            self._angular_parameters.shape[0] - 2,
            initial_parameters=adjusted_params,
            initial_coordinate=self(0),
            final_coordinate=self(1),
            using_spherical=False
        )

        new_path.interpolate_params()
        return new_path

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
            cartesian_points = zeros((len(self._radial_parameters), 3))
            for index in range(len(self._radial_parameters)):
                alpha = self._radial_parameters[index, 0]
                r_1 = Rotation.from_rotvec(self.normal_vec * alpha * self.total_angle)
                v_1 = r_1.apply(self.initial_point)
                rotation_vec_2 = Vector.unit_vector(cross(self.normal_vec, v_1))
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
