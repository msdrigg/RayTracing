from abc import ABC, abstractmethod
import numpy as np
import GIS
from scipy.interpolate import CubicSpline


class Path(ABC):
    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class QuasiParabolic(Path):
    def __init__(self, initial_coordinates, final_coordinates, atmosphere_model, point_number=None):
        # Parameters for quasiparabolic path only depend on atmospheric model and ignore magnetic field
        self.parameters = self.calculate_parameters(atmosphere_model)

        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(GIS.angle_between(initial_coordinates, final_coordinates)*20)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value
        self.points = np.zeros((point_number, 2))
        self.points[:, 0] = np.linspace(0, GIS.angle_between(initial_coordinates, final_coordinates),
                                        point_number)

        # Real representation of the line will be a poly fit of radius vs angle along great circle
        self._poly_fit = None
        self.compile_points()

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters):
        self._parameters = new_parameters
        self.compile_points()

    def __call__(self, *args, **kwargs):
        return self._poly_fit(*args, **kwargs)

    def compile_points(self):
        # This compiles the points
        self._poly_fit = Exception()

    @property
    def poly_fit(self):
        if self._poly_fit is None:
            self.compile_points()
        return self._poly_fit

    @staticmethod
    def calculate_parameters(atmosphere_model):
        return np.zeros((len(atmosphere_model),))


class GreatCircleDeviationPC(Path):
    def __init__(self, radial_deviations, angular_deviations, **kwargs):
        self.initial_point, self.final_point = kwargs.get('initial_coordinates'), kwargs.get('final_coordinates')
        self.radial_param_number, self.angular_param_number = len(radial_deviations), len(angular_deviations)

        # Each parameter is represented by a 2-vector: (position, value)
        # The extra 2 parameters are static parameters which correspond to fixing the start and end points
        self._radial_parameters = np.zeros((self.radial_param_number + 2, 2))
        self._angular_parameters = np.zeros((self.angular_param_number + 2, 2))

        # Set the position of all parameters by concatenating the lists of radial deviations and angular deviations
        # Position is the angular location along the great circle connecting initial and final points
        self._radial_parameters[1:-1, 0] = radial_deviations
        self._angular_parameters[1:-1, 0] = angular_deviations

        # Value is the value of the parameter (in radians for angular deviations and km for radial)
        # Positive value means upward for radial and rightward for angular
        # Right and left are defined by start and end points
        initial_radial_deviations = kwargs.get('initial_radial_deviations')
        initial_angular_deviations = kwargs.get('initial_angular_deviations')
        if initial_radial_deviations is not None or initial_angular_deviations is not None:
            if initial_radial_deviations is not None:
                # Set the radial deviations to the initial values, otherwise they remain all 0
                self._radial_parameters[1:-1, 1] = initial_radial_deviations

            if initial_angular_deviations is not None:
                # Set the angular deviations to the initial values, otherwise they remain all 0
                self._angular_parameters[1:-1, 1] = initial_angular_deviations

        elif "quasi_parabolic" in kwargs:
            quasi_parabolic_start = kwargs['quasi_parabolic']
            self.initial_point = quasi_parabolic_start(0)
            self.final_point = quasi_parabolic_start(1)

        elif "initial_parameters" in kwargs:
            init_params = kwargs["initial_parameters"]
            self._radial_parameters[1:-1, 1] = init_params[:self.radial_param_number]
            self._angular_parameters[1:-1, 1] = init_params[self.radial_param_number:]

        self._poly_fit_angular = None
        self._poly_fit_radial = None

    @property
    def parameters(self):
        return np.concatenate((self._radial_parameters, self._angular_parameters))

    @parameters.setter
    def parameters(self, parameter_tuple):
        if parameter_tuple[0] is not None:
            self._radial_parameters = parameter_tuple[0]
        if parameter_tuple[1] is not None:
            self._angular_parameters = parameter_tuple[1]

        # Redefine angular and radial poly_fits
        self.interpolate_params()

    def adjust_parameter(self, index, adjustment):
        if index < self.radial_param_number:
            self._radial_parameters[index, 1] = self._radial_parameters[index, 1] + adjustment
        else:
            self._angular_parameters[index - self.radial_param_number, 1] = \
                self._angular_parameters[index - self.radial_param_number, 1] + adjustment

    def __call__(self, *args, **kwargs):
        return self._poly_fit_radial(*args, **kwargs), self._poly_fit_angular(*args, **kwargs)

    @property
    def poly_fit_angular(self):
        if self._poly_fit_angular is None:
            self.interpolate_params()
        return self._poly_fit_angular

    @property
    def poly_fit_radial(self):
        if self._poly_fit_radial is None:
            self.interpolate_params()
        return self._poly_fit_radial

    def interpolate_params(self):
        self._poly_fit_angular = CubicSpline(self._angular_parameters[:, 0],
                                             self._angular_parameters[:, 1],
                                             bc_type='natural', extrapolate=False)
        self._poly_fit_radial = CubicSpline(self._radial_parameters[:, 0],
                                            self._radial_parameters[:, 1],
                                            bc_type='natural', extrapolate=False)
