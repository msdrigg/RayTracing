from abc import ABC, abstractmethod
import numpy as np
import GIS
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


class Path(ABC):
    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class QuasiParabolic(Path):
    # This class needs less optimization because its just a starting point.

    def __init__(self, initial_coordinates, final_coordinates, atmosphere_model, point_number=None):
        # Initial and final points are normalized, but poly_fit(0) and poly_fit(1) will return the radial component
        # for the initial and final point
        initial_rad, final_rad = initial_coordinates[0], final_coordinates[0]
        self.initial_point, self.final_point = GIS.cartesian(initial_coordinates), GIS.cartesian(final_coordinates)
        self.initial_point, self.final_point = self.initial_point/np.linalg.norm(self.initial_point), \
            self.final_point/np.linalg.norm(self.final_point)

        self.normal_vec = np.cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec/np.linalg.norm(self.normal_vec)

        # Parameters for quasiparabolic path only depend on atmospheric model and ignore magnetic field
        self._parameters = self.calculate_parameters(atmosphere_model)

        angle_between = GIS.angle_between(initial_coordinates, final_coordinates, use_spherical=False)
        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(angle_between*20)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value at each point (in km)
        self.points = np.zeros((point_number, 2))
        self.points[:, 0] = np.linspace(0, angle_between, point_number)
        self.points[:, 1] = np.linspace(initial_rad, final_rad, point_number)

        # Real representation of the line will be a poly fit of radius vs angle along great circle
        self._poly_fit = None
        self.compile_points()

    @property
    def parameters(self):
        # Parameters are the parameters defining the parabolic atmosphere 
        # Examples are upper and lower radius, midpoint, maximum, etc.
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters):
        self._parameters = new_parameters
        self.compile_points()

    def __call__(self, fraction, use_spherical=False, **kwargs):
        # This isn't optimized but its ok
        alpha = fraction*self.points[-1]
        r_1 = Rotation.from_rotvec(self.normal_vec*alpha)
        v_1 = r_1.apply(GIS.cartesian(self.initial_point))
        if use_spherical:
            v_1 = GIS.spherical(v_1)
            v_1[0] = self._poly_fit(alpha)
        else:
            v_1 = v_1*self._poly_fit(alpha)
        return v_1

    def compile_points(self):
        # TODO: Finish implementing this feature, just in spherical
        # Poly_fit needs to fit
        self._poly_fit = Exception()

    @property
    def poly_fit(self):
        if self._poly_fit is None:
            self.compile_points()
        return self._poly_fit

    @staticmethod
    def calculate_parameters(atmosphere_model):
        # TODO: Finish implementing this feature
        return np.zeros((len(atmosphere_model),))


class GreatCircleDeviationPC(Path):
    def __init__(self, radial_deviations, angular_deviations, **kwargs):
        qp_init = kwargs.get('quasi_parabolic')
        if qp_init is not None:
            self.initial_point = qp_init(0)
            self.final_point = qp_init(1)
        else:
            self.initial_point, self.final_point = kwargs['initial_coordinates'], kwargs['final_coordinates']
        self.normal_vec = np.cross(GIS.cartesian(self.initial_point), GIS.cartesian(self.final_point))
        self.normal_vec = self.normal_vec/np.linalg.norm(self.normal_vec)

        self.radial_param_number, self.angular_param_number = len(radial_deviations), len(angular_deviations)

        # Each parameter is represented by a 2-vector: (position, value)
        # The extra 2 parameters are static parameters which correspond to fixing the start and end points
        self._radial_positions = np.zeros((self.radial_param_number + 2, 2))
        self._angular_deviations = np.zeros((self.angular_param_number + 2, 2))

        # Set the position of all parameters by concatenating the lists of radial deviations and angular deviations
        # Position is the angular location along the great circle connecting initial and final points
        self._radial_positions[1:-1, 0] = radial_deviations
        self._angular_deviations[1:-1, 0] = angular_deviations

        # Value is the value of the parameter (in radians for angular deviations and km for radial)
        # Positive value means upward for radial and rightward for angular
        # Right and left are defined by start and end points
        initial_radial_deviations = kwargs.get('initial_radial_deviations')
        initial_angular_deviations = kwargs.get('initial_angular_deviations')
        if initial_radial_deviations is not None or initial_angular_deviations is not None:
            if initial_radial_deviations is not None:
                # Set the radial deviations to the initial values, otherwise they remain all 0
                self._radial_positions[1:-1, 1] = initial_radial_deviations

            if initial_angular_deviations is not None:
                # Set the angular deviations to the initial values, otherwise they remain all 0
                self._angular_deviations[1:-1, 1] = initial_angular_deviations

        elif qp_init is not None:
            self._radial_positions[:, 1] = qp_init(self._radial_positions[:, 0])

        elif "initial_parameters" in kwargs:
            init_params = kwargs["initial_parameters"]
            self._radial_positions[1:-1, 1] = init_params[:self.radial_param_number]
            self._angular_deviations[1:-1, 1] = init_params[self.radial_param_number:]

        # Initial and final points are normalized to help with the math
        self.initial_point[0], self.final_point[0] = 1, 1

        self._poly_fit_angular = None
        self._poly_fit_radial = None

    @property
    def parameters(self):
        return np.concatenate((self._radial_positions[1:-1], self._angular_deviations[1:-1]))

    @parameters.setter
    def parameters(self, parameter_tuple):
        if parameter_tuple[0] is not None:
            self._radial_positions = parameter_tuple[0]
        if parameter_tuple[1] is not None:
            self._angular_deviations = parameter_tuple[1]

        # Redefine angular and radial poly_fits
        self.interpolate_params()

    def adjust_parameter(self, indexes, adjustments, mutate=False):
        adjusted_params = self.parameters
        adjusted_params[indexes, 1] = adjusted_params[indexes, 1] + adjustments
        if not mutate:
            new_path = GreatCircleDeviationPC(self._radial_positions[1:-1, 0].copy(),
                                              self._angular_deviations[1:-1, 0].copy(),
                                              initial_parameters=adjusted_params)
            new_path.interpolate_params()
            return new_path
        else:
            self._radial_positions[1:-1] = adjusted_params[:self.radial_param_number]
            self._angular_deviations[1:-1] = adjusted_params[self.radial_param_number:]

            self.interpolate_params()

    def __call__(self, *args, **kwargs):
        alpha = args[0]*self._radial_positions[-1, 0]
        r_1 = Rotation.from_rotvec(self.normal_vec*alpha)
        v_1 = r_1.apply(GIS.cartesian(self.initial_point))
        rotation_vec_2 = np.cross(self.normal_vec, v_1)
        rotation_vec_2 = rotation_vec_2/np.linalg.norm(rotation_vec_2)*self._poly_fit_angular(alpha)
        r_2 = Rotation.from_rotvec(rotation_vec_2)
        v_2 = r_2.apply(v_1)
        v_2_norm = np.linalg.norm(v_2)
        v_2 = v_2/v_2_norm*(v_2_norm + self._poly_fit_radial(alpha))
        # TODO: Optimize this.
        # I think we can maybe save poly fits to the cartesian coordinates, and that'll make it easier
        return GIS.spherical(v_2)

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
        self._poly_fit_angular = CubicSpline(self._angular_deviations[:, 0],
                                             self._angular_deviations[:, 1],
                                             bc_type='natural', extrapolate=False)
        self._poly_fit_radial = CubicSpline(self._radial_positions[:, 0],
                                            self._radial_positions[:, 1],
                                            bc_type='natural', extrapolate=False)
