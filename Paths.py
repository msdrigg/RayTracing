from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import cross, outer, zeros, \
    linspace, concatenate, array, log, sqrt, square
import Vector
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.optimize import fsolve
from Atmosphere import ChapmanLayers
from Constants import pi, E_CHARGE, E_MASS, EARTH_RADIUS
from math import acos, cos, atan


class Path(ABC):
    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __call__(self, fraction, nu=0, use_spherical=False):
        pass


class QuasiParabolic(Path):
    # Currently, must initialize with a cartesian vector
    # This class needs less optimization because its just a starting point.
    def __init__(self, initial_coordinates, final_coordinates,
                 atmosphere_model, wave_frequency, point_number=None):
        # Initial and final points are normalized, but poly_fit(0) and poly_fit(1) will return the radial component
        # for the initial and final point
        initial_rad, final_rad = initial_coordinates[0], final_coordinates[0]
        self.initial_point, self.final_point = Vector.spherical_to_cartesian(initial_coordinates), \
            Vector.spherical_to_cartesian(final_coordinates)
        self.initial_point, self.final_point = self.initial_point/norm(self.initial_point), \
            self.final_point/norm(self.final_point)

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec/norm(self.normal_vec)

        # Parameters for quasiparabolic path only depend on atmospheric model and ignore magnetic field
        self._parameters = self.calculate_parameters(atmosphere_model, wave_frequency)

        angle_between = Vector.angle_between(initial_coordinates, final_coordinates, use_spherical=False)
        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(angle_between*Vector.EARTH_RADIUS)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value at each point (in km)
        self.points = zeros((point_number, 2))
        self.points[:, 0] = linspace(0, angle_between, point_number)
        self.points[:, 1] = linspace(initial_rad, final_rad, point_number)

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
        # Nu is not allowed as a parameter
        # This isn't optimized but its ok
        alpha = fraction*self.points[-1]
        rotations = Rotation.from_rotvec(outer(alpha, self.normal_vec))
        rotated_vecs = rotations.apply(self.initial_point)
        if use_spherical:
            output_vecs = Vector.cartesian_to_spherical(rotated_vecs)
            output_vecs[0] = self._poly_fit(alpha)
        else:
            output_vecs = rotated_vecs*self._poly_fit(alpha).reshape(-1, 1)
        if len(output_vecs) == 1:
            return output_vecs[0]
        else:
            return output_vecs

    def compile_points(self):
        e_max, rm, rb, ym, f = self._parameters
        fc = sqrt(80.62*e_max)
        a = 1 - (fc/f)**2 + (fc*rb/(f*ym))**2
        b = -2 * rm * (fc * rb / (f * ym)) ** 2
        total_angle = Vector.angle_between(self.initial_point, self.final_point)
        initial_radius = norm(self.initial_point)

        def beta_solver(beta_0_g):
            beta_0_g = beta_0_g[0]
            beta_b_g = acos(EARTH_RADIUS/rb*cos(beta_0_g))
            xb_g = rb**2 - (EARTH_RADIUS**2) * (cos(beta_0_g))**2
            c_g = (fc * rb * rm / (f * ym)) ** 2 - \
                  (EARTH_RADIUS ** 2) * (cos(beta_0_g)) ** 2
            return -total_angle/2*EARTH_RADIUS + EARTH_RADIUS*(beta_b_g - beta_0_g) + \
                (EARTH_RADIUS**2) * (cos(beta_0_g)) / sqrt(c_g) * \
                log((2*c_g + b*rb + 2*sqrt(c_g*xb_g) / (rb*sqrt(b**2 - 4*a*c_g))))

        # Guessing beta_0 assuming that the ray travels a straight line and meets between the
        # final and initial point at the point of greatest atmospheric e-density
        d_perp = norm(self.final_point - self.initial_point)
        alpha_0 = atan(rm/(d_perp/2))
        beta_0_initial_guess = alpha_0 - total_angle/2
        guess = array([beta_0_initial_guess])
        beta_0, info_dict, ier, msg = fsolve(beta_solver, guess)

        c = (fc*rb*rm/(f*ym))**2 - (EARTH_RADIUS**2)*(cos(beta_0))**2
        xb = rb**2 - (EARTH_RADIUS**2) * (cos(beta_0))**2
        beta_b = acos(EARTH_RADIUS/rb*cos(beta_0))
        apogee = -(b + sqrt(b**2 - 4*a*c))/(2*a)

        atmosphere_angle = (-beta_0 + beta_b)

        # We want 4 params for the straight part and 1 additional parameter per degree of longitude
        radius_params = zeros([int(total_angle*180/pi*2) + 4, 2])
        radius_params[::len(radius_params) - 1, 1] = initial_radius
        radius_params[slice(1, len(radius_params) - 1, len(radius_params) - 3)] = \
            [atmosphere_angle, self._parameters[3]]

        increasing = linspace(self._parameters[3], apogee, len(radius_params) - 4)
        radius_params[2:len(radius_params)/2, 1] = increasing
        radius_params[len(radius_params)/2: -2, 1] = increasing[::-1]  # Flip it

        def d_t(radius_vectorized):
            x = a*square(radius_vectorized) + b*radius_vectorized + c
            output = EARTH_RADIUS**2 * cos(beta_0)/sqrt(c) * \
                log(radius_vectorized*(2*c + self._parameters[3]*b +
                    2*sqrt(c*xb))/(self._parameters[3]*(2*c + radius_vectorized[:, 1]*b + 2*c*x)))
            output += EARTH_RADIUS*(beta_b - beta_0)
            return output

        radius_params[2: -2, 1] = d_t(radius_params[2: -2, 0])
        self._parameters = radius_params

    @property
    def poly_fit(self):
        if self._poly_fit is None:
            self.compile_points()
        return self._poly_fit

    @staticmethod
    def calculate_parameters(atmosphere_model, wave_frequency):
        if not isinstance(atmosphere_model, ChapmanLayers):
            raise NotImplementedError("Only chapman layers currently implemented.")
        e_max = atmosphere_model.parameters[0]**2*pi*E_MASS/E_CHARGE**2
        rm = atmosphere_model.parameters[1]
        ym = atmosphere_model.parameters[2]
        rb = rm - ym
        return array([e_max, rm, rb, ym, wave_frequency])


class GreatCircleDeviationPC(Path):
    def __init__(self, radial_deviations, angular_deviations, **kwargs):
        self.radial_param_number, self.angular_param_number = len(radial_deviations), len(angular_deviations)

        # Each parameter is represented by a 2-vector: (position, value)
        # The extra 2 parameters are static parameters which correspond to fixing the start and end points
        self._radial_positions = zeros((self.radial_param_number + 2, 2))
        self._angular_deviations = zeros((self.angular_param_number + 2, 2))

        # Set the position of all parameters by concatenating the lists of radial deviations and angular deviations
        # Position is the angular location along the great circle connecting initial and final points
        if isinstance(radial_deviations, int) or isinstance(angular_deviations, int):
            radial_deviations = linspace(0, 1, radial_deviations)
            angular_deviations = linspace(0, 1, angular_deviations)
        self._radial_positions[1:-1, 0] = radial_deviations
        self._angular_deviations[1:-1, 0] = angular_deviations

        # Set to either the provided value or None
        self.initial_point, self.final_point = kwargs.get('initial_coordinate'), kwargs.get('final_coordinate')

        # We provide 3 ways to initialize the parameters
        # Initialize with full parameters in one array
        if "initial_parameters" in kwargs:
            init_params = kwargs["initial_parameters"]
            self._radial_positions[1:-1, 1] = init_params[:self.radial_param_number]
            self._angular_deviations[1:-1, 1] = init_params[self.radial_param_number:]

        # Right and left are defined by start and end points
        elif 'initial_radial_parameters' in kwargs and 'initial_angular_parameters' in kwargs:
            self._radial_positions[1:-1, 1] = kwargs['initial_radial_parameters']
            self._angular_deviations[1:-1, 1] = kwargs['initial_angular_parameters']
            self.initial_point, self.final_point = Vector.spherical_to_cartesian(kwargs['initial_coordinate']), \
                Vector.spherical_to_cartesian(kwargs['final_coordinate'])

        # Initialize with the QuasiParabolic path
        elif 'quasi_parabolic' in kwargs:
            qp_initial = kwargs['quasi_parabolic']
            self.initial_point = qp_initial(0)
            self.final_point = qp_initial(1)
            self._radial_positions[1:-1, 1] = qp_initial(radial_deviations)
            self._radial_positions[0] = qp_initial(0)
            self._radial_positions[-1] = qp_initial(1)

        # Normalize initial and final points
        self.initial_point, self.final_point = self.initial_point/norm(self.initial_point), \
            self.final_point/norm(self.final_point)

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec/norm(self.normal_vec)

        self._total_angle = None

        self._poly_fit_angular = None
        self._poly_fit_radial = None
        self._poly_fit_cartesian = None

    @property
    def parameters(self):
        return concatenate((self._radial_positions[1:-1], self._angular_deviations[1:-1]))

    @property
    def radial_points(self):
        return self._radial_positions

    @parameters.setter
    def parameters(self, parameter_tuple):
        if parameter_tuple[0] is not None:
            self._radial_positions = parameter_tuple[0]
        if parameter_tuple[1] is not None:
            self._angular_deviations = parameter_tuple[1]

        # Redefine the poly_fits that control this method
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

    def __call__(self, fraction, nu=0, use_spherical=False):
        if self._poly_fit_cartesian is None:
            self.interpolate_params()

        point = array(list(map(lambda poly_fit: poly_fit(fraction, nu=nu), self._poly_fit_cartesian)))

        if use_spherical:
            output_vecs = Vector.cartesian_to_spherical(point)
        else:
            output_vecs = point
        if len(output_vecs) == 1:
            return output_vecs[1]
        else:
            return output_vecs

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

    @property
    def total_angle(self):
        if self._total_angle is None:
            self._total_angle = Vector.angle_between(self.initial_point, self.final_point)
        return self._total_angle

    def interpolate_params(self, radial=False):
        self._poly_fit_angular = CubicSpline(self._angular_deviations[:, 0],
                                             self._angular_deviations[:, 1],
                                             bc_type='natural', extrapolate=False)
        if radial:
            self._poly_fit_radial = CubicSpline(self._radial_positions[:, 0],
                                                self._radial_positions[:, 1],
                                                bc_type='natural', extrapolate=False)
        else:
            cartesian_points = zeros((len(self.parameters) + 2, 3))
            for index in range(len(self._radial_positions)):
                alpha = self._radial_positions[index, 0]
                r_1 = Rotation.from_rotvec(self.normal_vec * alpha)
                v_1 = r_1.apply(self.initial_point)
                rotation_vec_2 = cross(self.normal_vec, v_1)
                rotation_vec_2 *= (self._poly_fit_angular(alpha)/norm(rotation_vec_2))
                r_2 = Rotation.from_rotvec(rotation_vec_2)
                v_2 = r_2.apply(v_1)
                v_2 *= self._radial_positions[index, 1]
                cartesian_points[index] = v_2
            for index in range(3):
                self._poly_fit_cartesian[index] = CubicSpline(self._radial_positions[:, 0], cartesian_points[index])
