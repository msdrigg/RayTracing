from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import cross, outer, zeros, repeat, cos, arccos, arctan2, \
    linspace, concatenate, array, log, sqrt, square, asarray, amax
import numpy as np
from numpy import where as where_array
import Vector
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation
from scipy.optimize import fsolve
from Constants import PI, EARTH_RADIUS, TYPE_ABBREVIATION
from matplotlib import pyplot as plt
from os.path import join as join_path


class Path(ABC):
    def visualize(self, fig=None, ax=None, show=True, frequency=None, color=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        if frequency is None:
            _frequency = "?"
        else:
            _frequency = int(frequency / 1E6)
        ax.set_title(f"3D Ray Trace")
        ax.autoscale(False)
        if color is None:
            _color = 'black'
        else:
            _color = color
        class_type = self.__class__.__name__
        _label = f'{TYPE_ABBREVIATION[class_type]} - {_frequency} MHz'
        angular_distance = linspace(0, 1, 100)
        radii = self(angular_distance, use_spherical=True)[:, 0]
        radii = (radii - EARTH_RADIUS) / 1000
        km_range = angular_distance * self.total_angle * EARTH_RADIUS / 1000
        max_x = km_range[-1]
        max_y = amax(radii)
        plt.xlim(-max_x * .05, max_x * 1.05)
        plt.ylim(-max_y * .05, max_y * 1.05)
        ax.plot(km_range, radii, color=_color, label=_label)
        ax.legend()
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")

        if show:
            plt.show()
        else:
            return fig, ax

    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __call__(self, fraction, nu=0, use_spherical=False):
        pass

    @property
    def total_angle(self):
        return None


class QuasiParabolic(Path):
    # Currently, must initialize with a cartesian vector
    # This class needs less optimization because its just a starting point.
    def __init__(self, initial_coordinates, final_coordinates, atmosphere_model,
                 wave_frequency, degree=4, point_number=None):
        # Initial and final points are normalized, but poly_fit(0) and poly_fit(1) will return the radial component
        # for the initial and final point
        initial_rad, final_rad = norm(initial_coordinates), norm(final_coordinates)
        self.initial_point, self.final_point = initial_coordinates, final_coordinates
        self.initial_point, self.final_point = self.initial_point / norm(self.initial_point), \
                                               self.final_point / norm(self.final_point)

        self.degree = degree

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = self.normal_vec / norm(self.normal_vec)

        angle_between = Vector.angle_between(initial_coordinates, final_coordinates, use_spherical=False)

        # Parameters for quasiparabolic path only depend on atmospheric model and ignore magnetic field
        if atmosphere_model.gradient is None:
            gradient_effect = 0
        else:
            midpoint = (self.initial_point + self.final_point) / 2
            midpoint = Vector.unit_vector(midpoint) * (EARTH_RADIUS + atmosphere_model.parameters[1])
            f_1 = atmosphere_model.plasma_frequency(midpoint)
            f_0 = atmosphere_model.parameters[0]
            gradient_effect = f_1 - f_0
        self._parameters = self.calculate_parameters(atmosphere_model, gradient_effect, wave_frequency)
        # Point number is the number of points to calculate. All points in between are interpolated from PC spline
        if point_number is not None:
            self.point_number = point_number
        else:
            self.point_number = int(angle_between * EARTH_RADIUS)

        # Each point is evenly spaced along the great circle path connecting initial and final coordinates
        # Each point is a 2-vector holding its angular value in radians (along great circle path)
        # and its radial value at each point (in km)
        self.points = zeros((self.point_number, 2))
        self.points[:, 0] = linspace(0, angle_between, self.point_number)
        self.points[:, 1] = linspace(initial_rad, final_rad, self.point_number)

        # Real representation of the line will be a poly fit of radius vs angle along great circle
        self._poly_fit = None
        self._total_angle = None
        self.compile_points()

    @property
    def total_angle(self):
        if self._total_angle is None:
            self._total_angle = Vector.angle_between(self.initial_point, self.final_point)
        return self._total_angle

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
        if "nu" in kwargs and kwargs.get("nu") != 0:
            raise (NotImplementedError("Quasiparabolic class does not have derivative capabilities."))
        if self._poly_fit is None:
            self.compile_points()
        alpha = fraction * self.points[-1, 0]

        rotations = Rotation.from_rotvec(outer(alpha, self.normal_vec))
        rotated_vecs = rotations.apply(self.initial_point)
        if use_spherical:
            output_vecs = Vector.cartesian_to_spherical(rotated_vecs)
            output_vecs[:, 0] = self._poly_fit(alpha)
        else:
            output_vecs = rotated_vecs * self._poly_fit(alpha).reshape(-1, 1)
        if len(output_vecs) == 1:
            return output_vecs[0]
        else:
            return output_vecs

    def compile_points(self):
        fc, rm, rb, ym, f = self._parameters
        a = 1 - (fc / f) ** 2 + (fc * rb / (f * ym)) ** 2
        b = -2 * rm * (fc * rb / (f * ym)) ** 2
        total_angle = Vector.angle_between(self.initial_point, self.final_point)

        def beta_solver(beta_0_g):
            # According to equation 8 in the hill 1979 paper
            beta_b_g = arccos(EARTH_RADIUS * cos(beta_0_g) / rb)
            xb_g = rb ** 2 - (EARTH_RADIUS ** 2) * (cos(beta_0_g)) ** 2
            c_g = (fc * rb * rm / (f * ym)) ** 2 - \
                  (EARTH_RADIUS ** 2) * (cos(beta_0_g)) ** 2
            adder = EARTH_RADIUS * (beta_b_g - (total_angle / 2 + beta_0_g))
            multiplier = (EARTH_RADIUS ** 2) * (cos(beta_0_g)) / sqrt(c_g)
            numerator = 2 * c_g + b * rb + 2 * sqrt(c_g * xb_g)
            denominator = rb * sqrt(b ** 2 - 4 * a * c_g)
            print(f"Numerator {numerator}, denominatory {denominator}, beta_b_g {beta_b_g}, beta_0_g {beta_0_g}")
            output = adder + multiplier * log(numerator / denominator)
            return output

        # Guessing beta_0 assuming that the ray travels a straight line and meets between the
        # final and initial point at the point of greatest atmospheric e-density
        d_perp = norm(self.final_point * self.points[0, 1] - self.initial_point * self.points[-1, 1])
        alpha_0 = arctan2(rm - EARTH_RADIUS, d_perp / 2)
        beta_0_initial_guess = (alpha_0 - total_angle / 2)
        # beta_0_initial_guess = 0.3
        while b ** 2 < 4 * a * ((fc * rb * rm / (f * ym)) ** 2 -
                                (EARTH_RADIUS ** 2) * (cos(beta_0_initial_guess)) ** 2):
            print(f"Reducing initial beta guess. New guess: {beta_0_initial_guess}")
            if beta_0_initial_guess < .02:
                raise OverflowError(f"Too many runs {beta_0_initial_guess}")
            beta_0_initial_guess *= 2.0 / 3.0

        beta_0, info_dict, ier, msg = fsolve(beta_solver, x0=beta_0_initial_guess, full_output=True)
        print(f"Beta_0: {beta_0}")
        if ier != 1:
            print(f"Function Call Number: {info_dict['nfev']}")
            print(f"Last solution before failure: {beta_0}")
            raise RuntimeError(f"Error on beta_0 fsolve: {msg}")

        c = (fc * rb * rm / (f * ym)) ** 2 - (EARTH_RADIUS * cos(beta_0)) ** 2
        xb = rb ** 2 - (EARTH_RADIUS * cos(beta_0)) ** 2
        beta_b = arccos(EARTH_RADIUS * cos(beta_0) / rb)
        apogee = (-(b + sqrt(b ** 2 - 4 * a * c)) / (2 * a))[0]

        # We want 2 params for the straight part and 2 additional parameter per degree of longitude
        radius_params = zeros([int(total_angle * 180 / PI) * 8 + 1, 2])
        radius_params[::len(radius_params) - 1, 1] = self.points[0, 1], self.points[-1, 1]
        radius_params[-1, 0] = total_angle

        increasing = linspace(EARTH_RADIUS, apogee, int((len(radius_params)) / 2) + 1)
        radius_params[:int(len(radius_params) / 2) + 1, 1] = increasing
        radius_params[int(len(radius_params) / 2):, 1] = increasing[::-1]  # Flip it
        radius_params[int(len(radius_params) / 2), 1] = apogee

        def d_t(radius_vectorized):
            print(radius_vectorized)
            plt.plot(radius_vectorized)
            plt.show()
            below_output = EARTH_RADIUS * (arccos(EARTH_RADIUS * cos(beta_0) / radius_vectorized) - beta_0)
            x = a * square(radius_vectorized) + b * radius_vectorized + c
            multiplier = EARTH_RADIUS ** 2 * cos(beta_0) / sqrt(c)
            numerator = radius_vectorized * (2 * c + rb * b + 2 * sqrt(c * xb))
            print("C, x")
            print(c)
            print(x)
            denominator = rb * (2 * c + radius_vectorized * b + 2 * sqrt(c * x))
            adder = EARTH_RADIUS * (beta_b - beta_0)
            above_output = adder + multiplier * log(numerator / denominator)
            output = where_array(radius_vectorized <= rb, below_output, above_output)
            return output

        radius_params[:int(len(radius_params) / 2), 0] = \
            d_t(radius_params[:int(len(radius_params) / 2), 1]) / EARTH_RADIUS
        radius_params[int(len(radius_params) / 2) + 1:, 0] = \
            d_t(radius_params[int(len(radius_params) / 2) + 1:, 1]) / EARTH_RADIUS
        radius_params[int(len(radius_params) / 2), 0] = \
            total_angle/2
        print(f"Error: {radius_params[int(len(radius_params) / 2)]}")
        print(f"Part 2: {int(len(radius_params) / 2) + 1}")
        print(f"Part 1: {int(len(radius_params) / 2) + 1}")
        print(f"Size: {radius_params[:, 0].shape}")
        print(f"Pos: {radius_params[:, 0]}")
        plt.plot(radius_params[:, 0], radius_params[:, 1])
        plt.show()

        self.points = radius_params
        self._total_angle = total_angle
        # plt.plot(radius_params[:, 0], radius_params[:, 1], color='blue')
        # plt.plot(radius_params[:, 0], repeat(apogee, len(radius_params[:, 0])))
        # plt.savefig(join_path("SavedPlots", 'QP Plot.png'))
        plt.plot(radius_params[:, 0], color='red')
        plt.show()
        self._poly_fit = UnivariateSpline(radius_params[:, 0], radius_params[:, 1],
                                          k=self.degree, s=0, ext=0)

    @property
    def poly_fit(self):
        if self._poly_fit is None:
            self.compile_points()
        return self._poly_fit

    @staticmethod
    def calculate_parameters(atmosphere_model, gradient_effect, wave_frequency):
        f_max = (atmosphere_model.parameters[0] + gradient_effect)
        rm = atmosphere_model.parameters[1] + EARTH_RADIUS
        ym = atmosphere_model.parameters[2]
        rb = rm - ym
        if atmosphere_model.plasma_frequency(array([rb, 0, 0]), using_spherical=True) < 0:
            raise (RuntimeError("RB allows for negative atmosphere."))
        return array([f_max, rm, rb, ym, wave_frequency])


class GreatCircleDeviation(Path):
    def __init__(self, radial_deviations, angular_deviations, using_spherical=True, **kwargs):
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
        self._radial_positions = zeros((self.radial_param_number, 2))
        self._angular_deviations = zeros((self.angular_param_number, 2))
        self._radial_positions[:, 0] = radial_deviations
        self._angular_deviations[:, 0] = angular_deviations

        # Set to either the provided value or None
        self.initial_point, self.final_point = None, None
        if 'initial_coordinate' in kwargs and 'final_coordinate' in kwargs:
            self.initial_point, self.final_point = kwargs['initial_coordinate'], kwargs['final_coordinate']

        if using_spherical and self.initial_point is not None and self.final_point is not None:
            self.initial_point, self.final_point = Vector.spherical_to_cartesian(self.initial_point), \
                                                   Vector.spherical_to_cartesian(self.final_point)

        # We provide 3 ways to initialize the parameters
        # Initialize with full parameters in one array
        # We must provide initial and final point if we choose to use the 'initial_parameters' kwarg
        if "initial_parameters" in kwargs:
            init_params = kwargs["initial_parameters"]
            self._radial_positions[0, 1] = norm(self.initial_point)
            self._radial_positions[1:-1, 1] = init_params[:self.radial_param_number - 2]
            self._angular_deviations[1:-1, 1] = init_params[self.radial_param_number - 2:]
            self._radial_positions[-1, 1] = norm(self.final_point)

        # Right and left are defined by start and end points
        # We must provide initial and final point if we choose to use the 'initial_radial_parameters' kwarg
        elif 'initial_radial_parameters' in kwargs and 'initial_angular_parameters' in kwargs:
            self._radial_positions[0, 1] = norm(self.initial_point)
            self._radial_positions[1:-1, 1] = kwargs['initial_radial_parameters']
            self._angular_deviations[1:-1, 1] = kwargs['initial_angular_parameters']
            self._radial_positions[-1, 1] = norm(self.final_point)

        # Initialize with the QuasiParabolic path
        # We don't need initial and final point provided if we use this path kwarg
        elif 'quasi_parabolic' in kwargs:
            qp_initial = kwargs['quasi_parabolic']
            self.initial_point = qp_initial(0)
            self.final_point = qp_initial(1)
            self.initial_point, self.final_point = qp_initial.initial_point, qp_initial.final_point
            total_angle = Vector.angle_between(self.initial_point, self.final_point)
            self._radial_positions[:, 1] = qp_initial.poly_fit(radial_deviations * total_angle)
            self._radial_positions[0, 1] = norm(qp_initial(0))
            self._radial_positions[-1, 1] = norm(qp_initial(1))

        self.final_point = Vector.unit_vector(self.final_point)
        self.initial_point = Vector.unit_vector(self.initial_point)

        self.normal_vec = cross(self.initial_point, self.final_point)
        self.normal_vec = Vector.unit_vector(self.normal_vec)

        self._total_angle = None

        self._poly_fit_angular = None
        self._poly_fit_radial = None
        self._poly_fit_cartesian = None

    @property
    def parameters(self):
        # TODO: Make sure the parameters here match up with everything else without breaking them.
        # todo: I think returning same format as they are entered in makes perfect sense
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

    def adjust_parameters(self, indexes, adjustments, mutate=False):
        # TODO: Add some parameters to control the adjustment rate to account for unresponsive parameters
        #  specifically radial params in the case of GCD path
        adjusted_params = self.parameters
        indexes = asarray(indexes).flatten()
        copied_adjustments = asarray(adjustments).flatten()
        if len(indexes) != len(copied_adjustments):
            copied_adjustments = repeat(copied_adjustments, len(indexes))
        copied_adjustments[indexes >= self.radial_param_number - 2] = \
            copied_adjustments[indexes >= self.radial_param_number - 2] / EARTH_RADIUS
        adjusted_params[indexes, 1] = adjusted_params[indexes, 1] + copied_adjustments
        if self._poly_fit_cartesian is None:
            self.interpolate_params()
        if not mutate:
            new_path = GreatCircleDeviation(
                len(self._radial_positions) - 2,
                len(self._angular_deviations) - 2,
                initial_parameters=adjusted_params[:, 1],
                initial_coordinate=self(0),
                final_coordinate=self(1),
                using_spherical=False
            )

            new_path.interpolate_params()
            return new_path
        else:
            self._radial_positions[1:-1] = adjusted_params[:self.radial_param_number]
            self._angular_deviations[1:-1] = adjusted_params[self.radial_param_number:]

            self.interpolate_params()

    def __call__(self, fraction, nu=0, use_spherical=False):
        _ = self.poly_fit_cartesian
        point = array(list(map(lambda poly_fit: poly_fit(fraction, nu=nu), self._poly_fit_cartesian))).T

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

    def interpolate_params(self, radial=False, degree=3):
        self._poly_fit_angular = UnivariateSpline(self._angular_deviations[:, 0],
                                                  self._angular_deviations[:, 1],
                                                  k=min(degree, len(self._angular_deviations) - 1), s=0, ext=0)
        if radial:
            self._poly_fit_radial = UnivariateSpline(self._radial_positions[:, 0],
                                                     self._radial_positions[:, 1],
                                                     k=degree, s=0, ext=0)
        else:
            cartesian_points = zeros((len(self._radial_positions), 3))
            for index in range(len(self._radial_positions)):
                alpha = self._radial_positions[index, 0]
                r_1 = Rotation.from_rotvec(self.normal_vec * alpha * self.total_angle)
                v_1 = r_1.apply(self.initial_point)
                rotation_vec_2 = Vector.unit_vector(cross(self.normal_vec, v_1))
                rotation_vec_2 *= self._poly_fit_angular(alpha)
                r_2 = Rotation.from_rotvec(rotation_vec_2)
                v_2 = r_2.apply(v_1)
                v_2 *= self._radial_positions[index, 1]
                cartesian_points[index] = v_2
            self._poly_fit_cartesian = []
            for index in range(3):
                self._poly_fit_cartesian.append(UnivariateSpline(self._radial_positions[:, 0],
                                                                 cartesian_points[:, index], k=degree, s=0))
