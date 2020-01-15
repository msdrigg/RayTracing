from numpy import dot, zeros, array, sqrt, linspace, square
from numpy import sum as vector_sum
from numpy.linalg import norm
from Atmosphere import ChapmanLayers
from Field import BasicField
from Paths import QuasiParabolic, GreatCircleDeviationPC
from scipy.linalg import solve as sym_solve
from scipy.integrate import simps
from scipy.optimize import fsolve
import Vector
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from Constants import PI, EARTH_RADIUS


DEFAULT_PARAMS = 100, 10


class Tracer:
    def __init__(self, wave_frequency, atmosphere_model, magnetic_field, initial_path_generator):
        if None in (atmosphere_model, magnetic_field, initial_path_generator):
            raise ValueError("Model initializer parameters cannot be Null")
        self.field, self.atmosphere, self.path_generator = magnetic_field, atmosphere_model, initial_path_generator
        self.frequency = wave_frequency

        self.initial_coordinates, self.final_coordinates = None, None
        self.initial_path = None

        # If you manually set the parameters, you must also set the parameter number
        self.parameters = None
        self.parameter_number = None
        self.calculated_paths = None

    def compile_initial_path(self):
        if self.initial_coordinates is None or self.final_coordinates is None:
            raise ValueError("Initial and final coordinates must be defined before compiling")

        if self.parameters is None:
            self.parameters = DEFAULT_PARAMS
            self.parameter_number = DEFAULT_PARAMS[0] + DEFAULT_PARAMS[1]

        if self.calculated_paths is not None:
            raise ValueError("Calculated path is not Null. For safety reasons, you cannot recompile without first"
                             "setting the calculated path to None")

        self.initial_path = self.path_generator(self.initial_coordinates, self.final_coordinates,
                                                self.atmosphere, self.frequency)
        new_path = GreatCircleDeviationPC(*self.parameters, quasi_parabolic=self.initial_path)

        if self.calculated_paths is None:
            self.calculated_paths = [new_path]
        else:
            self.calculated_paths.append(new_path)

    def trace(self, steps=5, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = DEFAULT_PARAMS
            self.parameter_number = DEFAULT_PARAMS[0] + DEFAULT_PARAMS[1]

        if self.calculated_paths is None:
            self.compile_initial_path()

        for _ in range(steps):
            self.newton_raphson_step()

        return self.calculated_paths

    def newton_raphson_step(self, h=0.001):
        matrix, gradient = self.calculate_derivatives(h=h)
        # Calculate diagonal matrix elements
        b = matrix@self.calculated_paths[-1].parameters - gradient
        next_params = sym_solve(matrix, b, assume_a='sym')
        next_path = GreatCircleDeviationPC(*self.parameters, initial_parameters=next_params,
                                           initial_coordinate=self.initial_coordinates,
                                           final_coordinate=self.final_coordinates, using_spherical=False)
        self.calculated_paths.append(next_path)

    def calculate_derivatives(self, h=0.001):
        gradient = zeros((self.parameter_number,))
        matrix = zeros((self.parameter_number, self.parameter_number))
        curr_path = self.calculated_paths[-1]
        # Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
        for param in range(self.parameter_number):
            path_minus = curr_path.adjust_parameters(param, -h)
            path_plus = curr_path.adjust_parameters(param, -h)
            p_minus = self.integrate_parameter(path_minus)
            p_plus = self.integrate_parameter(path_plus)
            p_0 = self.integrate_parameter(curr_path)
            gradient[param] = (p_plus - p_minus)/(2*h)
            matrix[param, param] = (p_plus + p_minus - 2*p_0)/(h**2)

        # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
        for row in range(0, self.parameter_number - 1):
            for col in range(1 + row, self.parameter_number):
                path_mm = curr_path.adjust_parameters([row, col], -h)
                p_mm = self.integrate_parameter(path_mm)
                path_mp = curr_path.adjust_parameters([row, col], [-h, h])
                p_mp = self.integrate_parameter(path_mp)
                path_pm = curr_path.adjust_parameters([row, col], [h, -h])
                p_pm = self.integrate_parameter(path_pm)
                path_pp = curr_path.adjust_parameters([row, col], h)
                p_pp = self.integrate_parameter(path_pp)
                d2pdxy = (p_pp - p_pm - p_mp + p_mm)/(4*h**2)
                matrix[row, col] = d2pdxy
                matrix[col, row] = d2pdxy
        return matrix, gradient

    def integrate_parameter(self, path, h=0.01):

        # TODO: Fix this fricking thing.
        step_number = int(1/h)
        dp_array = zeros((step_number,))
        r = path(linspace(0, 1, step_number), nu=1)
        r_dot = path(linspace(0, 1, step_number), nu=1)
        t = Vector.unit_vector(r_dot)
        y2 = square(self.field.gyro_frequency(r) / self.frequency)
        y_vec = self.field.field_vec(r)*self.field.gyro_frequency(r).reshape(-1, 1) / self.frequency
        x = square(self.atmosphere.plasma_frequency(r)/self.frequency)
        yt = vector_sum(y_vec*t, axis=1)
        current_yp = dot(Vector.unit_vector(y_vec[0]), t[0])
        print(f"YP: {current_yp}")
        current_pt = 1
        print(f"PT: {current_pt}")
        print(f"y2: {y2}")
        print(f"x: {x}")
        for n in range(step_number):
            def equations(p):
                yp, pt = p[0], p[1]
                yp2 = yp**2
                print(f"xn: {x[n]}")
                print(f"y2n: {y2[n]}")
                print(f"yp2: {yp2}")
                a = 1 - x[n] - y2[n] + x[n]*yp2
                b = (x[n] - 1)*(1 - x[n] - y2[n]) + x[n]*y2[n]/2 - x[n]*yp2/2
                print(f"a, b: {a}, {b}")

                # Choosing ordinary ray
                mu2 = 1 - 2*x[n]*(1 - x[n])/(2*(1 - x[n]) - (y2[n] - yp2) +
                                             sqrt((y2[n] - yp2)**2 + 4*(1 - x[n])**2*yp2))
                print(f"mu2: {mu2}")
                fraction = -x[n]*yp*(mu2 - 1)/(a*mu2 + b)
                print(f"Fraction: {fraction}")
                f0 = pt*(pt - (yt[n] - yt[n]*pt)*fraction/2) - 1
                f1 = pt*(yp - (y2[n] - yp2)*fraction/2) - yt[n]
                print(f"F0: {f0}")
                print(f"F1: {f1}")
                return array([f0, f1])

            solution, info, ier, msg = fsolve(equations, array([current_yp, current_pt]), full_output=True)
            if ier != 1:
                print(f"Error on fsolve: {msg}")
                print(f"Function Call Number: {info['nfev']}")
                print(f"Last solution before failure: {solution}")
                raise Exception("It broke")

            current_yp2 = solution[0]**2
            current_pt = solution[1]

            current_mu2 = 1 - 2*x[n]*(1 - x[n]) / (2*(1 - x[n]) - (y2[n] - current_yp2) +
                                                   sqrt((y2[n] - current_yp2)**2 +
                                                   4*(1 - x[n])**2*current_yp2))

            dp_array[n] = current_mu2*current_pt*norm(r_dot)
        # noinspection PyTypeChecker
        integration = simps(dp_array, dx=h)
        return integration

    def visualize(self, plot_all=False):
        fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        ax.set_title(f"3D Ray Trace with a {int(self.frequency/1E6)} MHz frequency")
        atmosphere.visualize(self.initial_coordinates, self.final_coordinates, fig=fig, ax=ax, point_number=200)
        ax.autoscale(False)
        if plot_all and len(self.calculated_paths) > 1:
            custom_lines = [Line2D([0], [0], color='black', lw=4),
                            Line2D([0], [0], color='white', lw=4)]
            ax.legend(custom_lines, ['Best Trace', 'Earlier Traces'])
        else:
            custom_lines = [Line2D([0], [0], color='black', lw=4)]
            ax.legend(custom_lines, ['Best Trace'])
        if plot_all:
            for i in range(len(self.calculated_paths) - 1):
                path = self.calculated_paths[i]
                radii = path.radial_points[:, 1]
                radii = (radii - EARTH_RADIUS)/1000
                km_range = path.radial_points[:, 0]*path.total_angle*EARTH_RADIUS/1000
                ax.plot(km_range, radii, color='white')
        # We always plot the last ones
        path = self.calculated_paths[-1]
        radii = path.radial_points[:, 1]
        radii = (radii - EARTH_RADIUS)/1000
        km_range = path.radial_points[:, 0]*path.total_angle*EARTH_RADIUS/1000
        ax.plot(km_range, radii, color='black')
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")
        plt.show()


if __name__ == "__main__":
    field = BasicField()
    initial = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    atmosphere = ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / PI, -1), initial)
    path_generator = QuasiParabolic
    frequency = 10E6  # Hz

    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = initial, final

    basic_tracer.compile_initial_path()
    basic_tracer.visualize(plot_all=True)

    paths = basic_tracer.trace()
