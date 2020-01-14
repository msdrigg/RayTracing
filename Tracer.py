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
from Constants import pi


DEFAULT_PARAMS = 100, 10


class Tracer:
    def __init__(self, wave_frequency, atmosphere_model, magnetic_field, initial_path_generator):
        if None in (atmosphere_model, magnetic_field, initial_path_generator):
            raise ValueError("Model initializer parameters cannot be Null")
        self.field, self.atmosphere, self.path_generator = magnetic_field, atmosphere_model, initial_path_generator
        self.frequency = wave_frequency

        self.initial_coordinates, self.final_coordinates = None, None
        self.initial_path = None

        self.parameters = None
        self.parameter_number = None
        self.calculated_paths = None

    def compile_initial_path(self):
        if self.initial_coordinates is None or self.final_coordinates is None:
            raise ValueError("Initial and final coordinates must be defined before compiling")

        if self.parameters is None:
            self.parameters = DEFAULT_PARAMS
        self.parameter_number = len(self.parameters)

        if self.calculated_paths is not None:
            raise ValueError("Calculated path is not Null. For safety reasons, you cannot recompile without first"
                             "setting the calculated path to None")

        self.initial_path = self.path_generator(self.initial_coordinates, self.final_coordinates,
                                                self.atmosphere, self.field)
        self.calculated_paths = GreatCircleDeviationPC(*self.parameters, quasi_parabolic=self.initial_path)

    def trace(self, steps=5, parameters=None):
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = DEFAULT_PARAMS
        self.parameter_number = len(self.parameters)

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
        next_path = GreatCircleDeviationPC(*self.parameters, initial_parameters=next_params)
        self.calculated_paths.append(next_path)

    def calculate_derivatives(self, h=0.001):
        gradient = zeros((self.parameter_number,))
        matrix = zeros((self.parameter_number, self.parameter_number))
        curr_path = self.calculated_paths[-1]
        # Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
        for param in range(self.parameter_number):
            path_minus = curr_path.adjust_params(param, -h)
            path_plus = curr_path.adjust_params(param, -h)
            p_minus = self.integrate_parameter(path_minus)
            p_plus = self.integrate_parameter(path_plus)
            p_0 = self.integrate_parameter(curr_path)
            gradient[param] = (p_plus - p_minus)/(2*h)
            matrix[param, param] = (p_plus + p_minus - 2*p_0)/(h**2)

        # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
        for row in range(0, self.parameter_number - 1):
            for col in range(1 + row, self.parameter_number):
                path_mm = curr_path.adjust_params([row, col], -h)
                p_mm = self.integrate_parameter(path_mm)
                path_mp = curr_path.adjust_params([row, col], [-h, h])
                p_mp = self.integrate_parameter(path_mp)
                path_pm = curr_path.adjust_params([row, col], [h, -h])
                p_pm = self.integrate_parameter(path_pm)
                path_pp = curr_path.adjust_params([row, col], h)
                p_pp = self.integrate_parameter(path_pp)
                d2pdxy = (p_pp - p_pm - p_mp + p_mm)/(4*h**2)
                matrix[row, col] = d2pdxy
                matrix[col, row] = d2pdxy
        return matrix, gradient

    def integrate_parameter(self, path, h=0.01):
        step_number = int(1/h)
        dp_array = zeros((step_number,))
        r = path(linspace(0, 1, step_number), nu=1)
        r_dot = path(linspace(0, 1, step_number), nu=1)
        t = Vector.unit_vector(r_dot)
        y_vec = self.field.field_vec(r)*self.field.gyro_frequency(r)
        y_vec /= self.frequency
        x = square(self.atmosphere.plasma_frequency(r)/self.frequency)
        y2 = square(norm(y_vec, axis=1))
        yt = vector_sum(y_vec*t, axis=1)
        current_yp = dot(y_vec[0], t[0])
        current_pt = 1
        for n in range(step_number):
            def equations(p):
                yp, pt = p[0], p[1]
                yp2 = yp**2
                a = 1 - x[n] - y2[n] + x[n]*yp2
                b = (x[n] - 1)*(1 - x[n] - y2[n]) + x[n]*y2[n]/2 - x[n]*yp2/2

                # Choosing ordinary ray
                mu2 = 1 - 2*x[n]*(1 - x[n])/(2*(1 - x[n]) - (y2[n] - yp2) +
                                             sqrt((y2[n] - yp2)**2 + 4*(1 - x[n])**2*yp2))
                fraction = -x[n]*yp*(mu2 - 1)/(a*mu2 + b)
                f0 = pt*(pt - (yt[n] - yt[n]*pt)*fraction/2) - 1
                f1 = pt*(yp - (y2[n] - yp2)*fraction/2) - yt[n]
                return array([f0, f1])

            solution, info, ier, msg = fsolve(equations, array([current_yp, current_pt]))
            if ier != 1:
                print(f"Error on fsolve: {msg}")
                print(f"Function Call Number: {info['nfev']}")
                print(f"Last solution before failure: {solution}")

            current_yp = solution[0]
            current_pt = solution[1]

            dp_array[n] = self.atmosphere.u2(r, current_yp)*current_pt*norm(r_dot)
        # noinspection PyTypeChecker
        integration = simps(dp_array, dx=h)
        return integration

    def visualize(self, plot_all=False, plot_qp=False):
        fig, ax = plt.subplot(1, 1)
        atmosphere.visualize(self.initial_coordinates, self.final_coordinates, fig=fig, ax=ax, point_number=200)
        ax.autoscale(False)
        if plot_all:
            for i in range(len(self.calculated_paths) - 1):
                path = self.calculated_paths[i]
                radii = path.radial_points[:, 1]
                km_range = path.radial_points[:, 0]*path.total_angle*Vector.EARTH_RADIUS/1000
                ax.plot(km_range, radii[:, 0], color='white')
        # We always plot the last ones
        path = self.calculated_paths[-1]
        radii = path.radial_points[:, 1]
        km_range = path.radial_points[:, 0]*path.total_angle*Vector.EARTH_RADIUS/1000
        ax.plot(km_range, radii, color='black')
        plt.show()


if __name__ == "__main__":
    atmosphere = ChapmanLayers(7, 350E3, 100E3, (.375*180/pi, 2), array([Vector.EARTH_RADIUS, pi/2, 0]))
    field = BasicField()
    initial = array([0, 0, 0])
    final = array([1, 1, 1])
    path_generator = QuasiParabolic
    frequency = 30

    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = initial, final

    paths = basic_tracer.trace()
