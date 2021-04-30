import multiprocessing as mp
import warnings
from os.path import join as join_path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import simps
from scipy import linalg

import Constants
import Paths
from Atmosphere import Atmosphere
from Constants import EARTH_RADIUS
from Equations import equation_15
from Field import Field
from Paths import GreatCircleDeviation, Path

from SystemState import SystemState


def integrate_parameter(system_state: SystemState, path: Path, h=0.00001, show=False, save=False):
    step_number = int(1 / h)
    r = path(np.linspace(0, 1, step_number), nu=0)
    r_dot = path(np.linspace(0, 1, step_number), nu=1)
    y_squared = np.square(system_state.field.gyro_frequency(r) / system_state.operating_frequency)
    x = np.square(system_state.atmosphere.plasma_frequency(r) / system_state.operating_frequency)

    # TODO: Implement working yp/pt solver
    solved_yp = np.zeros(step_number)
    current_pt = np.repeat(1, step_number)
    sign = -1
    if system_state.is_extraordinary_ray:
        sign = 1

    current_mu2 = equation_15(solved_yp, x, y_squared, sign=sign)

    dp_array = np.sqrt(current_mu2) * current_pt * linalg.norm(r_dot, axis=1)
    integration = simps(dp_array, dx=h)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.plot(dp_array)
        if save is not None:
            fig.savefig(join_path("saved_plots", f'TotalPValues_{save}.png'))
        else:
            plt.show()
        plt.close(fig)
    return integration


def off_diagonal_dirs(inputs):
    system_state, index_pair, curr_path, int_h, vary_h = inputs

    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.

    if abs(index_pair[0] - index_pair[1]) > 9:
        return np.array([index_pair[0], index_pair[1], 0])

    path_mm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], -vary_h)
    p_mm = integrate_parameter(system_state, path_mm, h=int_h)
    path_mp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [-vary_h, vary_h])
    p_mp = integrate_parameter(system_state, path_mp, h=int_h)
    path_pm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [vary_h, -vary_h])
    p_pm = integrate_parameter(system_state, path_pm, h=int_h)
    path_pp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], vary_h)
    p_pp = integrate_parameter(system_state, path_pp, h=int_h)
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * vary_h ** 2)

    return np.array([index_pair[0], index_pair[1], output])


# Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
def diagonal_dirs(inputs):
    system_state, varied_parameter, curr_path, int_h, vary_h = inputs
    path_minus = curr_path.adjust_parameters(varied_parameter, -vary_h)
    path_plus = curr_path.adjust_parameters(varied_parameter, vary_h)
    p_minus = integrate_parameter(system_state, path_minus, h=int_h)
    p_plus = integrate_parameter(system_state, path_plus, h=int_h)
    p_0 = integrate_parameter(system_state, curr_path, h=int_h)
    dp_dx = (p_plus - p_minus) / (2 * vary_h)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (vary_h ** 2)
    return np.array([varied_parameter, dp_dx, d2pdx2])


class Tracer:
    def __init__(
            self,
            wave_frequency: float,
            atmosphere_model: Atmosphere,
            magnetic_field: Field,
            path_initializer: Paths.QuasiParabolic
    ):
        if None in (wave_frequency, atmosphere_model, magnetic_field, path_initializer):
            raise ValueError("Model initializer parameters cannot be Null")

        self.field, self.atmosphere, = magnetic_field, atmosphere_model
        self.initial_path = path_initializer
        self.frequency = wave_frequency

        self.initial_coordinates, self.final_coordinates = None, None

        # If you manually set the parameters, you must also set the parameter number
        self.parameters = None
        self.calculated_paths = None
        self.pool = None

    def compile_initial_path(self, high_ray=True):
        if self.initial_coordinates is None or self.final_coordinates is None:
            raise ValueError("Initial and final coordinates must be defined before compiling")

        self.initial_path.using_high_ray = high_ray
        self.initial_path.compile_points()

        new_path = GreatCircleDeviation(*self.parameters, quasi_parabolic=self.initial_path)
        new_path.interpolate_params()

        self.calculated_paths = [new_path]

    def get_system_state(self, is_extraordinary_ray: bool):
        return SystemState(self.field, self.atmosphere, self.frequency, is_extraordinary_ray)

    def trace(
            self, steps=50, h=1,
            parameters=None,
            debug_while_calculating=True,
            arrows=False,
            is_extraordinary_ray=False,
            high_ray=False
    ):
        if self.pool is None:
            self.pool = mp.Pool(mp.cpu_count() - 2)

        if debug_while_calculating == 'save':
            save_plots = True
        else:
            save_plots = False

        if parameters is not None:
            self.parameters = parameters
        elif self.parameters is None:
            self.parameters = Constants.DEFAULT_PARAMS

        self.compile_initial_path(high_ray=high_ray)

        if debug_while_calculating:
            self.visualize(show_history=True)

        for i in range(1, steps):
            print(f"Preforming Newton Raphson Step {i}")
            matrix, gradient, change_vec = self.newton_raphson_step(
                h=h, is_extraordinary_ray=is_extraordinary_ray
            )

            if debug_while_calculating:
                fig, ax = self.visualize(show_history=True, show=False)
                params = self.calculated_paths[-2].parameters
                total_angle = self.calculated_paths[-2].total_angle
                if arrows:
                    for n, param in enumerate(params[::int(len(change_vec) / 25)]):
                        # Plot change vec
                        x_c, dx_c = param[0] * EARTH_RADIUS * total_angle / 1000, 0
                        y_c, dy_c = (param[1] - EARTH_RADIUS) / 1000 - 20, -change_vec[
                            n * int(len(change_vec) / 25)] / 1000
                        ax.arrow(x_c, y_c, dx_c, dy_c, color='black', width=3, head_width=12, head_length=12)
                        x_g, dx_g = param[0] * EARTH_RADIUS * total_angle / 1000, 0
                        y_g, dy_g = (param[1] - EARTH_RADIUS) / 1000 + 20, gradient[
                            n * int(len(change_vec) / 25)] / 1000
                        ax.arrow(x_g, y_g, dx_g, dy_g, color='white', width=3, head_width=12, head_length=12)

                if save_plots:
                    fig.savefig(join_path("saved_plots", f'TotalChange_{i}.png'))
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)

                plt.plot(gradient)
                plt.suptitle("Gradient Graph")
                if save_plots:
                    plt.savefig(join_path("saved_plots", f'Gradient_{i}.png'))
                    plt.close()
                else:
                    plt.show()
                    plt.close()

                current_p = integrate_parameter(
                    self.get_system_state(is_extraordinary_ray=is_extraordinary_ray),
                    self.calculated_paths[-1],
                    show=True, save=False
                )
                print(f"Current total phase angle: {current_p}")
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                image = ax.imshow(matrix)
                color_bar = fig.colorbar(image, ax=ax)
                color_bar.set_label("Second Derivative")
                plt.suptitle("Matrix graph")

                if save_plots:
                    fig.savefig(join_path("saved_plots", f'Hessian Matrix_{i}.png'))
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)

            if linalg.norm(change_vec) < 10 * np.sqrt(len(change_vec)):
                # Break if the change vec goes too small (small means a change of less than 10 m per position)
                print(
                    f"Ending calculations after {i + 1} steps because change magnitude converged sufficiently \n"
                    f"Current change magnitude is {linalg.norm(change_vec)} \n"
                    f"which is less than {10 * np.sqrt(len(change_vec))}"
                )
                break
            evaluations = np.linspace(0, 1, 1000)
            starting_path_derivative = self.calculated_paths[0](evaluations, nu=1)
            current_path_derivative = self.calculated_paths[-1](evaluations, nu=1)
            current_derivative_integral = simps(linalg.norm(current_path_derivative, axis=-1), evaluations)
            initial_derivative_integral = simps(linalg.norm(starting_path_derivative, axis=-1), evaluations)
            if current_derivative_integral > 2 * initial_derivative_integral:
                # Break if the path gets too choppy
                warnings.warn(
                    f"Ending calculations after {i + 1} steps because path got too choppy \n"
                    f"Total path derivative norm was {current_derivative_integral} \n"
                    f"which is more than twice the starting value of {initial_derivative_integral}\n"
                    "It is likely that this path is not convergent"
                )
                break

        print("Done tracing path")
        return self.calculated_paths

    def newton_raphson_step(self, h=1, is_extraordinary_ray=False):
        matrix, gradient = self.calculate_derivatives(h=h, is_extraordinary_ray=is_extraordinary_ray)
        try:
            change = linalg.solve(matrix, gradient, assume_a='sym')
        except linalg.LinAlgError:
            warnings.warn(
                "Using pseudo-inverse to solve matrix equation because matrix is near-singular.\n"
                "Consider using less parameters, or having no normal-component parameters for this system"
            )
            change = np.matmul(linalg.pinvh(matrix), gradient)
        change_mag = linalg.norm(change)
        print(f"Change magnitude: {change_mag}")

        next_params = self.calculated_paths[-1].parameters[:, 1] - change
        next_path = GreatCircleDeviation(
            *self.parameters, initial_parameters=next_params,
            initial_coordinate=self.initial_coordinates,
            final_coordinate=self.final_coordinates, using_spherical=False
        )
        self.calculated_paths.append(next_path)
        return matrix, gradient, change

    def calculate_derivatives(self, h=1, is_extraordinary_ray=False):
        # We need to make sure our integration step size is significantly smaller than our derivative
        #   or else our truncation error will be too large
        integration_step = 1 / 2000.0

        parameter_number = self.parameters[0] + self.parameters[1]

        # dP/da_i
        gradient = np.zeros((parameter_number,))

        # d^2P/(da_ida_j)
        matrix = np.zeros((parameter_number, parameter_number))

        # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
        def pair_generator(item_number, param_number=parameter_number):
            counter = 0
            for row in range(0, param_number - 1):
                for col in range(1 + row, param_number):
                    if counter == item_number:
                        return [row, col]
                    counter += 1
            raise IndexError("You are indexing for a pair that doesn't exist")

        total_ints = 3 * parameter_number
        print("Calculating Derivatives for the diagonal elements")
        print(f"Expecting {total_ints} diagonal integrations")

        # Parallelize calculation of directional diagonal derivatives
        diagonal_d_results = self.pool.map_async(
            diagonal_dirs,
            zip(
                [self.get_system_state(is_extraordinary_ray) for _ in range(parameter_number)],
                list(range(parameter_number)),
                [self.calculated_paths[-1] for _ in range(parameter_number)],
                [integration_step for _ in range(parameter_number)],
                [h for _ in range(parameter_number)]
            )
        )

        elements = int(parameter_number * (parameter_number - 1) / 2)
        print("Calculating Derivatives for the off diagonal elements")
        print(f"Expecting {elements * 4} integrations")

        # Parallelize calculation of directional off-diagonal derivatives
        off_diagonal_d_results = self.pool.map_async(
            off_diagonal_dirs,
            zip(
                [self.get_system_state(is_extraordinary_ray) for _ in range(elements)],
                [pair_generator(n, parameter_number) for n in range(elements)],
                [self.calculated_paths[-1] for _ in range(elements)],
                [integration_step for _ in range(elements)],
                [h for _ in range(elements)]
            )
        )

        for result in diagonal_d_results.get():
            index = int(result[0])
            gradient[index] = result[1]
            matrix[index, index] = result[2]
        print("Completed Diagonal Integrations")

        for result in off_diagonal_d_results.get():
            row_idx, col_idx = int(result[0]), int(result[1])
            matrix[row_idx, col_idx] = result[2]
            matrix[col_idx, row_idx] = result[2]
        print("Completed off diagonal integrations")

        return matrix, gradient

    def visualize(self, show_history=False, show=True):
        fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        ax.set_title(f"3D Ray Trace with a {int(self.frequency / 1E6)} MHz frequency")
        self.atmosphere.visualize(self.initial_coordinates, self.final_coordinates, fig=fig, ax=ax, point_number=200)
        ax.autoscale(False)
        if show_history and len(self.calculated_paths) > 1:
            custom_lines = [Line2D([0], [0], color='black', lw=4),
                            Line2D([0], [0], color='white', lw=4)]
            ax.legend(custom_lines, ['Best Trace', 'Earlier Traces'])
        else:
            custom_lines = [Line2D([0], [0], color='black', lw=4)]
            ax.legend(custom_lines, ['Best Trace'])
        if show_history:
            for i in range(len(self.calculated_paths) - 1):
                path = self.calculated_paths[i]
                radii = path.radial_points[:, 1]
                radii = (radii - EARTH_RADIUS) / 1000
                km_range = path.radial_points[:, 0] * path.total_angle * EARTH_RADIUS / 1000
                ax.plot(km_range, radii, color='white')

        # We always plot the last ones
        path = self.calculated_paths[-1]
        radii = path.radial_points[:, 1]
        radii = (radii - EARTH_RADIUS) / 1000
        km_range = path.radial_points[:, 0] * path.total_angle * EARTH_RADIUS / 1000
        ax.plot(km_range, radii, color='black')
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")
        if show:
            plt.show()
            plt.close(fig)
        else:
            return fig, ax

    def cleanup(self):
        if self.pool is not None:
            print("Closing and shutting down pool")
            self.pool.close()
            self.pool.terminate()
            self.pool = None
