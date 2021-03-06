import multiprocessing as mp
import random
import time
import warnings
from os.path import join as join_path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg
from scipy.integrate import simps

from atmospheres import BaseAtmosphere
from magnetic_fields import BaseField
from paths import BasePath
from tracing.coleman_equations import equation_15, calculate_yp_pt_cheating, calculate_yp_pt_real
from utilities import Vector
from utilities.Constants import EARTH_RADIUS


class SystemState:
    def __init__(
            self, field: BaseField,
            atmosphere: BaseAtmosphere,
            operating_frequency: float,
            is_extraordinary_ray: bool
    ):
        self.field, self.atmosphere = field, atmosphere
        self.operating_frequency = operating_frequency
        self.is_extraordinary_ray = is_extraordinary_ray


class Tracer:
    def __init__(
            self,
            operating_frequency: float,
            atmosphere_model: BaseAtmosphere,
            magnetic_field: BaseField,
            path_initializer: BasePath,
            cores: Optional[int] = None
    ):
        if None in (operating_frequency, atmosphere_model, magnetic_field, path_initializer):
            raise ValueError("Model initializer parameters cannot be Null")

        self.field, self.atmosphere, = magnetic_field, atmosphere_model
        self.calculated_paths = [path_initializer]
        self.frequency = operating_frequency

        self.pool = None

        if cores is None:
            self.cores = mp.cpu_count() - 2
        else:
            self.cores = cores

    def replace_path(self, new_path: BasePath):
        self.calculated_paths = [new_path]

    def get_system_state(self, is_extraordinary_ray: bool):
        return SystemState(self.field, self.atmosphere, self.frequency, is_extraordinary_ray)

    def trace(
            self, h=10,
            debug_while_calculating=False,
            arrows=False,
            is_extraordinary_ray=False,
            use_cheater_solver: Optional[bool] = True,
            max_steps: int = 10
    ):
        if self.pool is None:
            self.pool = mp.Pool(self.cores)

        last_change = 0

        if debug_while_calculating == 'save':
            save_plots = True
        else:
            save_plots = False

        if debug_while_calculating:
            self.visualize(show_history=True)

        for i in range(1, max_steps):
            print(f"Preforming Newton Raphson Step {i}")
            matrix, gradient, change_vec = self.newton_raphson_step(
                h=h, is_extraordinary_ray=is_extraordinary_ray, use_cheater_solver=use_cheater_solver
            )

            if debug_while_calculating:
                fig, ax = self.visualize(show_history=True, show=False)
                params = self.calculated_paths[-2].adjustable_parameters
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

            total_change = linalg.norm(change_vec)
            if total_change < 10 * np.sqrt(len(change_vec)) and total_change < last_change:
                # Break if the change vec goes too small (small means a change of less than 10 m per position)
                print(
                    f"Ending calculations after {i + 1} steps because change magnitude is small enough and decreasing\n"
                    f"Current change magnitude is {linalg.norm(change_vec)} "
                    f"which is less than {10 * np.sqrt(len(change_vec))}"
                )
                break
            last_change = total_change

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

            if i == max_steps - 1:
                print(
                    "Ending calculations because max step limit reached. "
                    "If convergence isn't complete, rerun with more steps"
                )
        return self.calculated_paths

    def newton_raphson_step(self, h=10, is_extraordinary_ray=False, use_cheater_solver=True):
        matrix, gradient = self.calculate_derivatives(
            h=h,
            is_extraordinary_ray=is_extraordinary_ray,
            use_cheater_solver=use_cheater_solver
        )
        try:
            change = linalg.solve(matrix, gradient, assume_a='sym')
        except linalg.LinAlgError:
            warnings.warn(
                "Using pseudo-inverse to solve matrix equation because matrix is near-singular.\n"
                "Consider using less parameters, or having no angular parameters for this system"
            )
            change = np.matmul(linalg.pinvh(matrix), gradient)
        change_mag = linalg.norm(change)
        print(f"Change magnitude: {change_mag}")

        next_path = self.calculated_paths[-1].adjust_parameters(np.arange(change.shape[0]), -change)
        self.calculated_paths.append(next_path)
        return matrix, gradient, change

    def calculate_derivatives(self, h=10, is_extraordinary_ray=False, use_cheater_solver=True):
        # We need to make sure our integration step size is significantly smaller than our derivative
        #   or else our truncation error will be too large
        integration_step = 1 / 2000.0
        parameter_number = self.calculated_paths[-1].adjustable_parameters.shape[0]

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

        total_diagonal_ints = 3 * parameter_number

        # Parallelize calculation of directional diagonal derivatives
        diagonal_d_results = self.pool.map_async(
            diagonal_dirs,
            zip(
                [self.get_system_state(is_extraordinary_ray) for _ in range(parameter_number)],
                list(range(parameter_number)),
                [self.calculated_paths[-1] for _ in range(parameter_number)],
                [integration_step for _ in range(parameter_number)],
                [h for _ in range(parameter_number)],
                [use_cheater_solver for _ in range(parameter_number)]
            )
        )

        off_diagonal_elements = int(parameter_number * (parameter_number - 1) / 2)

        # Parallelize calculation of directional off-diagonal derivatives
        off_diagonal_d_results = self.pool.map_async(
            off_diagonal_dirs,
            zip(
                [self.get_system_state(is_extraordinary_ray) for _ in range(off_diagonal_elements)],
                [pair_generator(n, parameter_number) for n in range(off_diagonal_elements)],
                [self.calculated_paths[-1] for _ in range(off_diagonal_elements)],
                [integration_step for _ in range(off_diagonal_elements)],
                [h for _ in range(off_diagonal_elements)],
                [use_cheater_solver for _ in range(off_diagonal_elements)]
            )
        )

        # noinspection PyProtectedMember
        print(
            f"Calculating {total_diagonal_ints + off_diagonal_elements * 4} integrations "
            f"with {self.pool._processes} processes "
        )
        for result in diagonal_d_results.get():
            index = int(result[0])
            gradient[index] = result[1]
            matrix[index, index] = result[2]

        for result in off_diagonal_d_results.get():
            row_idx, col_idx = int(result[0]), int(result[1])
            matrix[row_idx, col_idx] = result[2]
            matrix[col_idx, row_idx] = result[2]

        return matrix, gradient

    def visualize(self, show_history=False, show=True, fig=None, ax=None, color='black'):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
            ax.set_title(f"3D Ray Trace with a {int(self.frequency / 1E6)} MHz frequency")
            self.atmosphere.visualize(
                self.calculated_paths[0](0),
                self.calculated_paths[0](1),
                fig=fig,
                ax=ax,
            )
            ax.autoscale(False)
            ax.set_ylabel("Altitude (km)")
            ax.set_xlabel("Range (km)")
            if show_history and len(self.calculated_paths) > 1:
                custom_lines = [Line2D([0], [0], color='black', lw=4),
                                Line2D([0], [0], color='white', lw=4)]
                ax.legend(custom_lines, ['Best Trace', 'Earlier Traces'])
            else:
                custom_lines = [Line2D([0], [0], color='black', lw=4)]
                ax.legend(custom_lines, ['Best Trace'])

        evaluations = np.linspace(0, 1, 200)
        if show_history:
            for i in range(len(self.calculated_paths) - 1):
                path = self.calculated_paths[i]
                points = path(evaluations)
                radii = (linalg.norm(points, axis=-1) - EARTH_RADIUS) / 1000
                km_range = evaluations * path.total_angle * EARTH_RADIUS / 1000
                ax.plot(km_range, radii, color='white')

        # We always plot the current ones
        path = self.calculated_paths[-1]
        points = path(evaluations)
        radii = (linalg.norm(points, axis=-1) - EARTH_RADIUS) / 1000
        km_range = evaluations * path.total_angle * EARTH_RADIUS / 1000
        ax.plot(km_range, radii, color=color)

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


def integrate_parameter(
        system_state: SystemState,
        path: BasePath, h=0.00001,
        show=False, save=None,
        use_cheater_solver=True
):
    step_number = int(1 / h)
    r = path(np.linspace(0, 1, step_number), nu=0)
    r_dot = path(np.linspace(0, 1, step_number), nu=1)
    r_dot_norm = linalg.norm(r_dot, axis=1)
    t = r_dot / r_dot_norm.reshape(-1, 1)

    y = system_state.field.gyro_frequency(r) / system_state.operating_frequency
    y_vec = system_state.field.field_vec(r) * y.reshape(-1, 1)

    y_squared = np.square(y)
    x = np.square(system_state.atmosphere.plasma_frequency(r) / system_state.operating_frequency)
    yt = Vector.row_dot_product(y_vec, t)

    sign = 1
    if system_state.is_extraordinary_ray:
        sign = -1

    # TODO: Fix real yp/pt solver
    if use_cheater_solver:
        solved_yp, solved_pt = calculate_yp_pt_cheating(yt)
    else:
        solved_yp, solved_pt = calculate_yp_pt_real(
            x, y, y_squared, yt, sign=sign
        )
    # plt.plot(solved_yp)
    # plt.plot(solved_pt)
    # plt.show()
    # plt.close()

    current_mu_squared = equation_15(solved_yp, x, y_squared, sign=sign)

    dp_array = np.sqrt(current_mu_squared) * solved_pt * r_dot_norm
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
    system_state, index_pair, curr_path, int_h, vary_h, use_cheater_solver = inputs

    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.

    if abs(index_pair[0] - index_pair[1]) > 9:
        return np.array([index_pair[0], index_pair[1], 0])

    path_mm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], -vary_h)
    p_mm = integrate_parameter(system_state, path_mm, h=int_h, use_cheater_solver=use_cheater_solver)
    path_mp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [-vary_h, vary_h])
    p_mp = integrate_parameter(system_state, path_mp, h=int_h, use_cheater_solver=use_cheater_solver)
    path_pm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [vary_h, -vary_h])
    p_pm = integrate_parameter(system_state, path_pm, h=int_h, use_cheater_solver=use_cheater_solver)
    path_pp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], vary_h)
    p_pp = integrate_parameter(system_state, path_pp, h=int_h, use_cheater_solver=use_cheater_solver)
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * vary_h ** 2)

    return np.array([index_pair[0], index_pair[1], output])


# Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
def diagonal_dirs(inputs):
    system_state, varied_parameter, curr_path, int_h, vary_h, use_cheater_solver = inputs
    path_minus = curr_path.adjust_parameters(varied_parameter, -vary_h)
    path_plus = curr_path.adjust_parameters(varied_parameter, vary_h)
    p_minus = integrate_parameter(system_state, path_minus, h=int_h, use_cheater_solver=use_cheater_solver)
    p_plus = integrate_parameter(system_state, path_plus, h=int_h, use_cheater_solver=use_cheater_solver)
    p_0 = integrate_parameter(system_state, curr_path, h=int_h, use_cheater_solver=use_cheater_solver)
    dp_dx = (p_plus - p_minus) / (2 * vary_h)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (vary_h ** 2)
    return np.array([varied_parameter, dp_dx, d2pdx2])
