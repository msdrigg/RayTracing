"""
Function that contains the methods to trace the path.
The entry point is the trace() function
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, integrate, optimize, interpolate
import os
from core import initialize
from core import vector, plotting, path, constants, equations, parallel, coordinates as coords
import multiprocessing as mp
from typing import *
import math
import warnings
import importlib
from atmosphere import base as atmosphere_base
from magnetic import base as magnetic_base

# Setting up global atmosphere and magnetic variables
atmosphere: atmosphere_base = None
magnetic: magnetic_base = None


# Variable to control a potential optimization. True to optimize, false to disable
INTEGRATING_PATH_PREEMPTIVE = True


def load_dynamic_modules(atmosphere_module_name: str, magnetic_field_module_name: str):
    """
    Loads the proper atmosphere module and magnetic field module into a global namespace
    This is necessary because atmosphere and magnetic field implementations are chosen at runtime
    :param atmosphere_module_name: Name of the file (in atmosphere package) to import
    :param magnetic_field_module_name: Name of the file (in magnetic package) to import
    """
    global atmosphere
    atmosphere = importlib.import_module('atmosphere.' + atmosphere_module_name)
    global magnetic
    magnetic = importlib.import_module('magnetic.' + magnetic_field_module_name)


def solve_yp_pt(x: float, y: float, y_squared: float, yt: float):
    """
    Given the parameters, return the solution to yp
    :return: yp solved
    """
    function_args = x, y_squared, yt
    # TODO: Optimize this. This function gets run nearly 2000*2500 times for each newton-raphson step
    #   Try brent's method. It's not as quickly convergent but it is written in C
    # noinspection PyTypeChecker
    yp_solved, details = optimize.brentq(
        equations.equation_13, -y, y,
        args=function_args, xtol=1E-15, rtol=1E-15, full_output=True
    )
    if not details.converged:
        warnings.warn(
            f"Error solving for yp using equation 13. Attempted {details.iterations} iterations "
            f"and resulted in {details.root}, "
            f"stopping with reason: {details.flag}."
        )
    return yp_solved


def integrate_over_path(
        cartesian_coordinate_callable: Callable,
        operating_frequency: float,
        integration_step_number: int = 2 ** 12 + 1,
        show: bool = False,
        save: bool = False,
        debug_zero_field: Optional[bool] = False) -> float:
    """
    Calculate P defined in Coleman as the integral of mu * pt over the path length
    See Section 3.1 for this definition.
    All single letter/double letter variable names match their corresponding value in Coleman 2011 equations
    :param cartesian_coordinate_callable: A callable that takes values between 0 and 1 and
    returns the cartesian components of the path
    :param operating_frequency: The frequency of the ray
    :param integration_step_number: The number of steps to use in integration
    :param show: Whether or not to display the path as a graph (debug only)
    :param save: Whether or not to save the plot (debug only)
    :param debug_zero_field: Whether or not to use the 0 field (debug only)
    :return: The integral of mu * pt over the path length (this is defined as P in Coleman 2011)
    """
    r = cartesian_coordinate_callable(np.linspace(0, 1, integration_step_number), nu=0)
    r_norm = linalg.norm(r, axis=1)
    r_dot = cartesian_coordinate_callable(np.linspace(0, 1, integration_step_number), nu=1)
    r_dot_norm = linalg.norm(r_dot, axis=1)

    t = r_dot / r_dot_norm
    # noinspection PyUnresolvedReferences
    gyro_frequency_squared_array = magnetic.calculate_gyro_frequency(r, r_norm)
    y = gyro_frequency_squared_array / operating_frequency
    y_squared = np.square(y)

    # noinspection PyUnresolvedReferences
    y_vec = magnetic.calculate_magnetic_field_unit_vec(r, r_norm) * gyro_frequency_squared_array / operating_frequency
    # noinspection PyUnresolvedReferences
    x = atmosphere.calculate_plasma_frequency(r, r_norm) / operating_frequency ** 2
    yt = vector.row_dot_product(y_vec, t)

    yp = np.zeros(integration_step_number)

    # TODO: Optimize this
    for n in range(integration_step_number):
        if debug_zero_field:
            yp[n] = 0
        else:
            yp[n] = solve_yp_pt(x[n], y[n], y_squared[n], yt[n])

    yp_squared = np.square(yp)
    fractions = equations.equation_16(yp, yp_squared, x, y_squared)
    current_pt = equations.equation_14(yp, yp_squared, y_squared, fractions, yt)
    yp[current_pt < 0] = -yp[current_pt < 0]
    current_pt = np.absolute(current_pt)
    mu_squared = equations.equation_15(yp_squared, x, y_squared)
    dp_array = np.sqrt(mu_squared) * current_pt * r_dot_norm
    integration = integrate.romb(dp_array, 1 / integration_step_number)

    if debug_zero_field:
        plotting.visualize_tracing_debug(r, r_dot)

    if show:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.plot(dp_array)
        if save is not None:
            fig.savefig(os.path.join("SavedPlots", f'TotalPValues_{save}.png'))
        else:
            plt.show()
        plt.close(fig)

    return np.asscalar(integration)


def calculate_off_diagonal_dirs(
        index_pair: Tuple[int, int],
        current_path_memory_name: str,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        integration_step_number: int,
        gradient_step_size: float,
        operating_frequency: float) -> np.ndarray:
    """
    Calculates the off-diagonal derivatives for calculating the hessian of P
    These calculations appear in Equation 19 of Coleman 2011
    :param index_pair: The second derivative is calculated with respect to this pair of parameters
    :param current_path_memory_name: The memory name for the current path (The evaluation point of the derivative)
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param integration_step_number: The integration step size
    :param gradient_step_size: The amount to vary the parameter in the calculation of the derivative
    :param operating_frequency: The frequency of the ray
    :return: The mixed second derivative of the functional P with respect to the parameters in index_pair
    """

    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.
    if abs(index_pair[0] - index_pair[1]) > 9:
        return np.array([*index_pair, 0])

    with parallel.read_array_shared_memory(current_path_memory_name) as current_path_shared:
        current_path = np.concatenate(current_path_shared)
        radial_parameter_count = current_path_shared[0].size

    path_mm_array = current_path.copy()
    path_mm_array[index_pair] -= gradient_step_size
    path_mm_parameters = np.split(path_mm_array, (radial_parameter_count, ))
    path_mm_callable = path.generate_cartesian_callable(path_mm_parameters, start_point_spherical, end_point_spherical)

    p_mm = integrate_over_path(
        path_mm_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )

    path_mp_array = current_path.copy()
    path_mp_array[index_pair] += [-gradient_step_size, gradient_step_size]
    path_mp_parameters = np.split(path_mp_array, (radial_parameter_count, ))
    path_mp_callable = path.generate_cartesian_callable(path_mp_parameters, start_point_spherical, end_point_spherical)
    p_mp = integrate_over_path(
        path_mp_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )
    path_pm_array = current_path.copy()
    path_pm_array[index_pair] += [gradient_step_size, -gradient_step_size]
    path_pm_parameters = np.split(path_pm_array, (radial_parameter_count, ))
    path_pm_callable = path.generate_cartesian_callable(path_pm_parameters, start_point_spherical, end_point_spherical)
    p_pm = integrate_over_path(
        path_pm_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )
    path_pp_array = current_path.copy()
    path_pp_array[index_pair] += gradient_step_size
    path_pp_parameters = np.split(path_pp_array, (radial_parameter_count, ))
    path_pp_callable = path.generate_cartesian_callable(path_pp_parameters, start_point_spherical, end_point_spherical)
    p_pp = integrate_over_path(
        path_pp_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * gradient_step_size ** 2)

    return np.array([*index_pair, output])


def calculate_diagonal_derivatives(
        varied_parameter: int,
        current_path_memory_name: str,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        integration_step_number: int,
        gradient_step_size: float,
        operating_frequency: float,
        current_path_integration: Optional[float] = None) -> np.ndarray:
    """
    Calculate the second derivative and the gradient vector.
    These calculations involve the same function calls
    The gradient is an implementation of Equation 18 in Coleman 2011, and the second derivative
    appears in Equation 19 of the same paper
    :param varied_parameter: The gradient is calculated with respect to this parameter
    :param current_path_memory_name: The memory name for the current path (The evaluation point of the derivative)
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param integration_step_number: The integration step size
    :param gradient_step_size: The amount to vary the parameter in the calculation of the derivative
    :param operating_frequency: The frequency of the ray
    :param current_path_integration: The result of integrating over the current unchanged path. This is an optimization
    that allows us to reuse current path integrations across different derivative calculations.
    :return: The first and second derivative of the functional P with respect to the parameter
    """
    with parallel.read_array_shared_memory(current_path_memory_name) as current_path_shared:
        current_path = np.concatenate(current_path_shared)
        radial_parameter_count = current_path_shared[0].size

    path_minus = current_path.copy()
    path_minus[varied_parameter] -= gradient_step_size
    path_minus_parameters = np.split(path_minus, (radial_parameter_count, ))
    path_minus_callable = path.generate_cartesian_callable(
        path_minus_parameters, start_point_spherical, end_point_spherical)

    path_plus = current_path.copy()
    path_plus[varied_parameter] += gradient_step_size
    path_plus_parameters = np.split(path_plus, (radial_parameter_count, ))
    path_plus_callable = path.generate_cartesian_callable(
        path_plus_parameters, start_point_spherical, end_point_spherical)

    p_minus = integrate_over_path(
        path_minus_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )
    p_plus = integrate_over_path(
        path_plus_callable,
        operating_frequency,
        integration_step_number=integration_step_number
    )
    if current_path_integration is None:
        path_current_callable = path.generate_cartesian_callable(
            current_path,
            start_point_spherical,
            end_point_spherical
        )
        p_0 = integrate_over_path(
            path_current_callable,
            operating_frequency,
            integration_step_number=integration_step_number
        )
    else:
        p_0 = current_path_integration

    dpdx = (p_plus - p_minus) / (2 * gradient_step_size)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (gradient_step_size ** 2)
    return np.array([varied_parameter, dpdx, d2pdx2])


def calculate_derivatives(
        current_path_parameters: Tuple[np.ndarray, np.ndarray],
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        operating_frequency: float,
        pool: Optional[mp.Pool],
        variable_indices: Tuple[np.ndarray, np.ndarray],
        derivative_step_size: Optional[float] = 1000,
        integration_step_number: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a newton raphson step as defined in Equation 19 of Coleman 2011. Thi
    See Section 3.1 for this definition.
    :param current_path_parameters: The current path parameters as a tuple of (radial parameters, normal parameters)
    These are evenly spaced
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param operating_frequency: The frequency of the ray
    :param variable_indices: Two arrays of indices for which we should calculate derivatives.
    The first corresponds to the radial components and the second to the normal components
    :param derivative_step_size: The step size in meters over which to calculate the gradient
    :param integration_step_number: The number of points to use in integrating over the path.
    We use romberg integration, so we must have integration_point_count be of the form 2 ^ n + 1
    :param pool: The processor pool to calculate the derivatives with. If none, create one
    :return: A tuple of (The hessian due to the The next path after the iteration)
    """
    # We need to make sure our integration step size is significantly smaller than our derivative
    #   or else our truncation error will be too large

    total_ground_distance = vector.angle_between_vector_collections(
        coords.spherical_to_cartesian(start_point_spherical),
        coords.spherical_to_cartesian(end_point_spherical)
    ).item()

    # Calculate integration_step_number using the derivative_step_size if not given
    if integration_step_number is None:
        integration_rough_estimate = \
            total_ground_distance / derivative_step_size * constants.INTEGRATION_STEP_SIZE_FACTOR
        approximate_power = math.ceil(math.log(integration_rough_estimate - 1, 2))
        integration_step_number = 2 ** approximate_power + 1
    else:
        approximate_power = round(math.log(integration_step_number - 1, 2))
        approximate_point_count = 2 ** approximate_power + 1
        if approximate_point_count != integration_step_number:
            warnings.warn("integration_point_count must be 2^n + 1 for rompberg integration"
                          "the nearest power of 2 is {}, "
                          "while the entered value is {}. Rounding to make it work"
                          .format(approximate_point_count, integration_step_number), RuntimeWarning)
            integration_step_number = approximate_point_count

    variable_indices_array = np.concatenate((variable_indices[0], variable_indices[1] + variable_indices[0].size))
    parameter_number = variable_indices_array.size

    # dP/da_i
    gradient = np.zeros(parameter_number)

    # d^2P/(da_ida_j)
    matrix = np.zeros((parameter_number, parameter_number))

    # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
    def pair_generator(matrix_size):
        """
        Generates pairs of numbers that correspond to numbers in the upper triangular portion of a matrix
        This is used to calculate the off-diagonal derivatives
        :param matrix_size: The number of elements in one row or column of the matrix (matrix needs to be square)
        :return: A generator generating index pairs
        """
        for pair_gen_row in range(0, matrix_size - 1):
            for pair_gen_col in range(1 + pair_gen_row, matrix_size):
                yield [pair_gen_row, pair_gen_col]
        yield StopIteration("You are looking for an index pair that doesn't exist. "
                            "All available have already been returned")

    if INTEGRATING_PATH_PREEMPTIVE:
        path_current_callable = path.generate_cartesian_callable(
            current_path_parameters,
            start_point_spherical,
            end_point_spherical
        )

        current_path_integration = integrate_over_path(
            path_current_callable,
            operating_frequency,
            integration_step_number=integration_step_number
        )
    else:
        current_path_integration = None

    # TODO: Should I also pass other shared elements as well
    #   I'm thinking path_start_point, path_end_point
    with parallel.create_parameters_shared_memory(current_path_parameters) as shared_memory:
        total_ints = 3*parameter_number
        print("Calculating Derivatives for the diagonal elements")
        print(f"Expecting {total_ints} diagonal integrations")

        # Parallelize calculation of directional diagonal derivatives
        diagonal_async = pool.starmap(
            calculate_diagonal_derivatives,
            zip(
                variable_indices_array,
                [start_point_spherical] * parameter_number,
                [end_point_spherical] * parameter_number,
                [shared_memory.name] * parameter_number,
                [integration_step_number] * parameter_number,
                [derivative_step_size] * parameter_number,
                [operating_frequency] * parameter_number,
                [current_path_integration] * parameter_number
            )
        )

        print("Calculating Derivatives for the off diagonal elements")
        elements = int(parameter_number*(parameter_number-1)/2)
        print(f"Expecting {elements*4} off-diagonal integrations")

        # Parallelize calculation of directional off diagonal derivatives
        off_diagonal_async = pool.starmap(
            calculate_off_diagonal_dirs,
            zip(
                variable_indices_array[list(pair_generator(parameter_number))],
                [start_point_spherical] * elements,
                [end_point_spherical] * elements,
                [shared_memory.name] * elements,
                [integration_step_number] * elements,
                [derivative_step_size] * elements,
                [operating_frequency] * elements,
            )
        )

    for result in diagonal_async.get():
        index = int(result[0])
        gradient[index] = result[1]
        matrix[index, index] = result[2]
    print("Completed Diagonal Integrations")

    for result in off_diagonal_async.get():
        row, col = int(result[0]), int(result[1])
        matrix[row, col] = result[2]
        matrix[col, row] = result[2]
    print("Completed All Integrations")

    return matrix, gradient


def newton_raphson_step(
        current_path_parameters: Tuple[np.ndarray, np.ndarray],
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        operating_frequency: float,
        fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = (np.empty((0,)), np.empty((0,))),
        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a newton raphson step as defined in Equation 19 of Coleman 2011. Thi
    See Section 3.1 for this definition.
    :param current_path_parameters: The current path parameters (alpha_i) before the newton raphson step as
    a tuple of (radial parameters, normal paramters)
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param operating_frequency: The frequency of the ray
    :param fixed_indices: Two arrays of indices which do not change.
    The first corresponds to the radial components and the second to the normal components
    :param kwargs: The rest of the arguments are passed to the derivative calculation
    :return: A tuple of (The hessian due to the The next path after the iteration)
    """
    variable_params = []
    for i in range(2):
        variable_params.append(np.delete(np.arange(current_path_parameters[i].size), fixed_indices[i]))

    incomplete_hessian, incomplete_gradient = calculate_derivatives(
        current_path_parameters,
        start_point_spherical,
        end_point_spherical,
        operating_frequency,
        variable_params,
        **kwargs
    )

    inverse_matrix = linalg.pinvh(incomplete_hessian)
    incomplete_change = np.matmul(inverse_matrix, incomplete_gradient)

    variable_params_array = np.concatenate((variable_params[0], variable_params[1] + variable_params[0].size))
    change = np.zeros_like(current_path_parameters)
    change[variable_params_array] = incomplete_change

    change_mag = linalg.norm(change)

    print(f"Change magnitude: {change_mag}")

    return incomplete_hessian, incomplete_gradient, change


def trace(
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        atmosphere_module_name: str,
        magnetic_field_module_name: str,
        operating_frequency: float,
        atmosphere_parameters: Tuple[float, ...],
        max_newton_raphson_steps: Optional[int] = 50,
        minimum_change_per_position: Optional[int] = 50,
        integration_step_number: Optional[int] = None,
        derivative_step_size: Optional[float] = 1000,
        parameters: Optional[Tuple] = None,
        interpolated_degree: Optional[int] = 2,
        visualize: Optional[bool] = True,
        arrows: Optional[bool] = False,
        save_plots: Optional[bool] = False) -> List[np.ndarray]:
    """
    Traces the path using the specified path parameters
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param magnetic_field_module_name: The file name (in the 'magnetic' package) that contains the
    required magnetic_field functions
    :param atmosphere_module_name: The file name (in the 'atmosphere' package) that contains the
    required atmosphere functions
    :param operating_frequency: The frequency of the ray
    :param atmosphere_parameters: The parameters defining the atmosphere
    :param max_newton_raphson_steps: Stop iterating if there are more total iterations than this
    :param minimum_change_per_position: If the average change between paths in one iteration is less than
    minimum_change_per_position, then stop iterating
    :param integration_step_number: The number of steps to use in integration
    :param derivative_step_size: The step size to use to calculate the finite step derivatives
    :param parameters: A tuple of (radial parameter count, normal parameter count)
    :param interpolated_degree: The degree of interpolation (2 for quadratic, 3 for cubic)
    :param visualize: Visualize the path as we are generating it.
    Primarily for debug reasons
    :param arrows: If true, plot arrows in visualization
    :param save_plots: If true, save the plot after creating it
    :return: The list of calculated paths
    """
    if parameters is not None:
        parameters = parameters
    elif parameters is None:
        parameters = constants.DEFAULT_TRACER_PARAMETERS

    calculated_paths = []
    calculated_paths_components = []

    load_dynamic_modules(atmosphere_module_name, magnetic_field_module_name)

    total_angle = vector.angle_between_vector_collections(
        coords.spherical_to_cartesian(start_point_spherical),
        coords.spherical_to_cartesian(end_point_spherical)
    ).item()

    qp_path_approximation = initialize.get_quasi_parabolic_path(
        total_angle * coords.EARTH_RADIUS, *atmosphere_parameters
    )
    heights_interpolation = interpolate.InterpolatedUnivariateSpline(
        qp_path_approximation[0],
        qp_path_approximation[1],
        k=interpolated_degree,
        ext=3
    )

    ground_distances = np.linspace(0, 1, parameters[0]) * total_angle * coords.EARTH_RADIUS
    initial_path_parameters = (heights_interpolation(ground_distances), np.zeros(parameters[1]))

    calculated_paths.append(initial_path_parameters)
    calculated_paths_components.append(path.generate_path_components(
        initial_path_parameters,
        degree=interpolated_degree
    ))

    if visualize:
        plotting.visualize_trace(calculated_paths_components, plot_all=True, show=True)

    # Create the pool to use in all derivative calculations
    pool = mp.Pool(
        mp.cpu_count() - 2,
        load_dynamic_modules,
        initargs=(atmosphere_module_name, magnetic_field_module_name)
    )

    # Iterate through the newton raphson steps
    for i in range(1, max_newton_raphson_steps):
        print(f"Preforming Newton Raphson Step {i}")
        incomplete_hessian, incomplete_gradient, change_vec = newton_raphson_step(
            calculated_paths[-1],
            start_point_spherical,
            end_point_spherical,
            operating_frequency,
            derivative_step_size=derivative_step_size,
            integration_step_number=integration_step_number,
            pool=pool
        )

        next_path = (
            calculated_paths[-1][0] + change_vec[:parameters[0]],
            calculated_paths[-1][0] + change_vec[parameters[0]:]
        )
        calculated_paths.append(next_path)
        calculated_paths_components.append(path.generate_path_components(
            next_path,
            degree=interpolated_degree
        ))

        if visualize:
            fig, ax = plotting.visualize_trace(calculated_paths_components, plot_all=True, show=False)
            params = np.concatenate(calculated_paths[-2])
            if arrows:
                for n, param in enumerate(params[::int(len(change_vec)/25)]):
                    # Plot change vec
                    x_c, dx_c = param[0]*coords.EARTH_RADIUS*total_angle/1000, 0
                    y_c = (param[1] - coords.EARTH_RADIUS)/1000 - 20
                    dy_c = -change_vec[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_c, y_c, dx_c, dy_c, color='black', width=3, head_width=12, head_length=12)
                    x_g = param[0]*coords.EARTH_RADIUS*total_angle/1000
                    dx_g = 0
                    y_g = (param[1] - coords.EARTH_RADIUS)/1000 + 20
                    dy_g = incomplete_gradient[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_g, y_g, dx_g, dy_g, color='white', width=3,  head_width=12, head_length=12)

            if save_plots:
                fig.savefig(os.path.join("SavedPlots", f'TotalChange_{i}.png'))
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

            plt.plot(incomplete_gradient)
            plt.suptitle("Gradient Graph")
            if save_plots:
                plt.savefig(os.path.join("SavedPlots", f'Gradient_{i}.png'))
                plt.close()
            else:
                plt.show()
                plt.close()

            cartesian_callable = path.generate_cartesian_callable(
                calculated_paths[-1],
                start_point_spherical,
                end_point_spherical
            )
            current_p = integrate_over_path(
                cartesian_callable,
                operating_frequency,
                show=True
            )

            print(f"Current total phase angle: {current_p}")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            image = ax.imshow(incomplete_hessian)
            color_bar = fig.colorbar(image, ax=ax)
            color_bar.set_label("Second Derivative")
            plt.suptitle("Matrix graph")

            if save_plots:
                fig.savefig(os.path.join("SavedPlots", f'Hessian Matrix_{i}.png'))
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

        if linalg.norm(change_vec) < minimum_change_per_position*np.sqrt(len(change_vec)):
            # Stop iterating if the change vec goes too small
            # (small means a change of less than minimum_change per position)
            print(f"Ending with final change vector of {linalg.norm(change_vec)}")
            break

    # Close the pool
    pool.close()

    return calculated_paths_components
