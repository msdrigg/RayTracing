import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, integrate, interpolate, optimize
import os
from utils import vector, plotting, path, constants, equations, coordinates as coords
import multiprocessing as mp
import typing
import math
import warnings


def integrate_over_path(
        cartesian_coordinate_callable: typing.Callable,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable,
        integration_step_number: int = 2 ** 12 + 1,
        show: bool = False,
        save: bool = False,
        debug_zero_field: bool = False):
    """
    Calculate P defined in Coleman as the integral of mu * pt over the path length
    See Section 3.1 for this definition.
    :param cartesian_coordinate_callable: A callable that takes values between 0 and 1 and
    returns the cartesian components of the path
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
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
    gyro_frequency_array = gyro_frequency_callable(r, r_norm)
    y2 = np.square(gyro_frequency_array / frequency)
    y_vec = magnetic_field_vec_callable(r, r_norm) * gyro_frequency_array / frequency
    x = np.square(plasma_frequency_callable(r, r_norm) / frequency)
    yt = vector.row_dot_product(y_vec, t)

    # solved_yp_old = np.zeros(step_number)
    max_fev = 200
    solved_yp = np.zeros(integration_step_number)

    # TODO: Disable zero field in this case
    for n in range(integration_step_number):
        if debug_zero_field:
            solved_yp[n] = 0
        else:
            # TODO: Profile different solvers here. Maybe call CPP function here
            function_args = x[n], y2[n], yt[n]
            y_norm = math.sqrt(y2)
            result, details = optimize.toms748(
                equations.equation_13, -y_norm, y_norm,
                function_args, xtol=1E-15, rtol=1E-15, full_output=True
            )
            if not details.converged:
                warnings.warn(
                    f"Error solving for yp using equation 13. Attempted {details.iterations} iterations "
                    f"and resulted in {details.root}, "
                    f"stopping with reason: {details.flag}."
                )
            solved_yp[n] = result

    solved_yp2 = np.square(solved_yp)
    # fractions = equation_16(solved_yp, solved_yp2, x, y2)
    # current_pt = equation_14(solved_yp, solved_yp2, y2, fractions, yt)
    current_pt = np.repeat(1, integration_step_number)
    solved_yp[current_pt < 0] = -solved_yp[current_pt < 0]
    current_pt = np.absolute(current_pt)
    current_mu2 = equations.equation_15(solved_yp, x, y2)
    rx = integrate.simps(r_dot[:, 0], dx=step_size)
    ry = integrate.simps(r_dot[:, 1], dx=step_size)
    rz = integrate.simps(r_dot[:, 2], dx=step_size)
    # r_estimate = np.zeros((step_number, 3))
    ip = initial_path.initial_point * initial_path.points[0, 1]
    fp = initial_path.final_point * initial_path.points[-1, 1]
    # r_estimate[0] = r[0]
    # for i in range(1, step_number):
    #     r_estimate[i, 0] = integrate.simps(r_dot[:i, 0], dx=h) + r[0, 0]
    #     r_estimate[i, 1] = integrate.simps(r_dot[:i, 1], dx=h) + r[0, 1]
    #     r_estimate[i, 2] = integrate.simps(r_dot[:i, 2], dx=h) + r[0, 2]
    # r_estimate[-1] = r[-1]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # u = np.np.linspace(0, 2 * np.pi, 100)
    # v = np.np.linspace(0, np.pi, 100)
    # x = coords.EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    # y = coords.EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    # z = coords.EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x, y, z, color='b', alpha=0.4)
    # ax.plot3D(r[:, 0], r[:, 1], r[:, 2], 'red')
    # ax.plot3D(r_estimate[:, 0], r_estimate[:, 1], r_estimate[:, 2], 'green')
    # plt.show()
    # plt.plot(r_dot[:, 0], color='blue')
    # plt.plot(r_dot[:, 1], color='red')
    # plt.plot(r_dot[:, 2], color='green')
    # plt.show()
    # print(f"Total r-dot: {array([rx, ry, rz])}.")
    # print(f"Total change: {fp - ip}")
    dp_array = np.sqrt(current_mu2) * current_pt * r_dot_norm
    integration = integrate.romb(dp_array, 1 / integration_step_number)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.plot(dp_array)
        if save is not None:
            fig.savefig(os.path.join("SavedPlots", f'TotalPValues_{save}.png'))
        else:
            plt.show()
        plt.close(fig)
    return integration


def calculate_off_diagonal_dirs(
        index_pair: typing.Tuple[int],
        curr_path: np.ndarray,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        integration_step_number: int,
        gradient_step_size: float,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable):
    """
    Calculates the off-diagonal derivatives for calculating the hessian of P
    These calculations appear in Equation 19 of Coleman 2011
    :param index_pair: The second derivative is calculated with respect to this pair of parameters
    :param curr_path: The current state of the path (The evaluation point of the derivative)
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param integration_step_number: The integration step size
    :param gradient_step_size: The amount to vary the parameter in the calculation of the derivative
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
    :return: The mixed second derivative of the functional P with respect to the parameters in index_pair
    """

    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.
    if abs(index_pair[0] - index_pair[1]) > 9:
        return np.array([*index_pair, 0])

    path_mm_array = curr_path.copy()
    path_mm_array[index_pair] -= gradient_step_size
    path_mm_callable = path.generate_cartesian_callable(path_mm_array, start_point_spherical, end_point_spherical)
    p_mm = integrate_over_path(
        path_mm_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    path_mp_array = curr_path.copy()
    path_mp_array[index_pair] += [-gradient_step_size, gradient_step_size]
    path_mp_callable = path.generate_cartesian_callable(path_mp_array, start_point_spherical, end_point_spherical)
    p_mp = integrate_over_path(
        path_mp_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    path_pm_array = curr_path.copy()
    path_pm_array[index_pair] += [gradient_step_size, -gradient_step_size]
    path_pm_callable = path.generate_cartesian_callable(path_pm_array, start_point_spherical, end_point_spherical)
    p_pm = integrate_over_path(
        path_pm_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    path_pp_array = curr_path.copy()
    path_pp_array[index_pair] += gradient_step_size
    path_pp_callable = path.generate_cartesian_callable(path_pp_array, start_point_spherical, end_point_spherical)
    p_pp = integrate_over_path(
        path_pp_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * gradient_step_size ** 2)

    return np.array([*index_pair, output])


def calculate_diagonal_derivatives(
        varied_parameter: int,
        curr_path: np.ndarray,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        integration_step_number: int,
        gradient_step_size: float,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable):
    """
    Calculate the second derivative and the gradient vector.
    These calculations involve the same function calls
    The gradient is an implementation of Equation 18 in Coleman 2011, and the second derivative
    appears in Equation 19 of the same paper
    :param varied_parameter: The gradient is calculated with respect to this parameter
    :param curr_path: The current state of the path (The evaluation point of the derivative)
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param integration_step_number: The integration step size
    :param gradient_step_size: The amount to vary the parameter in the calculation of the derivative
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
    :return: The first and second derivative of the functional P with respect to the parameter
    """
    path_minus = curr_path.copy()
    path_minus[varied_parameter] -= gradient_step_size
    path_minus_callable = path.generate_cartesian_callable(path_minus, start_point_spherical, end_point_spherical)

    path_plus = curr_path.copy()
    path_plus[varied_parameter] += gradient_step_size
    path_plus_callable = path.generate_cartesian_callable(path_plus, start_point_spherical, end_point_spherical)

    path_current_callable = path.generate_cartesian_callable(curr_path, start_point_spherical, end_point_spherical)

    p_minus = integrate_over_path(
        path_minus_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    p_plus = integrate_over_path(
        path_plus_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )
    p_0 = integrate_over_path(
        path_current_callable,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        integration_step_number=integration_step_number
    )

    dpdx = (p_plus - p_minus) / (2 * gradient_step_size)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (gradient_step_size ** 2)
    return np.array([varied_parameter, dpdx, d2pdx2])


def calculate_derivatives(
        current_path_parameters: np.ndarray,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable,
        variable_indices: np.ndarray = np.array([]),
        derivative_step_size: float = 1000,
        integration_step_number: int = None,
        pool: mp.Pool = None):
    """
    Perform a newton raphson step as defined in Equation 19 of Coleman 2011. Thi
    See Section 3.1 for this definition.
    :param current_path_parameters: The current path parameters
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
    :param variable_indices: Array of indices for which we should NOT calculate derivatives
    :param derivative_step_size: The step size in meters over which to calculate the gradient
    :param integration_step_number: The number of points to use in integrating over the path.
    We use romberg integration, so we must have integration_point_count be of the form 2 ^ n + 1
    :param pool: The processor pool to calculate the derivatives with. If none, create one
    :return: A tuple of (The hessian due to the The next path after the iteration)
    """
    # We need to make sure our integration step size is significantly smaller than our derivative
    #   or else our truncation error will be too large

    total_ground_distance = math.asin(linalg.norm(np.cross(
        coords.spherical_to_cartesian(start_point_spherical),
        coords.spherical_to_cartesian(end_point_spherical)
    )))

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

    parameter_number = variable_indices.size

    # dP/da_i
    gradient = np.zeros(parameter_number)

    # d^2P/(da_ida_j)
    matrix = np.zeros((parameter_number, parameter_number))

    # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
    def pair_generator(item_number, param_number):
        counter = 0
        for pair_gen_row in range(0, param_number - 1):
            for pair_gen_col in range(1 + pair_gen_row, param_number):
                if counter == item_number:
                    return [pair_gen_row, pair_gen_col]
                counter += 1
        raise IndexError("You are indexing for a pair that doesn't exist")

    # Create the pool if one was not passed in
    if pool is None:
        usable_pool = mp.Pool(mp.cpu_count() - 1)
    else:
        usable_pool = pool

    total_ints = 3*parameter_number
    print("Calculating Derivatives for the diagonal elements")
    print(f"Expecting {total_ints} diagonal integrations")

    # Parallelize calculation of directional diagonal derivatives
    diagonal_d_results = usable_pool.starmap(
        calculate_diagonal_derivatives,
        zip(
            variable_indices,
            [start_point_spherical] * parameter_number,
            [end_point_spherical] * parameter_number,
            [current_path_parameters] * parameter_number,
            [integration_step_number] * parameter_number,
            [derivative_step_size] * parameter_number,
            [gyro_frequency_callable] * parameter_number,
            [magnetic_field_vec_callable] * parameter_number,
            [plasma_frequency_callable] * parameter_number
        )
    )

    for result in diagonal_d_results:
        index = int(result[0])
        gradient[index] = result[1]
        matrix[index, index] = result[2]

    print("Completed Diagonal Integrations")
    print("Calculating Derivatives for the off diagonal elements")
    elements = int(parameter_number*(parameter_number-1)/2)
    print(f"Expecting {elements*4} integrations")

    # Parallelize calculation of directional off diagonal derivatives
    off_diagonal_d_results = usable_pool.starmap(
        calculate_off_diagonal_dirs,
        zip(
            [
                variable_indices[pair_generator(n, parameter_number)]
                for n in range(elements)
            ],
            [start_point_spherical] * elements,
            [end_point_spherical] * elements,
            [current_path_parameters] * elements,
            [integration_step_number] * elements,
            [derivative_step_size] * elements,
            [gyro_frequency_callable] * elements,
            [magnetic_field_vec_callable] * elements,
            [plasma_frequency_callable] * elements
        )
    )

    if pool is None:
        # Close the pool if one was not passed in
        print("Closing Pool")
        usable_pool.close()

    for result in off_diagonal_d_results:
        row, col = int(result[0]), int(result[1])
        matrix[row, col] = result[2]
        matrix[col, row] = result[2]

    print("Completed off diagonal integrations")
    return matrix, gradient


def newton_raphson_step(
        current_path_parameters: np.ndarray,
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable,
        variable_indices: np.ndarray = np.array([1, -1]),
        **kwargs):
    """
    Perform a newton raphson step as defined in Equation 19 of Coleman 2011. Thi
    See Section 3.1 for this definition.
    :param current_path_parameters: The current path parameters (alpha_i) before the newton raphson step
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
    :param variable_indices: Array of indices for which we should NOT change or calculate change
    (usually first and last)
    :param kwargs: The rest of the arguments are passed to the derivative calculation
    :return: A tuple of (The hessian due to the The next path after the iteration)
    """
    incomplete_hessian, incomplete_gradient = calculate_derivatives(
        current_path_parameters,
        start_point_spherical,
        end_point_spherical,
        gyro_frequency_callable,
        magnetic_field_vec_callable,
        plasma_frequency_callable,
        variable_indices=variable_indices,
        **kwargs
    )

    inverse_matrix = linalg.pinvh(incomplete_hessian)
    incomplete_change = np.matmul(inverse_matrix, incomplete_gradient)

    change = np.zeros_like(current_path_parameters)
    change[variable_indices] = incomplete_change

    change_mag = linalg.norm(change)

    print(f"Change magnitude: {change_mag}")

    return incomplete_hessian, incomplete_gradient, change


def trace(
        start_point_spherical: np.ndarray,
        end_point_spherical: np.ndarray,
        gyro_frequency_callable: typing.Callable,
        magnetic_field_vec_callable: typing.Callable,
        plasma_frequency_callable: typing.Callable,
        max_newton_raphson_steps: int = 50,
        minimum_change_per_position: int = 50,
        integration_step_number: int = None,
        derivative_step_size: float = None,
        parameters: typing.Tuple = None,
        visualize: bool = True,
        arrows: bool = False,
        save_plots: bool = False):
    """
    Traces the path using the specified path parameters
    :param start_point_spherical: The path start point in spherical coordinates
    :param end_point_spherical: The path end point in spherical coordinates
    :param gyro_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the gyro frequency
    :param magnetic_field_vec_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the magnetic field vec
    :param plasma_frequency_callable: A callable that takes inputs of cartesian coordinates and their norms
    and returns the plasma frequency
    :param max_newton_raphson_steps: Stop iterating if there are more total iterations than this
    :param minimum_change_per_position: If the average change between paths in one iteration is less than
    minimum_change_per_position, then stop iterating
    :param integration_step_number: The number of steps to use in integration
    :param derivative_step_size: The step size to use to calculate the finite step derivatives
    :param parameters: A tuple of (radial parameter count, normal parameter count)
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
        parameter_number = constants.DEFAULT_TRACER_PARAMETERS[0] + constants.DEFAULT_TRACER_PARAMETERS[1]

    calculated_paths = []
    # TODO: Create the first calculated_path with parameter_number[0]
    #  radial parameters and parameter_number[1] normal ones

    if visualize:
        plotting.visualize_trace(calculated_paths, plot_all=True, show=True)

    # Create the pool to use in all derivative calculations
    pool = mp.Pool(mp.cpu_count() - 2)

    # Iterate through the newton raphson steps
    for i in range(1, max_newton_raphson_steps):
        print(f"Preforming Newton Raphson Step {i}")
        incomplete_hessian, incomplete_gradient, change_vec = newton_raphson_step(
            calculated_paths[-1],
            start_point_spherical,
            end_point_spherical,
            gyro_frequency_callable,
            magnetic_field_vec_callable,
            plasma_frequency_callable,
            derivative_step_size=derivative_step_size,
            integration_step_number=integration_step_number,
            pool=pool
        )

        if visualize:
            fig, ax = plotting.visualize_trace(calculated_paths, plot_all=True, show=False)
            params = calculated_paths[-2].parameters
            total_angle = calculated_paths[-2].total_angle
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
                gyro_frequency_callable,
                magnetic_field_vec_callable,
                plasma_frequency_callable,
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

    return calculated_paths


if __name__ == "__main__":
    field = BasicField()
    initial = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    atmosphere = ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / math.pi, -1), initial)
    path_generator = QuasiParabolic
    frequency = 10E6  # Hz
    atmosphere.visualize(initial, final, ax=None, fig=None, point_number=400, show=True)
    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.parameters = (50, 0)
    basic_tracer.parameter_number = 50
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = initial, final

    basic_tracer.compile_initial_path()
    # basic_tracer.visualize(plot_all=True)

    paths = basic_tracer.trace()
