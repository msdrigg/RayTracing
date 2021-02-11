import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, integrate, interpolate
import os
from utils import vector, plotting, path, constants, equations, coordinates as coords
import multiprocessing as mp


def off_diagonal_dirs(index_pair, curr_path, int_h, vary_h):
    """
    Calculates the off-diagonal derivatives for calculating the hessian
    """
    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.

    if abs(index_pair[0] - index_pair[1]) > 9:
        return np.array([index_pair[0], index_pair[1], 0])

    path_mm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], -vary_h)
    p_mm = integrate_parameter(path_mm, step_size=int_h)
    path_mp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [-vary_h, vary_h])
    p_mp = integrate_parameter(path_mp, step_size=int_h)
    path_pm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [vary_h, -vary_h])
    p_pm = integrate_parameter(path_pm, step_size=int_h)
    path_pp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], vary_h)
    p_pp = integrate_parameter(path_pp, step_size=int_h)
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * vary_h ** 2)

    # total = (index_pair[0] + 1) * (parameter_number - 1) - int((index_pair[0] + 1) * (index_pair[0]) / 2)
    # if total % 500 == 0:
    #     print(f"Completed integration {total*4} of "
    #           f"{int(parameter_number*(parameter_number-1)/2) * 4} (asynchronous)")

    return np.array([index_pair[0], index_pair[1], output])


# Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
def diagonal_dirs(varied_parameter, curr_path, int_h, vary_h):
    path_minus = curr_path.adjust_parameters(varied_parameter, -vary_h)
    path_plus = curr_path.adjust_parameters(varied_parameter, vary_h)
    p_minus = integrate_parameter(path_minus, step_size=int_h)
    p_plus = integrate_parameter(path_plus, step_size=int_h)
    p_0 = integrate_parameter(curr_path, step_size=int_h)
    dpdx = (p_plus - p_minus) / (2 * vary_h)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (vary_h ** 2)
    return np.array([varied_parameter, dpdx, d2pdx2])


def integrate_parameter(
        path: interpolate.InterpolatedUnivariateSpline,
        step_size: float=0.00001,
        show: bool=False,
        save: bool=False):
    step_number = int(1 / step_size)
    r = path(np.linspace(0, 1, step_number), nu=0)
    r_dot = path(np.linspace(0, 1, step_number), nu=1)
    t = r_dot / linalg.norm(r_dot)
    y2 = np.square(field.gyro_frequency(r) / frequency)
    y_vec = field.field_vec(r) * field.gyro_frequency(r).reshape(-1, 1) / frequency
    x = np.square(atmosphere.plasma_frequency(r) / frequency)
    yt = vector.row_dot_product(y_vec, t)

    estimated_yp = yt.copy()
    # solved_yp_old = np.zeros(step_number)
    max_fev = 200
    solved_yp = np.zeros(step_number)
    for n in range(step_number):
        args_fsolve = x[n], y2[n], yt[n]
        # ret_val = hybrj(equation_13_new, equation_13_prime_new, asarray(estimated_yp[n]).flatten(), args_fsolve, 1,
        # True, 1.50E-8, max_fev, 100, None)
        # if ret_val[-1] != 1:
        #     print(f"Status: {errors.get(ret_val[-1])}")
        # print(f"{ret_val[1].get('nfev')} calls")
        # solved_yp[n] = ret_val[0]
        solved_yp[n] = 0

    solved_yp2 = np.square(solved_yp)
    # fractions = equation_16(solved_yp, solved_yp2, x, y2)
    # current_pt = equation_14(solved_yp, solved_yp2, y2, fractions, yt)
    current_pt = np.repeat(1, step_number)
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
    dp_array = np.sqrt(current_mu2) * current_pt * linalg.linalg.norm(r_dot, axis=1)
    integration = integrate.simps(dp_array, dx=step_size)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.plot(dp_array)
        if save is not None:
            fig.savefig(os.path.join("SavedPlots", f'TotalPValues_{save}.png'))
        else:
            plt.show()
        plt.close(fig)
    return integration


def newton_raphson_step(step_size=1000):
    matrix, gradient = calculate_derivatives(h=step_size)
    change = linalg.solve(matrix, gradient, assume_a='sym')
    change_mag = linalg.norm(change)
    print(f"Change magnitude: {change_mag}")

    next_params = calculated_paths[-1].parameters[:, 1] - change
    next_path = GreatCircleDeviation(*parameters, initial_parameters=next_params,
                                     initial_coordinate=initial_coordinates,
                                     final_coordinate=final_coordinates, using_spherical=False)
    calculated_paths.append(next_path)
    return matrix, gradient, change

def calculate_derivatives(h=5000):
    # We need to make sure our integration step size is significantly smaller than our derivative
    #   or else our truncation error will be too large
    integration_step = 1/500.0

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

    pool = mp.Pool(mp.cpu_count() - 2)
    total_ints = 3*parameter_number
    print("Calculating Derivatives for the diagonal elements")
    print(f"Expecting {total_ints} diagonal integrations")
    # Parallelize calculation of directional diagonal derivatives
    diagonal_d_results = pool.map(diagonal_dirs,
                                  zip([for _ in range(parameter_number)],
                                      list(range(parameter_number)),
                                      [calculated_paths[-1] for _ in range(parameter_number)],
                                      [integration_step for _ in range(parameter_number)],
                                      [h for _ in range(parameter_number)]))

    for result in diagonal_d_results:
        index = int(result[0])
        gradient[index] = result[1]
        matrix[index, index] = result[2]

    print("Completed Diagonal Integrations")
    print("Calculating Derivatives for the off diagonal elements")
    elements = int(parameter_number*(parameter_number-1)/2)
    print(f"Expecting {elements*4} integrations")

    # Parallelize calculation of directional off diagonal derivatives
    off_diagonal_d_results = pool.map(off_diagonal_dirs, zip([for _ in range(elements)],
                                                             [pair_generator(n, parameter_number)
                                                              for n in range(elements)],
                                                             [calculated_paths[-1] for _ in range(elements)],
                                                             [integration_step for _ in range(elements)],
                                                             [h for _ in range(elements)]))
    print("Closing Pool")
    pool.close()
    for result in off_diagonal_d_results:
        row, col = int(result[0]), int(result[1])
        matrix[row, col] = result[2]
        matrix[col, row] = result[2]
    print("Completed off diagonal integrations")
    return matrix, gradient


def trace(initial_point, final_point,
          steps=50, step_size=1000,
          parameters=None,
          visualize=True,
          arrows=False,
          save_plots=False):
    if parameters is not None:
        parameters = parameters
    elif parameters is None:
        parameters = constants.DEFAULT_TRACER_PARAMETERS
        parameter_number = constants.DEFAULT_TRACER_PARAMETERS[0] + constants.DEFAULT_TRACER_PARAMETERS[1]

    calculated_paths = ()
    radial_components,

    if visualize:
        plotting.visualize_trace(calculated_paths, plot_all=True, show=True)

    for i in range(1, steps):
        print(f"Preforming Newton Raphson Step {i}")
        matrix, gradient, change_vec = newton_raphson_step(step_size=step_size)

        if visualize:
            fig, ax = plotting.visualize_trace(calculated_paths, plot_all=True, show=False)
            params = calculated_paths[-2].parameters
            total_angle = calculated_paths[-2].total_angle
            if arrows:
                for n, param in enumerate(params[::int(len(change_vec)/25)]):
                    # Plot change vec
                    x_c, dx_c = param[0]*coords.EARTH_RADIUS*total_angle/1000, 0
                    y_c, dy_c = (param[1] - coords.EARTH_RADIUS)/1000 - 20, -change_vec[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_c, y_c, dx_c, dy_c, color='black', width=3, head_width=12, head_length=12)
                    x_g, dx_g = param[0]*coords.EARTH_RADIUS*total_angle/1000, 0
                    y_g, dy_g = (param[1] - coords.EARTH_RADIUS)/1000 + 20, gradient[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_g, y_g, dx_g, dy_g, color='white', width=3,  head_width=12, head_length=12)

            if save_plots:
                fig.savefig(os.path.join("SavedPlots", f'TotalChange_{i}.png'))
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

            plt.plot(gradient)
            plt.suptitle("Gradient Graph")
            if save_plots:
                plt.savefig(os.path.join("SavedPlots", f'Gradient_{i}.png'))
                plt.close()
            else:
                plt.show()
                plt.close()

            current_p = integrate_parameter(calculated_paths[-1], step_size=0.00001, show=True, save=i)
            print(f"Current total phase angle: {current_p}")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            image = ax.imshow(matrix)
            color_bar = fig.colorbar(image, ax=ax)
            color_bar.set_label("Second Derivative")
            plt.suptitle("Matrix graph")

            if save_plots:
                fig.savefig(os.path.join("SavedPlots", f'Hessian Matrix_{i}.png'))
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

        if linalg.norm(change_vec) < 500*np.sqrt(len(change_vec)):
            # Break if the change vec goes too small (small means a change of less than 500 m per position)
            print(f"Ending with final change vector of {linalg.norm(change_vec)}")
            break

    return calculated_paths


if __name__ == "__main__":
    field = BasicField()
    initial = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    atmosphere = ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / PI, -1), initial)
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
