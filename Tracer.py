from numpy import zeros, array, sqrt, linspace, \
    square, absolute, savetxt, repeat
from numpy import sum as vector_sum
from numpy.linalg import norm
from Atmosphere import ChapmanLayers
from Field import BasicField
from Paths import QuasiParabolic, GreatCircleDeviation
from scipy.linalg import solve as sym_solve
from scipy.integrate import simps
# from scipy.optimize._minpack import _hybrj as hybrj
import Vector
from matplotlib import use as matplotlib_use
matplotlib_use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from Constants import PI, EARTH_RADIUS
import multiprocessing as mp
# import numpy as np
from os.path import join as join_path
# from mpl_toolkits.mplot3d import Axes3D
# import os


DEFAULT_PARAMS = 50, 5

errors = {0: "Improper input parameters were entered.",
          1: "The solution converged.",
          2: "The number of calls to function has "
             "reached max_fev = %d." % 200,
          3: "x_tol=%f is too small, no further improvement "
             "in the approximate\n  solution "
             "is possible." % 1E-7,
          4: "The iteration is not making good progress, as measured "
             "by the \n  improvement from the last five "
             "Jacobian evaluations.",
          5: "The iteration is not making good progress, "
             "as measured by the \n  improvement from the last "
             "ten iterations.",
          'unknown': "An error occurred."}


# I wanted to make these static methods of the Tracer class but I think this is faster,
#   and we really need the speed
def equation_13_new(yp, *args):
    yp2 = square(yp)
    x, y2, yt = args

    fractions = equation_16(yp, yp2, x, y2)
    pt = equation_14(yp, yp2, y2, fractions, yt)

    output = -1 + pt * (pt - yt * (1 - pt) * fractions * 0.5)
    return output


# def equation_13_prime(yp, *args):
#     x, y2, yt = args
#     yp2 = square(yp)
#     radical = sqrt(square(y2 - yp2) + 4*yp2*square(1 - x))
#     a = 1 - x - y2 + x*yp2
#
#     # Choosing ordinary ray
#     alpha = 2*(1-x)/((2 + yp2 + radical) - (2*x + y2))
#     d_alpha = -4*(1-x)*yp*(-1 + (-(4*x + y2) +
#                                  (yp2 + 2 + 2*square(x)))/radical)/square(-(2*x + y2) + (2 + yp2 + radical))
#
#     frac = 2*x*yp*alpha/(-(2 + yp2) + (y2 + 2*a*alpha + 2*x))
#     d_frac = 2*x*yp*d_alpha*(-2 + 2*x + y2 - yp2)/square(-(2*x + y2 + 2*a*alpha) + (2 + yp2))
#
#     pt = yt/(yp - 0.5*frac*(y2 - yp2))
#     d_pt = -2*yt*(2 + 2*yp*frac - (y2 - yp2)*d_frac)/square(2*yp - (y2 - yp2)*frac)
#
#     output = 0.5*(pt*(frac*pt + (-yt + yp*pt)*d_frac) + (-yt*frac + 2*(2 + yp*frac)*pt)*d_pt)
#     # print("Outputting Derivatives")
#     # print(output)
#     return output


def equation_13_prime_new(yp, *args):
    x, y2, yt = args
    yp2 = square(yp)
    radical = sqrt(square(y2 - yp2) + 4*yp2*square(1 - x))
    a = 1 - x - y2 + x*yp2

    # Choosing ordinary ray
    alpha = 2*(1-x)/((2 + yp2 + radical) - (2*x + y2))
    d_alpha = -4*(1-x)*yp*(-1 + (-(4*x + y2) +
                                 (yp2 + 2 + 2*square(x)))/radical)/square(-(2*x + y2) + (2 + yp2 + radical))

    frac = 2*x*yp*alpha/(-(2 + yp2) + (y2 + 2*a*alpha + 2*x))
    d_frac = 2*x*yp*d_alpha*(-2 + 2*x + y2 - yp2)/square(-(2*x + y2 + 2*a*alpha) + (2 + yp2))

    pt = yt/(yp - 0.5*frac*(y2 - yp2))
    d_pt = -2*yt*(2 + 2*yp*frac - (y2 - yp2)*d_frac)/square(2*yp - (y2 - yp2)*frac)

    output = 0.5*(pt*(frac*pt + (-yt + yp*pt)*d_frac) + (-yt*frac + 2*(2 + yp*frac)*pt)*d_pt)
    # print("Outputting Derivatives")
    # print(output)
    return output


def equation_14(yp, yp2, y2, fractions, yt):
    return yt / (yp - (y2 - yp2) * fractions * 0.5)


def equation_15(yp, x, y2):
    # We choose ordinary ray in our calculation of mu2
    yp2 = square(yp)
    return 1 - 2 * x * (1 - x) / (2 * (1 - x) - (y2 - yp2) +
                                  sqrt(square(y2 - yp2) + 4 * square(1 - x) * yp2))


def equation_16(yp, yp2, x, y2):
    a = 1 - x - y2 + x * yp2
    beta = 2 * (1 - x) / (2 * (1 - x) - (y2 - yp2) +
                          sqrt(square((y2 - yp2)) + 4 * square(1 - x) * yp2))
    # print(f"a, beta: {a}, {beta}")
    return x * yp * beta / (1 + 0.5 * (yp2 - y2) - a*beta - x)


def off_diagonal_dirs(inputs):
    self, index_pair, curr_path, int_h, vary_h = inputs

    # We know for a cubic spline, the derivative wrt parameter i within the integral only affects the spline
    # in the interval (i - 2, i + 2) and for a quartic (i - 3, i + 3), so we consider
    # two derivatives cannot affect each other if these intervals do not overlap.

    if abs(index_pair[0] - index_pair[1]) > 9:
        return array([index_pair[0], index_pair[1], 0])

    path_mm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], -vary_h)
    p_mm = self.integrate_parameter(path_mm, h=int_h)
    path_mp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [-vary_h, vary_h])
    p_mp = self.integrate_parameter(path_mp, h=int_h)
    path_pm = curr_path.adjust_parameters([index_pair[0], index_pair[1]], [vary_h, -vary_h])
    p_pm = self.integrate_parameter(path_pm, h=int_h)
    path_pp = curr_path.adjust_parameters([index_pair[0], index_pair[1]], vary_h)
    p_pp = self.integrate_parameter(path_pp, h=int_h)
    output = (p_pp - p_pm - p_mp + p_mm) / (4 * vary_h ** 2)

    # total = (index_pair[0] + 1) * (self.parameter_number - 1) - int((index_pair[0] + 1) * (index_pair[0]) / 2)
    # if total % 500 == 0:
    #     print(f"Completed integration {total*4} of "
    #           f"{int(self.parameter_number*(self.parameter_number-1)/2) * 4} (asynchronous)")

    return array([index_pair[0], index_pair[1], output])


# Calculate the diagonal elements and the gradient vector. These calculations involve the same function calls
def diagonal_dirs(inputs):
    self, varied_parameter, curr_path, int_h, vary_h = inputs
    path_minus = curr_path.adjust_parameters(varied_parameter, -vary_h)
    path_plus = curr_path.adjust_parameters(varied_parameter, vary_h)
    p_minus = self.integrate_parameter(path_minus, h=int_h)
    p_plus = self.integrate_parameter(path_plus, h=int_h)
    p_0 = self.integrate_parameter(curr_path, h=int_h)
    dpdx = (p_plus - p_minus) / (2 * vary_h)
    d2pdx2 = (p_plus + p_minus - 2 * p_0) / (vary_h ** 2)
    return array([varied_parameter, dpdx, d2pdx2])


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
        new_path = GreatCircleDeviation(*self.parameters, quasi_parabolic=self.initial_path)
        new_path.interpolate_params()

        if self.calculated_paths is None:
            self.calculated_paths = [new_path]
        else:
            self.calculated_paths.append(new_path)

    def trace(self, steps=50, h=1000, parameters=None, visualize=True):
        if visualize == 'save':
            save_plots = True
        else:
            save_plots = False
        if parameters is not None:
            self.parameters = parameters
        elif self.parameters is None:
            self.parameters = DEFAULT_PARAMS
            self.parameter_number = DEFAULT_PARAMS[0] + DEFAULT_PARAMS[1]

        if self.calculated_paths is None:
            self.compile_initial_path()
        if visualize:
            self.visualize(plot_all=True)

        for i in range(1, steps):
            print(f"Preforming Newton Raphson Step {i}")
            matrix, gradient, change_vec = self.newton_raphson_step(h=h)

            if visualize:
                fig, ax = self.visualize(plot_all=True, show=False)
                params = self.calculated_paths[-2].parameters
                total_angle = self.calculated_paths[-2].total_angle
                for n, param in enumerate(params[::int(len(change_vec)/25)]):
                    # Plot change vec
                    x_c, dx_c = param[0]*EARTH_RADIUS*total_angle/1000, 0
                    y_c, dy_c = (param[1] - EARTH_RADIUS)/1000 - 20, -change_vec[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_c, y_c, dx_c, dy_c, color='black', width=3, head_width=12, head_length=12)
                    x_g, dx_g = param[0]*EARTH_RADIUS*total_angle/1000, 0
                    y_g, dy_g = (param[1] - EARTH_RADIUS)/1000 + 20, gradient[n*int(len(change_vec)/25)]/1000
                    ax.arrow(x_g, y_g, dx_g, dy_g, color='white', width=3,  head_width=12, head_length=12)

                if save_plots:
                    fig.savefig(join_path("SavedPlots", f'TotalChange_{i}.png'))
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)

                plt.plot(gradient)
                plt.suptitle("Gradient Graph")
                if save_plots:
                    plt.savefig(join_path("SavedPlots", f'Gradient_{i}.png'))
                    plt.close()
                else:
                    plt.show()
                    plt.close()

                current_p = self.integrate_parameter(self.calculated_paths[-1], h=0.00001, show=True, save=i)
                print(f"Current total phase angle: {current_p}")
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                image = ax.imshow(matrix)
                color_bar = fig.colorbar(image, ax=ax)
                color_bar.set_label("Second Derivative")
                plt.suptitle("Matrix graph")

                if save_plots:
                    fig.savefig(join_path("SavedPlots", f'Hessian Matrix_{i}.png'))
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)

            if norm(change_vec) < 500*sqrt(len(change_vec)):
                # Break if the change vec goes too small (small means a change of less than 500 m per position)
                print(f"Ending with final change vector of {norm(change_vec)}")
                break

        return self.calculated_paths

    def newton_raphson_step(self, h=1000):
        matrix, gradient = self.calculate_derivatives(h=h)
        # print(f"Matrix shape: {matrix.shape}")
        # print(f"Gradient shape: {gradient.shape}")
        # eof_string = '_'.join(map(str, self.parameters))
        # eof_string += f'_{int(self.frequency/1000)}'
        # try:
        #     savetxt(join_path("SavedData", f"Matrix{eof_string}.txt"), matrix)
        #     savetxt(join_path("SavedData", f"Gradient{eof_string}.txt"), gradient)
        #     savetxt(join_path("SavedData", f"CurrentParams{eof_string}.txt"),
        #             self.calculated_paths[-1].parameters[:, 1])
        # except FileNotFoundError:
        #     savetxt(f"Matrix{eof_string}.txt", matrix)
        #     savetxt(f"Gradient{eof_string}.txt", gradient)
        #     savetxt(f"CurrentParams{eof_string}.txt", self.calculated_paths[-1].parameters[:, 1])
        # # Calculate diagonal matrix elements

        change = sym_solve(matrix, gradient, assume_a='sym')
        change_mag = norm(change)
        print(f"Change magnitude: {change_mag}")

        next_params = self.calculated_paths[-1].parameters[:, 1] - change
        next_path = GreatCircleDeviation(*self.parameters, initial_parameters=next_params,
                                         initial_coordinate=self.initial_coordinates,
                                         final_coordinate=self.final_coordinates, using_spherical=False)
        self.calculated_paths.append(next_path)
        return matrix, gradient, change

    def calculate_derivatives(self, h=5000):
        # We need to make sure our integration step size is significantly smaller than our derivative
        #   or else our truncation error will be too large
        integration_step = 1/500.0

        # dP/da_i
        gradient = zeros((self.parameter_number,))

        # d^2P/(da_ida_j)
        matrix = zeros((self.parameter_number, self.parameter_number))

        # Calculate the off-diagonal elements (Only calculate uppers and set the lower equal to upper)
        def pair_generator(item_number, param_number=self.parameter_number):
            counter = 0
            for row in range(0, param_number - 1):
                for col in range(1 + row, param_number):
                    if counter == item_number:
                        return [row, col]
                    counter += 1
            raise IndexError("You are indexing for a pair that doesn't exist")

        pool = mp.Pool(mp.cpu_count() - 2)
        total_ints = 3*self.parameter_number
        print("Calculating Derivatives for the diagonal elements")
        print(f"Expecting {total_ints} diagonal integrations")
        # Parallelize calculation of directional diagonal derivatives
        diagonal_d_results = pool.map(diagonal_dirs,
                                      zip([self for _ in range(self.parameter_number)],
                                          list(range(self.parameter_number)),
                                          [self.calculated_paths[-1] for _ in range(self.parameter_number)],
                                          [integration_step for _ in range(self.parameter_number)],
                                          [h for _ in range(self.parameter_number)]))

        for result in diagonal_d_results:
            index = int(result[0])
            gradient[index] = result[1]
            matrix[index, index] = result[2]

        print("Completed Diagonal Integrations")
        print("Calculating Derivatives for the off diagonal elements")
        elements = int(self.parameter_number*(self.parameter_number-1)/2)
        print(f"Expecting {elements*4} integrations")

        # Parallelize calculation of directional off diagonal derivatives
        off_diagonal_d_results = pool.map(off_diagonal_dirs, zip([self for _ in range(elements)],
                                                                 [pair_generator(n, self.parameter_number)
                                                                  for n in range(elements)],
                                                                 [self.calculated_paths[-1] for _ in range(elements)],
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

    def integrate_parameter(self, path, h=0.00001, show=False, save=False):
        step_number = int(1/h)
        r = path(linspace(0, 1, step_number), nu=0)
        r_dot = path(linspace(0, 1, step_number), nu=1)
        t = Vector.unit_vector(r_dot)
        y2 = square(self.field.gyro_frequency(r) / self.frequency)
        y_vec = self.field.field_vec(r)*self.field.gyro_frequency(r).reshape(-1, 1) / self.frequency
        x = square(self.atmosphere.plasma_frequency(r)/self.frequency)
        yt = vector_sum(y_vec*t, axis=1)

        estimated_yp = yt.copy()
        # solved_yp_old = zeros(step_number)
        max_fev = 200
        solved_yp = zeros(step_number)
        for n in range(step_number):
            args_fsolve = x[n], y2[n], yt[n]
            # ret_val = hybrj(equation_13_new, equation_13_prime_new, asarray(estimated_yp[n]).flatten(), args_fsolve, 1,
                            # True, 1.50E-8, max_fev, 100, None)
            # if ret_val[-1] != 1:
            #     print(f"Status: {errors.get(ret_val[-1])}")
                # print(f"{ret_val[1].get('nfev')} calls")
            # solved_yp[n] = ret_val[0]
            solved_yp[n] = 0

        solved_yp2 = square(solved_yp)
        # fractions = equation_16(solved_yp, solved_yp2, x, y2)
        # current_pt = equation_14(solved_yp, solved_yp2, y2, fractions, yt)
        current_pt = repeat(1, step_number)
        solved_yp[current_pt < 0] = -solved_yp[current_pt < 0]
        current_pt = absolute(current_pt)
        current_mu2 = equation_15(solved_yp, x, y2)
        rx = simps(r_dot[:, 0], dx=h)
        ry = simps(r_dot[:, 1], dx=h)
        rz = simps(r_dot[:, 2], dx=h)
        # r_estimate = zeros((step_number, 3))
        ip = self.initial_path.initial_point*self.initial_path.points[0, 1]
        fp = self.initial_path.final_point*self.initial_path.points[-1, 1]
        # r_estimate[0] = r[0]
        # for i in range(1, step_number):
        #     r_estimate[i, 0] = simps(r_dot[:i, 0], dx=h) + r[0, 0]
        #     r_estimate[i, 1] = simps(r_dot[:i, 1], dx=h) + r[0, 1]
        #     r_estimate[i, 2] = simps(r_dot[:i, 2], dx=h) + r[0, 2]
        # r_estimate[-1] = r[-1]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
        # y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
        # z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
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
        dp_array = sqrt(current_mu2) * current_pt * norm(r_dot, axis=1)
        integration = simps(dp_array, dx=h)
        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            ax.plot(dp_array)
            if save is not None:
                fig.savefig(join_path("SavedPlots", f'TotalPValues_{save}.png'))
            else:
                plt.show()
            plt.close(fig)
        return integration

    def visualize(self, plot_all=False, show=True):
        fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        ax.set_title(f"3D Ray Trace with a {int(self.frequency/1E6)} MHz frequency")
        self.atmosphere.visualize(self.initial_coordinates, self.final_coordinates, fig=fig, ax=ax, point_number=200)
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
        if show:
            plt.show()
            plt.close(fig)
        else:
            return fig, ax


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
    # atmosphere.visualize(initial, final, ax=None, fig=None, point_number=400, show=True)
    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.parameters = (50, 0)
    basic_tracer.parameter_number = 50
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = initial, final

    basic_tracer.compile_initial_path()
    # basic_tracer.visualize(plot_all=True)

    paths = basic_tracer.trace()
