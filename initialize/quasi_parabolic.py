"""
This file holds equations that come primarily from the
1979 Jay Roderick Hill paper on ray tracing in the
Quasi-Parabolic atmosphere. There is also some work from
the 1968 Croft and Hoogasian paper on the same topic.
I will cite each equation as it comes from that paper.
"""

import numpy as np
from scipy import optimize
from utils import coordinates as coords
import math


def calculate_param_a(
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Calculates the A parameter following Hill 1979
    """
    semi_width = atmosphere_base_height - atmosphere_height_of_max
    t1 = 80.62*atmosphere_max_e_density / operating_frequency ** 2
    t2 = 80.62 * atmosphere_max_e_density * (atmosphere_base_height / (operating_frequency * semi_width)) ** 2
    return (1 + t2) - t1


def calculate_param_b(
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Calculates the B parameter following Hill 1979
    """
    semi_width = atmosphere_base_height - atmosphere_height_of_max
    top = -2 * atmosphere_height_of_max * 80.62 * atmosphere_max_e_density * atmosphere_base_height ** 2
    bottom = (operating_frequency * semi_width) ** 2
    return top/bottom


def calculate_param_c(
        launch_angle: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Calculates the C parameter following Hill 1979
    """
    semi_width = atmosphere_base_height - atmosphere_height_of_max
    top = 80.62 * atmosphere_max_e_density * (atmosphere_base_height * atmosphere_height_of_max) ** 2
    bottom = (operating_frequency * semi_width) ** 2
    return top/bottom - (coords.EARTH_RADIUS * math.cos(launch_angle)) ** 2


def calculate_param_beta_b(launch_angle: float, atmosphere_base_height: float) -> float:
    """
    Calculates $beta_b$ for atmosphere calculations following equation 6 in Hill 1979
    """
    return math.acos(coords.EARTH_RADIUS * math.cos(launch_angle) / atmosphere_base_height)


def calculate_param_x_b(launch_angle: float, atmosphere_base_height: float) -> float:
    """
    Calculates $x_b$ for atmosphere calculations following equation 6 in Hill 1979
    """
    return atmosphere_base_height ** 2 - (coords.EARTH_RADIUS * math.cos(launch_angle)) ** 2


def ground_distance_derivative(
        launch_angle: float,
        *atmosphere_params):
    """
    Calculates the derivative of ground distance with respect to launch angle
    This is my own work, and is derived using mathematica
    See equation 6 in Hill 1979
    """
    atmosphere_base_height = atmosphere_params[1]
    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    c = calculate_param_c(launch_angle, *atmosphere_params)

    discriminant = b ** 2 - 4 * a * c
    x_b = calculate_param_x_b(launch_angle, atmosphere_base_height)
    r_b2_minus_x_b = (coords.EARTH_RADIUS * math.cos(launch_angle)) ** 2

    log_numerator = (2 * c + 2 * math.sqrt(c * x_b)) + b * atmosphere_base_height
    log_denominator = math.sqrt(discriminant) * atmosphere_base_height
    log_term = math.log(log_numerator / log_denominator)

    sqrt_c_x_b = math.sqrt(c * x_b)

    neg_term_1 = -1
    neg_term_2 = -coords.EARTH_RADIUS * log_term * math.sin(launch_angle) / math.sqrt(c)
    neg_term_3 = -coords.EARTH_RADIUS * r_b2_minus_x_b * log_term * math.sin(launch_angle) / math.sqrt(c ** 3)

    pos_term_1 = coords.EARTH_RADIUS * math.sin(launch_angle) / math.sqrt(x_b)

    pos_term_2_numerator = (2 * coords.EARTH_RADIUS * r_b2_minus_x_b *
                            (c + x_b + 2 * sqrt_c_x_b) * math.sin(launch_angle))
    pos_term_2_denominator = math.sqrt(c) * (b * atmosphere_base_height * sqrt_c_x_b + 2 * c * (x_b + sqrt_c_x_b))
    pos_term_2 = pos_term_2_numerator / pos_term_2_denominator

    pos_term_3 = 4 * a * coords.EARTH_RADIUS ** 3 * (math.cos(launch_angle)) ** 2 * math.sin(launch_angle) / \
        (math.sqrt(c) * discriminant)

    return coords.EARTH_RADIUS * ((pos_term_1 + pos_term_2 + pos_term_3) + (neg_term_1 + neg_term_2 + neg_term_3))


def get_angle_of_shortest_path(*atmosphere_params) -> float:
    """
    Calculates the angle that minimizes ground distance
    :return: The angle of launch (beta_0) that yields the
    shortest QP path
    """
    try:
        pedersen_angle = get_pedersen_angle(*atmosphere_params)
        result, _ = optimize.bisect(lambda a: ground_distance_derivative(a, *atmosphere_params),
                                    0, pedersen_angle - 1E-6,
                                    disp=True, full_output=True)
        return result

    except ValueError:
        raise ValueError("Attempting shortest path calculation for a atmosphere that will always reflect the ray."
                         "You can assume the shortest path is pi/2")


def get_pedersen_angle(
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Calculates the pedersen angle for the atmosphere.
    See Equation 10 in Hill 1979
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :param operating_frequency: see get_quasi_parabolic_path
    :return: The pedersen angle (in radians)
    """
    atmosphere_params = (
        atmosphere_height_of_max, atmosphere_base_height,
        atmosphere_max_e_density, operating_frequency
    )

    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    radical = -b * (atmosphere_height_of_max + b / (2 * a)) / 2

    try:
        inside_term = math.sqrt(radical)/coords.EARTH_RADIUS
        return math.acos(inside_term)
    except ValueError:
        raise ValueError("Error finding pedersen angle, due to. "
                         "The likely cause is invalid atmosphere parameters.")


def get_apogee_height(
        launch_angle: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Returns the max ray height above the origin (not ground) in qp atmosphere
    See Equation 7 in Hill 1979
    :param launch_angle: path launch angle
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :param operating_frequency: see get_quasi_parabolic_path
    :return: the radius of the path apogee
    """
    atmosphere_params = (
        atmosphere_height_of_max, atmosphere_base_height,
        atmosphere_max_e_density, operating_frequency
    )

    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    c = calculate_param_c(launch_angle, *atmosphere_params)

    discriminant = b ** 2 - (4 * a * c)

    if discriminant <= 0:
        raise ValueError("Parameters are not valid. "
                         "In the given configuration, the ray will never return to the ground")

    return - (b + math.sqrt(discriminant)) / (2 * a)


def get_qp_ground_distances(
        launch_angle: float,
        heights: np.ndarray,
        *atmosphere_params) -> np.ndarray:
    """
    NOTE: THIS FUNCTION IS THE ONLY UNTESTED FUNCTION.
          IT LIKELY HAS BUGS AND/OR TYPOS. I DIDN'T TEST IT BECAUSE IT
          ISN'T USED IN THE ACTUAL QP PATH CALCULATION.
          TEST ON YOUR OWN BEFORE USING!!
    Gets the ground distances of a ray path in the qp atmosphere.
    See equation 6 in Hill 1979
    :param launch_angle: launch angle of the path
    :param heights: array of heights for which path ground distances will be computed.
    :param atmosphere_params: parameters describing the atmosphere. See get_quasi_parabolic_path
    :return: the ground distances of the path as an array of size (N, ) where array elements are in meters
             There are 2 possible ground distances for every height. These ground distances calculated as the shortest
             of the two, but you can do (full_distance - returned_distance) to get the larger of the possibilities
    """
    atmosphere_base_height = atmosphere_params[1]

    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    c = calculate_param_c(launch_angle, *atmosphere_params)

    if b ** 2 - (4 * a * c) <= 0:
        raise ValueError("Parameters are not valid. "
                         "In the given configuration, the ray will never return to the ground")

    below_atmosphere_output = coords.EARTH_RADIUS * (
            np.arccos(coords.EARTH_RADIUS * math.cos(launch_angle) / heights) - launch_angle)

    x_b = calculate_param_x_b(launch_angle, atmosphere_base_height)
    beta_b = calculate_param_beta_b(launch_angle, atmosphere_base_height)
    x = a * np.square(heights) + b * heights + c

    term_1 = coords.EARTH_RADIUS * (beta_b - launch_angle)

    multiplier = coords.EARTH_RADIUS ** 2 * np.cos(launch_angle) / np.sqrt(c)
    log_numerator = heights * (2 * c + atmosphere_base_height * b + 2 * np.sqrt(c * x_b))
    log_denominator = atmosphere_base_height * (2 * c + heights * b + 2 * np.sqrt(c * x))

    inside_atmosphere_output = term_1 + multiplier * np.log(log_numerator / log_denominator)

    output = np.where(heights <= atmosphere_base_height, below_atmosphere_output, inside_atmosphere_output)
    return output


def get_apogee_ground_distance(
        launch_angle: float,
        *atmosphere_params) -> np.ndarray:
    """
    Gets the path in the qp atmosphere
    :param launch_angle: launch angle of the path
    :param atmosphere_params: parameters defining the atmosphere
    :return: the ground distances of the path as an array of size (N, )
    where array elements are ground distances in meters
    """
    atmosphere_base_height = atmosphere_params[1]
    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    c = calculate_param_c(launch_angle, *atmosphere_params)

    discriminant = b ** 2 - (4 * a * c)

    if discriminant <= 0:
        raise ValueError("Parameters are not valid. "
                         "In the given configuration, the ray will never return to the ground")

    beta_b = calculate_param_beta_b(launch_angle, atmosphere_base_height)
    x_b = calculate_param_x_b(launch_angle, atmosphere_base_height)

    term1 = coords.EARTH_RADIUS * (beta_b - launch_angle)

    multiplier = (coords.EARTH_RADIUS ** 2 * math.cos(launch_angle)) / math.sqrt(c)
    log_numerator = 2 * c + b * atmosphere_base_height + 2 * math.sqrt(c * x_b)
    log_denominator = atmosphere_base_height * math.sqrt(discriminant)

    return term1 + multiplier * math.log(log_numerator / log_denominator)


def get_qp_heights(
        launch_angle: float,
        ground_distances: np.ndarray,
        *atmosphere_params: float) -> np.ndarray:
    """
    Gets the heights of a ray path in the qp atmosphere.
    See equation 11 in Hill 1979
    :param launch_angle: launch angle of the path
    :param ground_distances: array of ground distances for which path height will be computed
    :return: the height of the path as an array of size (N, ) where array elements are heights in meters
    """
    atmosphere_base_height = atmosphere_params[1]
    a = calculate_param_a(*atmosphere_params)
    b = calculate_param_b(*atmosphere_params)
    c = calculate_param_c(launch_angle, *atmosphere_params)
    beta_b = calculate_param_beta_b(launch_angle, atmosphere_base_height)
    x_b = calculate_param_x_b(launch_angle, atmosphere_base_height)

    if b ** 2 - (4 * a * c) <= 0:
        raise ValueError("Parameters are not valid. "
                         "In the given configuration, the ray will never return to the ground")

    middle_distance = get_apogee_ground_distance(launch_angle, *atmosphere_params)
    ground_distances_unified = np.where(
        ground_distances <= middle_distance,
        ground_distances,
        (middle_distance * 2 - ground_distances)
    )

    u_numerator = math.sqrt(c) * ((ground_distances_unified + coords.EARTH_RADIUS * launch_angle) -
                                  coords.EARTH_RADIUS * beta_b)
    u_denominator = coords.EARTH_RADIUS ** 2 * math.cos(launch_angle)
    u = u_numerator / u_denominator
    v_numerator = 2 * c + atmosphere_base_height*b + 2 * math.sqrt(c * x_b)
    v_denominator = atmosphere_base_height * np.exp(u)
    v = v_numerator / v_denominator

    heights_numerator = 4 * v * c
    heights_denominator = (np.square(v) + b ** 2) - (4 * a * c + 2 * v * b)

    heights_above_atmosphere = heights_numerator / heights_denominator
    beta = ground_distances_unified / coords.EARTH_RADIUS + launch_angle
    heights_below_atmosphere = (coords.EARTH_RADIUS * math.cos(launch_angle)) / np.cos(beta)

    # Found by using equation 5 (Hill 1979) with r = r_b
    atmosphere_penetration_ground_distance = coords.EARTH_RADIUS * (
        math.acos((coords.EARTH_RADIUS * math.cos(launch_angle)) / atmosphere_base_height) - launch_angle
    )

    return np.where(
        ground_distances_unified <= atmosphere_penetration_ground_distance,
        heights_below_atmosphere,
        heights_above_atmosphere
    )


def get_quasi_parabolic_path(
        path_ground_distance: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float,
        step_size_horizontal: float = 5) -> np.ndarray:
    """
    This is the primary function to generate the
    points along a path in a quasi-parabolic atmosphere
    :param path_ground_distance: The length of the path along the earths surface (in meters)
    :param atmosphere_height_of_max: The point where the atmosphere is at its max density (r_m)
    :param atmosphere_base_height: The base of the atmosphere (r_b)
    :param atmosphere_max_e_density: The e density of the atmosphere at its peak (N_m)
    :param operating_frequency: the frequency of the ray (f)
    :param step_size_horizontal: step size between adjacent points in the path in meters
    :return: a tuple of possible paths (high, low), or (single) that make a valid path in the QP atmosphere
    Paths are an array of (N, 2) where each row is of the form (ground distance, height)
    """
    if path_ground_distance < 1E-2:
        raise RuntimeError(f"You are using {path_ground_distance} as your ground distance in meters"
                           "Please use a reasonable ground distance in meters.")

    atmosphere_params = (
        atmosphere_height_of_max, atmosphere_base_height,
        atmosphere_max_e_density, operating_frequency
    )
    try:
        angle_of_shortest_path = get_angle_of_shortest_path(*atmosphere_params)
    except ValueError:
        angle_of_shortest_path = math.pi/2

    shortest_ground_distance = get_apogee_ground_distance(angle_of_shortest_path, *atmosphere_params) * 2
    if shortest_ground_distance > path_ground_distance:
        raise ValueError("Shortest ground distance possible in this atmosphere "
                         "({}) is greater than the required ground distance {}"
                         .format(shortest_ground_distance, path_ground_distance))

    angle_calculation_intervals = ()
    epsilon = 1E-7
    try:
        angle_of_shortest_path = get_angle_of_shortest_path(*atmosphere_params)
        angle_calculation_intervals += (
            (angle_of_shortest_path + epsilon,
             get_pedersen_angle(*atmosphere_params) - epsilon),
        )
        angle_calculation_intervals += ((0 + epsilon, angle_of_shortest_path - epsilon), )
    except ValueError:
        angle_calculation_intervals += ((epsilon, math.pi/2 - epsilon), )

    # Function who's roots are the launch angles we want
    def minimization_function(launch_angle: float) -> float:
        return get_apogee_ground_distance(launch_angle, *atmosphere_params) * 2 - path_ground_distance

    # Get path angles
    angles = ()
    for interval in angle_calculation_intervals:
        try:
            new_angle, _ = optimize.bisect(
                minimization_function, interval[0], interval[1],
                full_output=True, disp=True
            )
        except ValueError:
            raise ValueError("Error finding optimal path angle in "
                             "interval (a, b) = ({}, {}), "
                             "with (f(a), f(b)) = ({}, {})"
                             .format(*interval, *[minimization_function(i) for i in interval]))
        new_angle_is_unique = True
        for angle in angles:
            if abs(angle - new_angle) < 1E-6:
                new_angle_is_unique = False

        if new_angle_is_unique:
            angles += (new_angle, )

    paths = ()
    num = 1 + math.ceil(path_ground_distance/step_size_horizontal)
    distances = np.linspace(0.0, path_ground_distance, num=num)
    for angle in angles:
        full_vector = np.empty((distances.shape[0], 2))
        full_vector[:, 0] = distances
        full_vector[:, 1] = get_qp_heights(angle, distances, *atmosphere_params)
        paths = paths + (full_vector,)

    # Checking that these paths match desired results
    for path in paths:
        if abs(path[-1, 0] - path_ground_distance) > 1E-2:
            abs_error = path[-1, 0] - path_ground_distance
            rel_error = (path[-1, 0] - path_ground_distance) * 2 / (path[-1, 0] + path_ground_distance)
            raise RuntimeError("One path launch angle calculation failed "
                               "to converge to the expected result.\n"
                               f"Abs error: {abs_error}.  "
                               f"Relative error: {rel_error}")

    return paths
