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


def get_angle_of_shortest_path(
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float) -> float:
    """
    Calculates the angle that minimizes ground distance
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: The angle of launch (beta_0) that yields the shortest QP path
    """
    
    pass


def get_pedersen_angle(
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        operating_frequency: float) -> float:
    """
    Calculates the pedersen angle for the atmosphere. See Equation 10 in Hill 1979
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
    radical = -b * (atmosphere_base_height + b / (2 * a)) / 2
    return math.acos(radical/coords.EARTH_RADIUS)


def get_full_path_ground_distance(launch_angle: float, *atmosphere_args):
    """
    Returns the full distance along the ground in qp atmosphere
    :param launch_angle: path launch angle
    :param atmosphere_args: common atmosphere args including r_m, r_b, N_m, f
    :return: the full distance in meters along the ground of the path
    """
    return get_apogee_ground_distance(launch_angle, *atmosphere_args) * 2


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
    Gets the ground distances of a ray path in the qp atmosphere. See equation 6 in Hill 1979
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

    output = np.where_array(heights <= atmosphere_base_height, below_atmosphere_output, inside_atmosphere_output)
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
    u_denominator = coords.EARTH_RADIUS * math.cos(launch_angle)
    u = u_numerator / u_denominator

    v_numerator = 2 * c + atmosphere_base_height*b + 2 * math.sqrt(c * x_b)
    v_denominator = atmosphere_base_height * np.exp(u)
    v = v_numerator / v_denominator

    heights_numerator = 4 * v * c
    heights_denominator = (np.square(v) + b ** 2) - (4 * a * c + 2 * v * b)

    heights_above_atmosphere = heights_numerator / heights_denominator
    beta = ground_distances / coords.EARTH_RADIUS - launch_angle
    heights_below_atmosphere = (coords.EARTH_RADIUS * math.cos(launch_angle)) / np.cos(beta)

    # Found by using equation 5 (Hill 1979) with r = r_b
    atmosphere_penetration_ground_distance = coords.EARTH_RADIUS * (
        math.acos((coords.EARTH_RADIUS * math.cos(launch_angle)) / atmosphere_base_height) - launch_angle
    )

    return np.where(
        ground_distances <= atmosphere_penetration_ground_distance,
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
    points along a path in a quasiparabolic atmosphere
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
    angle_of_shortest_path = get_angle_of_shortest_path(*atmosphere_params)

    # Function who's roots are the launch angles we want
    def minimization_function(launch_angle):
        get_full_path_ground_distance(launch_angle, *atmosphere_params) - path_ground_distance

    # Get path angles
    high_path_launch_angle, high_results = optimize.bisect(
        minimization_function, angle_of_shortest_path, get_pedersen_angle(*atmosphere_params), full_output=True
    )
    low_path_launch_angle, low_results = optimize.bisect(
        minimization_function, 0, angle_of_shortest_path, full_output=True
    )

    # Showing convergence errors
    if not low_results.converged:
        raise RuntimeError("Low path launch angle calculation failed to "
                           f"converge with reason:\n{low_results.flag}")
    if not high_results.converged:
        raise RuntimeError("High path launch angle calculation failed "
                           f"to converge with reason:\n{high_results.flag}")

    # Getting angles as tuple
    if abs(high_path_launch_angle - low_path_launch_angle) < 1E-6:
        angles = (high_path_launch_angle,)
    else:
        angles = (high_path_launch_angle, low_path_launch_angle)

    paths = ()
    distances = np.linspace(0, path_ground_distance, num=1 + (path_ground_distance//step_size_horizontal))
    for angle in angles:
        # TODO: Fix this path calculation. We need to get ground distances, and combine with heights.
        paths = paths + (get_qp_ground_distances(angle, distances, *atmosphere_params),)

    # Checking that these paths match desired results
    for path in paths:
        if abs(path[-1, 0] - path_ground_distance) < 1E-2:
            abs_error = path[-1, 0] - path_ground_distance
            rel_error = (path[-1, 0] - path_ground_distance) * 2 / (path[-1, 0] + path_ground_distance)
            raise RuntimeError("One path launch angle calculation failed "
                               "to converge to the expected result.\n"
                               f"Abs error: {abs_error}.  "
                               f"Relative error: {rel_error}")

    return paths
