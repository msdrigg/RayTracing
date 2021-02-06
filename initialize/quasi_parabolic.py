import numpy as np
from scipy import optimize


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
        atmosphere_max_e_density: float) -> float:
    """
    Calculates the pedersen angle for the atmosphere
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: The pedersen angle (in radians)
    """
    pass


def get_full_path_ground_distance(
        launch_angle: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float):
    """
    Returns the full distance along the ground in qp atmosphere
    :param launch_angle: path launch angle
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: the full distance in meters along the ground of the path
    """
    return get_apogee_ground_distance(
        launch_angle,
        atmosphere_height_of_max,
        atmosphere_base_height,
        atmosphere_max_e_density) * 2


def get_apogee_height(
        launch_angle: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float):
    """
    Returns the full distance along the ground in qp atmosphere
    :param launch_angle: path launch angle
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: the height of the path apogee
    """
    pass


def get_apogee_ground_distance(
        launch_angle: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float):
    """
    Returns the full distance along the ground in qp atmosphere
    :param launch_angle: path launch angle
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: the distance along the ground of the path apogee
    """
    pass


def get_qp_path(
        angle: float,
        vertical_step_size: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float):
    """
    Gets the path in the qp atmosphere
    :param angle: launch angle of the path
    :param vertical_step_size: see get_quasi_parabolic_path
    :param atmosphere_max_e_density: see get_quasi_parabolic_path
    :param atmosphere_base_height: see get_quasi_parabolic_path
    :param atmosphere_height_of_max: see get_quasi_parabolic_path
    :return: the path as an array of size (N, 2) where rows are (ground distance, height) in meters
    """
    pass


def get_quasi_parabolic_path(
        path_ground_distance: float,
        atmosphere_height_of_max: float,
        atmosphere_base_height: float,
        atmosphere_max_e_density: float,
        step_size_vertical: float = 5) -> np.ndarray:
    """
    This is the primary function to generate the
    points along a path in a quasiparabolic atmosphere
    :param path_ground_distance: The length of the path along the earths surface (in meters)
    :param atmosphere_height_of_max: The point where the atmosphere is at its max density (r_m)
    :param atmosphere_base_height: The base of the atmosphere (r_b)
    :param atmosphere_max_e_density: The e density of the atmosphere at its peak (N_m)
    :param step_size_vertical: step size between adjacent points in the path in meters
    :return: a tuple of possible paths (high, low), or (single) that make a valid path in the QP atmosphere
    Paths are an array of (N, 2) where each row is of the form (ground distance, height)
    """
    if path_ground_distance < 1E-2:
        raise RuntimeError(f"You are using {path_ground_distance} as your ground distance in meters"
                           "Please use a reasonable ground distance in meters.")

    atmosphere_params = (atmosphere_height_of_max, atmosphere_base_height, atmosphere_max_e_density)
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
    for angle in angles:
        paths = paths + (get_qp_path(angle, step_size_vertical, *atmosphere_params),)

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
