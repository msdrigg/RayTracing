import math
import warnings
from scipy import linalg
import numpy as np
from scipy.spatial.transform import Rotation


# Earth radius meters. DO NOT EDIT
EARTH_RADIUS = 6.371E6


# This takes geographic as a numpy vector of coordinates, or single coordinates
#   (latitude: degrees, longitude: degrees, altitude: meters)
# It returns
#   (radius: meters, polar: radians, azimuthal: radians)
def geographic_to_spherical(geographic: np.ndarray) -> np.ndarray:
    geographic_vectorized = np.atleast_2d(geographic)
    if np.any(geographic_vectorized[:, 2] < - EARTH_RADIUS):
        warnings.warn(RuntimeWarning("Getting at least one geographic vectors with altitudes less than -EARTH_RADIUS"))
    spherical = np.empty_like(geographic_vectorized, dtype=np.float)
    spherical[:, 0] = geographic_vectorized[:, 2] + EARTH_RADIUS
    spherical[:, 1] = (90 + geographic_vectorized[:, 0]) * math.pi / 180
    spherical[:, 2] = geographic_vectorized[:, 1] * math.pi / 180

    return _flatten_if_necessary(spherical)


# This performs the inverse of geographic_to_spherical
def spherical_to_geographic(spherical: np.ndarray) -> np.ndarray:
    spherical_vectorized = np.atleast_2d(spherical)
    if np.any(spherical_vectorized[:, 0] < 0):
        warnings.warn(RuntimeWarning("Getting at least one spherical vectors with radius's less than 0"))
    geographic = np.empty_like(spherical_vectorized, dtype=np.float)
    geographic[:, 0] = 180 / math.pi * spherical_vectorized[:, 1] - 90
    geographic[:, 1] = spherical_vectorized[:, 2] * 180 / math.pi
    geographic[:, 2] = spherical_vectorized[:, 0] - EARTH_RADIUS

    return _flatten_if_necessary(geographic)


def spherical_to_cartesian(spherical):
    """Returns the cartesian form of the spherical vector.
        spherical_vector can be a numpy array where rows are spherical vectors
        (or a single vector)"""
    spherical_vector = np.atleast_2d(spherical)
    cartesian_vector = np.empty_like(spherical_vector, dtype=np.float)

    cartesian_vector[:, 0] = spherical_vector[:, 0] * np.sin(spherical_vector[:, 1]) * np.cos(spherical_vector[:, 2])
    cartesian_vector[:, 1] = spherical_vector[:, 0] * np.sin(spherical_vector[:, 1]) * np.sin(spherical_vector[:, 2])
    cartesian_vector[:, 2] = spherical_vector[:, 0] * np.cos(spherical_vector[:, 1])

    return _flatten_if_necessary(cartesian_vector)


def cartesian_to_spherical(cartesian):
    """Returns the spherical form of the cartesian vector.
        cartesian can be a numpy array where rows are cartesian vectors
        (or a single vector)"""
    cartesian_vector = np.atleast_2d(cartesian)
    spherical_vector = np.empty_like(cartesian_vector, dtype=np.float)

    r = linalg.norm(cartesian_vector, axis=1)
    spherical_vector[:, 0] = r
    spherical_vector[:, 1] = np.arccos(cartesian_vector[:, 2] / r)
    spherical_vector[:, 2] = np.arctan2(cartesian_vector[:, 1], cartesian_vector[:, 0])

    return _flatten_if_necessary(spherical_vector)


# Converts a spherical coordinate to a coordinate in the form of
# (radius, 'distance in radians along path', 'normal distance in radians to path')
def spherical_to_path_component(
        spherical: np.ndarray,
        path_start: np.ndarray,
        path_end: np.ndarray) -> np.ndarray:
    """
    Converts coordinates to path component form (distance along path (along earths surface),
        distance normal to path (along earths surface), height (above earths surface))
    :param spherical: the coordinate or array of coordinates to rotate (in spherical form)
    :param path_start: the coordinate of path start (spherical coordinates)
    :param path_end: the coordinate of the path end (spherical coordinates)
    :return: coordinate or array of coordinates in path component form
    """
    spherical_vector = np.atleast_2d(spherical)
    cartesian_components = spherical_to_cartesian(spherical_vector)
    cartesian_start = spherical_to_cartesian(path_start)
    cartesian_end = spherical_to_cartesian(path_end)

    path_components = np.empty_like(spherical_vector, dtype=np.float)

    normal_vec_to_plane = np.cross(cartesian_start, cartesian_end)
    normalized_normal_vec = normal_vec_to_plane / linalg.norm(normal_vec_to_plane)
    vector_normal_component = np.dot(cartesian_components, normalized_normal_vec)
    vectors_projected_onto_plane = cartesian_components - normalized_normal_vec * vector_normal_component

    path_components[:, 0] = _angle_between_vector_collections(path_start, vectors_projected_onto_plane) * EARTH_RADIUS
    path_components[:, 1] = _angle_between_vector_collections(vectors_projected_onto_plane,
                                                              cartesian_components)*EARTH_RADIUS
    path_components[:, 2] = linalg.norm(spherical_vector[:, 0] - EARTH_RADIUS)

    return _flatten_if_necessary(path_components)


def _flatten_if_necessary(vecs: np.ndarray):
    if vecs.shape[0] == 1:
        return vecs.flatten()
    else:
        return vecs


def _angle_between_vector_collections(vec1, vec2):
    """
    Internal utility. For speed, vec1 and vec2 need to be of shape (N, 3)
    :param vec1: vector of shape (N, 3)
    :param vec2: vector of shape (N, 3)
    :return: angles between collections in array of shape (N, )
    """
    return np.arccos(np.dot(vec1, vec2) / (
            linalg.norm(vec2) * linalg.norm(vec1)))


def path_component_to_spherical(
        path_components: np.ndarray,
        path_start: np.ndarray,
        path_end: np.ndarray) -> np.ndarray:
    """
    NOTE: path_start and path_end cannot be at opposite poles
        (obviously, because then path between them is arbitrary)
    :param path_components: the coordinate or array of coordinates to rotate (in path component form)
    :param path_start: the coordinate of path start (spherical coordinates)
    :param path_end: the coordinate of the path end (spherical coordinates)
    :return: coordinate or array of coordinates in spherical coordinates
    """
    path_components_vector = np.atleast_2d(path_components)

    cartesian_start = spherical_to_cartesian(path_start)
    cartesian_end = spherical_to_cartesian(path_end)

    normal_vector_to_path = np.cross(cartesian_start, cartesian_end)
    unit_normal_vector_to_path = normal_vector_to_path/linalg.norm(normal_vector_to_path)

    rotations = Rotation.from_rotvec(
        unit_normal_vector_to_path[np.newaxis, :] * path_components_vector[:, 1]/EARTH_RADIUS[:, np.newaxis]
    )
    vecs_along_path = rotations.apply(cartesian_start)

    perpendicular_rotation_vecs = np.cross(vecs_along_path, unit_normal_vector_to_path)
    unit_rotation_vecs = perpendicular_rotation_vecs/linalg.norm(perpendicular_rotation_vecs)
    perpendicular_rotations = Rotation.from_rotvec(
        unit_rotation_vecs*path_components_vector[:, 2]/EARTH_RADIUS
    )
    cartesian_components = perpendicular_rotations.apply(vecs_along_path)

    return cartesian_to_spherical(cartesian_components)


def regularize_spherical_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Returns coordinates in standard form (r > 0, theta in [0, pi], phi in [0, 2pi]
    :param coordinates: coordinate or array of coordinates in spherical form
    :return: coordinates in spherical form, but this time with proper coordinates
    """
    return cartesian_to_spherical(spherical_to_cartesian(coordinates))
