import math
import warnings
from scipy import linalg
import numpy as np
from scipy.spatial.transform import Rotation
from utils import vector

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
    spherical = np.empty_like(geographic_vectorized, dtype=float)
    spherical[:, 0] = geographic_vectorized[:, 2] + EARTH_RADIUS
    spherical[:, 1] = (90 + geographic_vectorized[:, 0]) * math.pi / 180
    spherical[:, 2] = geographic_vectorized[:, 1] * math.pi / 180

    return vector.flatten_if_necessary(spherical)


# This performs the inverse of geographic_to_spherical
def spherical_to_geographic(spherical: np.ndarray) -> np.ndarray:
    spherical_vectorized = np.atleast_2d(spherical)
    if np.any(spherical_vectorized[:, 0] < 0):
        warnings.warn(RuntimeWarning("Getting at least one spherical vectors with radius's less than 0"))
    geographic = np.empty_like(spherical_vectorized, dtype=float)
    geographic[:, 0] = 180 / math.pi * spherical_vectorized[:, 1] - 90
    geographic[:, 1] = spherical_vectorized[:, 2] * 180 / math.pi
    geographic[:, 2] = spherical_vectorized[:, 0] - EARTH_RADIUS

    return vector.flatten_if_necessary(geographic)


def spherical_to_cartesian(spherical):
    """Returns the cartesian form of the spherical vector.
        spherical_vector can be a numpy array where rows are spherical vectors
        (or a single vector)"""
    spherical_vector = np.atleast_2d(spherical)
    cartesian_vector = np.empty_like(spherical_vector, dtype=float)

    cartesian_vector[:, 0] = spherical_vector[:, 0] * np.sin(spherical_vector[:, 1]) * np.cos(spherical_vector[:, 2])
    cartesian_vector[:, 1] = spherical_vector[:, 0] * np.sin(spherical_vector[:, 1]) * np.sin(spherical_vector[:, 2])
    cartesian_vector[:, 2] = spherical_vector[:, 0] * np.cos(spherical_vector[:, 1])

    return vector.flatten_if_necessary(cartesian_vector)


def cartesian_to_spherical(cartesian):
    """Returns the spherical form of the cartesian vector.
        cartesian can be a numpy array where rows are cartesian vectors
        (or a single vector)"""
    cartesian_vector = np.atleast_2d(cartesian)
    spherical_vector = np.empty_like(cartesian_vector, dtype=float)

    r = linalg.norm(cartesian_vector, axis=1)
    spherical_vector[:, 0] = r
    spherical_vector[:, 1] = np.arccos(cartesian_vector[:, 2] / r)
    spherical_vector[:, 2] = np.arctan2(cartesian_vector[:, 1], cartesian_vector[:, 0])

    return vector.flatten_if_necessary(spherical_vector)


# Converts a spherical coordinate to a coordinate in the form of
# (radius, 'distance in radians along path', 'normal distance in radians to path')
def standard_to_path_component(
        standard: np.ndarray,
        path_start_spherical: np.ndarray,
        path_end_spherical: np.ndarray,
        from_spherical: bool = True) -> np.ndarray:
    """
    Converts coordinates to path component form (distance along path (along earths surface),
        distance normal to path (along earths surface), height (above earths surface))
    :param standard: the coordinate or array of coordinates to rotate (in some standard form)
    :param path_start_spherical: the coordinate of path start (spherical coordinates)
    :param path_end_spherical: the coordinate of the path end (spherical coordinates)
    :param from_spherical: if true, assume components are spherical, else they are cartesian
    :return: coordinate or array of coordinates in path component form
    """
    if from_spherical:
        spherical_vector = np.atleast_2d(standard)
        cartesian_components = np.atleast_2d(spherical_to_cartesian(spherical_vector))
    else:
        cartesian_components = standard
        spherical_vector = np.empty(1)

    # Path start and end
    cartesian_start = spherical_to_cartesian(path_start_spherical)
    cartesian_end = spherical_to_cartesian(path_end_spherical)

    path_components = np.empty_like(cartesian_components, dtype=float)

    normal_vec_to_plane = np.cross(cartesian_start, cartesian_end)
    unit_normal_vec = normal_vec_to_plane / linalg.norm(normal_vec_to_plane)

    vector_normal_component = np.einsum("ij,j->i", cartesian_components, unit_normal_vec)

    vectors_projected_onto_plane = cartesian_components - \
        np.outer(vector_normal_component, unit_normal_vec)

    path_components[:, 1] = vector.angle_between_vector_collections(
        cartesian_start.reshape(-1, 3),
        vectors_projected_onto_plane
    ) * EARTH_RADIUS * np.sign(vector.row_dot_product(
        np.cross(cartesian_start.reshape(-1, 3), vectors_projected_onto_plane),
        normal_vec_to_plane[np.newaxis, :])
    )

    path_components[:, 2] = vector.angle_between_vector_collections(
        vectors_projected_onto_plane,
        cartesian_components
    ) * EARTH_RADIUS * np.sign(vector_normal_component)

    if from_spherical:
        path_components[:, 0] = spherical_vector[:, 0] - EARTH_RADIUS
    else:
        path_components[:, 0] = linalg.norm(cartesian_components) - EARTH_RADIUS

    return vector.flatten_if_necessary(path_components)


def path_component_to_standard(
        path_components: np.ndarray,
        path_start_spherical: np.ndarray,
        path_end_spherical: np.ndarray,
        to_spherical: bool = True) -> np.ndarray:
    """
    NOTE: path_start and path_end cannot be at opposite poles
        (obviously, because then path between them is arbitrary)
    :param path_components: the coordinate or array of coordinates to rotate (in path component form)
    :param path_start_spherical: the coordinate of path start (spherical coordinates)
    :param path_end_spherical: the coordinate of the path end (spherical coordinates)
    :param to_spherical: if true, return spherical, else return cartesian
    :return: coordinate or array of coordinates in spherical coordinates
    """
    path_components_vector = np.atleast_2d(path_components)

    cartesian_start = spherical_to_cartesian(path_start_spherical)
    cartesian_end = spherical_to_cartesian(path_end_spherical)

    normal_vector_to_path = np.cross(cartesian_start, cartesian_end)
    unit_normal_vector_to_path = normal_vector_to_path / linalg.norm(normal_vector_to_path)

    # Make sure we have positive rotation vector
    rotations = Rotation.from_rotvec(
        np.outer(path_components_vector[:, 1] / EARTH_RADIUS, unit_normal_vector_to_path)
    )
    vecs_along_path = np.atleast_2d(rotations.apply(cartesian_start))

    perpendicular_rotation_vecs = np.cross(vecs_along_path, unit_normal_vector_to_path.reshape(-1, 3))
    unit_rotation_vecs = np.einsum(
        'ij,i->ij', perpendicular_rotation_vecs,
        1 / linalg.norm(perpendicular_rotation_vecs, axis=1)
    )

    perpendicular_rotations = Rotation.from_rotvec(
        np.einsum(
            "ij,i->ij",
            unit_rotation_vecs,
            path_components_vector[:, 2] / EARTH_RADIUS
        )
    )

    cartesian_components = perpendicular_rotations.apply(vecs_along_path)

    if not to_spherical:
        return cartesian_components

    spherical_components = np.atleast_2d(cartesian_to_spherical(cartesian_components))

    spherical_components[:, 0] = path_components_vector[:, 0] + EARTH_RADIUS
    return vector.flatten_if_necessary(spherical_components)


def regularize_spherical_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Returns coordinates in standard form (r > 0, theta in [0, pi], phi in [0, 2pi]
    :param coordinates: coordinate or array of coordinates in spherical form
    :return: coordinates in spherical form, but this time with proper coordinates
    """
    regularized = np.atleast_2d(cartesian_to_spherical(spherical_to_cartesian(coordinates)))
    regularized[:, 2] = np.mod(regularized[:, 2] + 2 * math.pi, 2 * math.pi)
    return vector.flatten_if_necessary(regularized)
