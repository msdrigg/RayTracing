"""
Functions to perform conversions between certain coordinate systems
"""
import math
import warnings
from scipy import linalg
import numpy as np
from scipy.spatial.transform import Rotation
import Vector as vector
from typing import Optional
from Constants import EARTH_RADIUS


# This takes geographic as a numpy vector of coordinates, or single coordinates
#   (latitude: degrees, longitude: degrees, altitude: meters)
# It returns
#   (radius: meters, polar: radians, azimuthal: radians)
def geographic_to_spherical(geographic: np.ndarray) -> np.ndarray:
    """
    Converts coordinates in (latitude, longitude) format with angles in degrees
    to spherical coordinates with angles in radians
    :param geographic: An array of shape (N, 3) with rows being (latitude, longitude, altitude)
    :return: An array of shape (N, 3) with rows the spherical coordinates corresponding to the provided locations
    """
    if np.any(geographic[..., 2] < - EARTH_RADIUS):
        warnings.warn(RuntimeWarning("Getting at least one geographic vectors with altitudes less than -EARTH_RADIUS"))
    spherical = np.empty_like(geographic, dtype=float)
    spherical[..., 0] = geographic[..., 2] + EARTH_RADIUS
    spherical[..., 1] = (90 + geographic[..., 0]) * math.pi / 180
    spherical[..., 2] = geographic[..., 1] * math.pi / 180

    return spherical.reshape(geographic.shape)


# This performs the inverse of geographic_to_spherical
def spherical_to_geographic(spherical: np.ndarray) -> np.ndarray:
    """
    Converts spherical coordinates to geographic
    :param spherical: An array of size (N, 3) with rows corresponding to spherical coordinates
    :return: An array of size (N, 2) with rows being (latitude, longitude),
     which correspond to the provided coordinates
    """
    if np.any(spherical[..., 0] < 0):
        warnings.warn(RuntimeWarning("Getting at least one spherical vectors with radius's less than 0"))
    geographic = np.empty_like(spherical, dtype=float)
    geographic[..., 0] = 180 / math.pi * spherical[..., 1] - 90
    geographic[..., 1] = spherical[..., 2] * 180 / math.pi
    geographic[..., 2] = spherical[..., 0] - EARTH_RADIUS

    return geographic.reshape(spherical.shape)


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """Returns the cartesian form of the spherical vector.
        spherical_vector can be a numpy array where rows are spherical vectors
        (or a single vector)"""
    cartesian = np.empty_like(spherical, dtype=float)

    cartesian[..., 0] = spherical[..., 0] * np.sin(spherical[..., 1]) * np.cos(spherical[..., 2])
    cartesian[..., 1] = spherical[..., 0] * np.sin(spherical[..., 1]) * np.sin(spherical[..., 2])
    cartesian[..., 2] = spherical[..., 0] * np.cos(spherical[..., 1])

    return cartesian


def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
    """Returns the spherical form of the cartesian vector.
        cartesian can be a numpy array where rows are cartesian vectors
        (or a single vector)"""
    spherical = np.empty_like(cartesian, dtype=float)

    r = linalg.norm(cartesian, axis=-1)
    spherical[..., 0] = r
    spherical[..., 1] = np.arccos(cartesian[..., 2] / r)
    spherical[..., 2] = np.arctan2(cartesian[..., 1], cartesian[..., 0])

    return spherical.reshape(cartesian.shape)


# Converts a spherical coordinate to a coordinate in the form of
# (radius, 'distance in radians along path', 'normal distance in radians to path')
def standard_to_path_component(
        standard: np.ndarray,
        path_start_spherical: np.ndarray,
        path_end_spherical: np.ndarray) -> np.ndarray:
    """
    Converts coordinates to path component form (distance along path (along earths surface),
        distance normal to path (along earths surface), height (above earths surface))
    :param standard: the coordinate or array of coordinates to rotate (in some standard form)
    :param path_start_spherical: the coordinate of path start (spherical coordinates)
    :param path_end_spherical: the coordinate of the path end (spherical coordinates)
    :return: coordinate or array of coordinates in path component form
    """
    cartesian = standard
    spherical = np.empty(1)

    # Path start and end
    cartesian_start = spherical_to_cartesian(path_start_spherical)
    cartesian_end = spherical_to_cartesian(path_end_spherical)

    path_components = np.empty_like(cartesian, dtype=float)

    normal_vec_to_plane = np.cross(cartesian_start, cartesian_end)
    unit_normal_vec = normal_vec_to_plane / linalg.norm(normal_vec_to_plane)

    vector_normal_component = vector.row_dot_product(cartesian, unit_normal_vec)

    vectors_projected_onto_plane = cartesian - \
        np.outer(vector_normal_component, unit_normal_vec)

    path_components[..., 1] = vector.angle_between_vector_collections(
        cartesian_start,
        vectors_projected_onto_plane
    ) * EARTH_RADIUS * np.sign(vector.row_dot_product(
        np.cross(np.atleast_2d(cartesian_start), vectors_projected_onto_plane),
        normal_vec_to_plane)
    )

    path_components[..., 2] = vector.angle_between_vector_collections(
        vectors_projected_onto_plane,
        cartesian
    ) * EARTH_RADIUS * np.sign(vector_normal_component)

    path_components[..., 0] = linalg.norm(cartesian, axis=-1)

    return path_components


def path_component_to_standard(
        path_components: np.ndarray,
        path_start_cartesian: np.ndarray,
        path_end_cartesian: np.ndarray) -> np.ndarray:
    """
    NOTE: path_start and path_end cannot be at opposite poles
        (obviously, because then path between them is arbitrary)
    :param path_components: the coordinate or array of coordinates to rotate (in path component form)
    :param path_start_cartesian: the coordinate of path start (spherical coordinates)
    :param path_end_cartesian: the coordinate of the path end (spherical coordinates)
    :return: coordinate or array of coordinates in spherical coordinates
    """

    normal_vector_to_path = np.cross(path_start_cartesian, path_end_cartesian)
    unit_normal_vector_to_path = normal_vector_to_path / linalg.norm(normal_vector_to_path)

    # Make sure we have positive rotation vector
    rotations = Rotation.from_rotvec(
        np.outer(path_components[..., 1] / EARTH_RADIUS, unit_normal_vector_to_path)
    )
    vecs_along_path = rotations.apply(path_start_cartesian)

    perpendicular_rotation_vecs = np.cross(np.atleast_2d(vecs_along_path), np.atleast_2d(unit_normal_vector_to_path))
    unit_rotation_vecs = np.einsum(
        'ij,i->ij', perpendicular_rotation_vecs,
        1 / linalg.norm(perpendicular_rotation_vecs, axis=1)
    )

    perpendicular_rotations = Rotation.from_rotvec(
        np.einsum(
            "ij,i->ij",
            unit_rotation_vecs,
            np.atleast_1d(path_components[..., 2] / EARTH_RADIUS)
        )
    )

    cartesian_components = perpendicular_rotations.apply(vecs_along_path)

    cartesian_components = cartesian_components / linalg.norm(path_start_cartesian) * \
        path_components[..., 0].reshape(-1, 1)
    return cartesian_components.reshape(path_components.shape)


def regularize_spherical_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Returns coordinates in standard form (r > 0, theta in [0, pi], phi in [0, 2pi]
    :param coordinates: coordinate or array of coordinates in spherical form
    :return: coordinates in spherical form, but this time with proper coordinates
    """
    regularized = cartesian_to_spherical(spherical_to_cartesian(coordinates))
    regularized[..., 2] = np.mod(regularized[..., 2] + 2 * math.pi, 2 * math.pi)
    return regularized
