from numpy.linalg import norm
from numpy import cos, arccos, sin, arctan2, vstack, clip, sum
from utilities.Constants import PI
import numpy as np
import warnings


def spherical_to_cartesian(spherical_vector):
    """Returns the cartesian form of the spherical vector."""
    spherical_vector = spherical_vector.reshape(-1, 3)
    output = vstack([(spherical_vector[:, 0]*sin(spherical_vector[:, 1])*cos(spherical_vector[:, 2])),
                    spherical_vector[:, 0]*sin(spherical_vector[:, 1])*sin(spherical_vector[:, 2]),
                    spherical_vector[:, 0]*cos(spherical_vector[:, 1])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def cartesian_to_spherical(cartesian_vector):
    """Returns the cartesian form of the spherical vector."""
    cartesian_vector = cartesian_vector.reshape(-1, 3)
    r = norm(cartesian_vector, axis=1)
    output = vstack([r, arccos(cartesian_vector[:, 2]/r),
                     arctan2(cartesian_vector[:, 1], cartesian_vector[:, 0])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norms = norm(vector.reshape(-1, 3), axis=1)
    output = vector / norms.reshape(-1, 1)
    if len(output) == 1:
        return output[0]
    else:
        return output


def angle_between(s1, s2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u, v2_u = unit_vector(s1), unit_vector(s2)
    output = arccos(clip(sum(v1_u.reshape((-1, 3))*v2_u.reshape((-1, 3)), axis=1), -1.0, 1.0))
    if len(output) == 1:
        return output[0]
    else:
        return output


def unit_radius(position):
    position = cartesian_to_spherical(position).reshape(-1, 3)
    sin_phi = sin(position[:, 2])
    unit_radii = vstack([cos(position[:, 1])*sin_phi,
                         sin(position[:, 1])*sin_phi,
                         cos(position[:, 2])]).T
    if len(unit_radii) == 1:
        return unit_radii[0]
    else:
        return unit_radii


def unit_theta(position):
    position = cartesian_to_spherical(position).reshape(-1, 3)
    cos_phi = cos(position[:, 2])
    unit_thetas_return = vstack([cos(position[:, 1])*cos_phi,
                                sin(position[:, 1])*cos_phi,
                                -sin(position[:, 2])]).T
    if len(unit_thetas_return) == 1:
        return unit_thetas_return[0]
    else:
        return unit_thetas_return


def latitude_to_spherical(vector):
    vector = vector.reshape(-1, 3)
    vector[:, 1] = 90 - vector[:, 1]
    vector[:, 1:] *= PI / 180.0

    if len(vector) == 1:
        return vector[0]
    else:
        return vector


def row_dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns a vectorized dot product between the rows of a and b
    :param a: An array of shape (N, M) or (M, )
     (or a shape that can be broadcast to (N, M))
    :param b: An array of shape (N, M) or (M, )
    (or a shape that can be broadcast to (N, M))
    :return: A vector of shape (N, ) whose elements are the dot product of rows a, b
    """
    return np.einsum('ij,ij->i', np.atleast_2d(a), np.atleast_2d(b))


def normalize_last_axis(a: np.ndarray) -> np.ndarray:
    """
    Normalized the provided vector row-wise
    :param a: An array of shape (N, M)
    :return: A vector of shape (N, M), where each row is unit length and parallel to the row in the original vector
    """
    vec_norm = np.asarray(norm(a, axis=-1))
    zero_parts = vec_norm == 0

    if np.any(zero_parts):
        warnings.warn(RuntimeWarning(
            "Attempting to normalize a 0 vector. "
            "Returning 0 vector on rows instead of infinity"
        ))
        vec_norm[zero_parts] = 1

    return a / vec_norm[..., np.newaxis]


def angle_between_vector_collections(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Internal utility. For speed, vec1 and vec2 need to be of shape (N, M)
    :param vec1: vector of shape (N, M)
    :param vec2: vector of shape (N, M)
    :return: angles between collections in array of shape (N, )
    """
    if vec1.ndim > 2 or vec2.ndim > 2:
        raise ValueError("Vectors need to have 1 or 2 dimensions")
    if vec1.shape[-1] != vec2.shape[-1] or vec1.shape[-1] < 2 or vec2.shape[-1] < 2:
        raise ValueError("Vectors have unequal shape in their last axes: {}, {}"
                         .format(vec1.shape[-1], vec2.shape[-1]))
    return np.arccos(np.clip(row_dot_product(vec1, vec2) / (
            norm(vec2, axis=-1) * norm(vec1, axis=-1)), -1.0, 1.0))
