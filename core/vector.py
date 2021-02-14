"""
General core to help with vectorized calculations
"""
import numpy as np
from scipy import linalg
import warnings


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


def normalize_rows(a: np.ndarray) -> np.ndarray:
    """
    Normalized the provided vector row-wise
    :param a: An array of shape (N, M)
    :return: A vector of shape (N, M), where each row is unit length and parallel to the row in the original vector
    """
    norm = np.asarray(linalg.norm(a, axis=-1))
    zero_parts = norm == 0

    if np.any(zero_parts):
        warnings.warn(RuntimeWarning(
            "Attempting to normalize a 0 vector. "
            "Returning 0 vector on rows instead of infinity"
        ))
        norm[zero_parts] = 1

    return a / norm[..., np.newaxis]


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
            linalg.norm(vec2, axis=-1) * linalg.norm(vec1, axis=-1)), -1.0, 1.0))
