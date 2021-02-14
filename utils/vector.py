"""
General utils to help with vectorized calculations
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


def normalize_single_vector(a: np.ndarray) -> np.ndarray:
    """
    Normalized the provided vector
    :param a: A vector of shape (N, )
    :return: A vector of shape (N, ) parallel to a with unit length
    """
    norm = linalg.norm(a)

    return a / linalg.norm(a)


def normalize_rows(a: np.ndarray) -> np.ndarray:
    """
    Normalized the provided vector row-wise
    :param a: An array of shape (N, M)
    :return: A vector of shape (N, M), where each row is unit length and parallel to the row in the original vector
    """
    norm = linalg.norm(a, axis=-1)
    zero_parts = norm == 0
    if np.any(zero_parts):
        warnings.warn(RuntimeWarning(
            "Attempting to normalize a 0 vector. "
            "Returning 0 vector on rows instead of infinity"
        ))
        norm[zero_parts] = 1

    return a / norm


def flatten_if_necessary(vecs: np.ndarray) -> np.ndarray:
    """
    If vecs is of shape (1, N), flatten it to (N, )
    :param vecs: An array of shape (M, N)
    :return: An array of shape (M, N) or (N, ) depending on whether M is 1 or not
    """
    if vecs.shape[0] == 1:
        return vecs.flatten()
    else:
        return vecs


def angle_between_vector_collections(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Internal utility. For speed, vec1 and vec2 need to be of shape (N, M)
    :param vec1: vector of shape (N, M)
    :param vec2: vector of shape (N, M)
    :return: angles between collections in array of shape (N, )
    """
    return np.arccos(row_dot_product(vec1, vec2) / (
            linalg.norm(vec2, axis=1) * linalg.norm(vec1, axis=1)))
