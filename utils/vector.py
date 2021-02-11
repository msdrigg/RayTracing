import numpy as np
from scipy import linalg


def row_dot_product(a, b):
    return np.einsum('ij,ij->i', np.atleast_2d(a), np.atleast_2d(b))


def normalize_single_vector(a):
    return a / linalg.norm(a)


def normalize_rows(a):
    return a / linalg.norm(a, axis=1)


def flatten_if_necessary(vecs: np.ndarray):
    if vecs.shape[0] == 1:
        return vecs.flatten()
    else:
        return vecs


def angle_between_vector_collections(vec1, vec2):
    """
    Internal utility. For speed, vec1 and vec2 need to be of shape (N, 3)
    :param vec1: vector of shape (N, 3)
    :param vec2: vector of shape (N, 3)
    :return: angles between collections in array of shape (N, )
    """
    return np.arccos(row_dot_product(vec1, vec2) / (
            linalg.norm(vec2, axis=1) * linalg.norm(vec1, axis=1)))
