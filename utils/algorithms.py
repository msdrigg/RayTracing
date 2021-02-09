import numpy as np


def conjugate_gradient(mat_a, b, x_1):
    """
    This function will solve the system Ax = b using th
    :param mat_a: Symmetric but not necessarily positive definite matrix in the system
    :param b: Vector. Seen in matA*x = b
    :param x_1: Initial guess for the solution
    :return x_f: Our solution to system
    """
    r_k = b - mat_a @ x_1
    r_k_bar = mat_a @ r_k
    error = np.linalg.norm(r_k)
    matrix_tilde_inverse = np.diagflat(np.reciprocal(np.diagonal(mat_a)))
    z_k = matrix_tilde_inverse@r_k
    z_k_bar = matrix_tilde_inverse@r_k_bar
    p_k = z_k
    p_k_bar = z_k_bar
    x_k = x_1
    counter = 0
    while error > 1E-12:
        counter += 1
        a_k = np.dot(r_k_bar, z_k)/(p_k_bar @ mat_a @ p_k)
        r_next = r_k - a_k * mat_a @ p_k
        r_next_bar = r_k_bar - a_k * mat_a.T @ p_k_bar
        z_next = matrix_tilde_inverse@r_next
        z_next_bar = matrix_tilde_inverse@r_next_bar
        b_k = np.dot(r_next_bar, z_next)/np.dot(r_k_bar, z_k)
        p_next = z_next + b_k * p_k
        p_next_bar = z_next_bar + b_k * p_k_bar
        x_k = x_k + a_k * p_k
        r_k = r_next
        r_k_bar = r_next_bar
        p_k = p_next
        p_k_bar = p_next_bar
        z_k = z_next
    return x_k
