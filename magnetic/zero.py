import numpy as np


def field_vec(position):
    return np.zeros((position.shape[0], 3))


def field_mag(position: np.ndarray):
    return np.zeros(position.shape[0])


def gyro_frequency(position):
    return np.zeros(position.shape[0])
