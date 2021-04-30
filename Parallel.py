"""
Utilities to help with parallel processing
"""
from multiprocessing import shared_memory
import multiprocessing as mp
from contextlib import contextmanager
import numpy as np
from numpy.typing import ArrayLike
import typing


@contextmanager
def create_parameters_shared_memory(
        path_parameters: typing.Tuple[ArrayLike, ArrayLike]) -> shared_memory.SharedMemory:
    """
    Writes the provided array to a shared memory file, and returns that shared memory file
    :param path_parameters: The numpy array to write
    :returns: The newly created shared memory
    """
    total_array = np.concatenate((np.atleast_1d(path_parameters[0].size), *path_parameters))
    memory = shared_memory.SharedMemory(
        create=True,
        size=total_array.nbytes
    )

    memory_array = np.ndarray(shape=total_array.shape, dtype=float, buffer=memory.buf)
    memory_array[:] = total_array[:]

    try:
        yield memory
    finally:
        memory.unlink()
        memory.close()


@contextmanager
def read_array_shared_memory(memory_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Writes the provided array to a shared memory file, and returns that shared memory file
    :param memory_name: The name of the memory to read from
    """
    memory = shared_memory.SharedMemory(name=memory_name, create=False)

    shape = memory.size // np.dtype(float).itemsize

    memory_array = np.ndarray(shape=(shape,), dtype=float, buffer=memory.buf)

    if abs(round(memory_array[0]) - memory_array[0]) > 1E-15:
        raise ValueError("First memory position needs to be an integer value representing the "
                         "radial parameter count")

    parameter_boundary = round(memory_array[0]) + 1
    radial_parameters = np.copy(memory_array[1:parameter_boundary])
    normal_parameters = np.copy(memory_array[parameter_boundary:])

    try:
        yield radial_parameters, normal_parameters
    finally:
        memory.close()
