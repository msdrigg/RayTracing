from abc import abstractmethod, ABC
from Constants import B_FACTOR as _B_FACTOR
import numpy as np


class BaseField(ABC):
    @abstractmethod
    def field_vec(self, position: np.ndarray) -> np.ndarray:
        """
        This function returns a NORMALIZED magnetic field vector using the given position vector
        This is a vectorized function, so position may be either a single position vector, or an array whose
        rows are position vectors

        :param position : np.ndarray, shape (N, 3) or (3, )
            3-vec or array of 3-vecs in a cartesian coordinate system
        :returns The normalized magnetic field cooresponding to the position vector. Its shape should match 
        the shape of the position input
        """
        raise NotImplementedError("Inheriting classes must override the field_vec function")

    @abstractmethod
    def field_mag(self, position: np.ndarray) -> np.ndarray:
        """
        This function returns the magnitude of the magnetic field given the parameters described above

        :param position : np.ndarray, shape (N, 3) or (3, )
            3-vec or array of 3-vecs in a cartesian coordinate system
        :returns The magnetic field magnitude cooresponding to the position vector. Its shape should match
        the incoming position vector's shape. If position is of shape (N, 3), the output should be 
        of shape (N, ). If the position is of shape (3, ), the output should be a scalar.
        """
        raise NotImplementedError("Inheriting classes must override the field_mag function")

    def gyro_frequency(self, position: np.ndarray) -> np.ndarray:
        """
        This function returns the gyro frequency at the given point using the class's defined functions for
        b_factor and field_mag

        :param position : np.ndarray, shape (N, 3) or (3, )
            3-vec or array of 3-vecs in a cartesian coordinate system
        :returns The gyro frequency cooresponding to the position vector. Its shape should match
        the incoming position vector's shape. If position is of shape (N, 3), the output should be 
        of shape (N, ). If the position is of shape (3, ), the output should be a scalar.
        """

        # Standard implementation. This does not have to be overridden 
        # as long as you override field_mag
        return _B_FACTOR * self.field_mag(position)

