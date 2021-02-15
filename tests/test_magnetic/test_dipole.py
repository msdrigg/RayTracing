"""
Testing the dipole moment field functions
"""

from tests.test_magnetic.base import *
from magnetic.dipole import *


class DipoleGyroFrequencyTestCase(BaseGyroFrequencyTestCase):
    gyro_frequency_calculations_file_name = "calculations/dipole_gyro_frequency_calculated.json"

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        return calculate_gyro_frequency(*args, **kwargs)


class DipoleFieldVecTestCase(BaseFieldVecTestCase):
    field_vec_calculations_file_name = "calculations/dipole_field_vec_calculated.json"

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        return calculate_magnetic_field_unit_vec(*args, **kwargs)
