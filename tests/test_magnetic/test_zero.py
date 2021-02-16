"""
Testing the zero field functions
"""

from tests.test_magnetic.base import *
from magnetic.zero import *


class ZeroFieldGyroFrequencyTestCase(BaseGyroFrequencyTestCase):
    gyro_frequency_calculations_file_name = os.path.join(
        THIS_DIR,
        "calculations/zero_field_mag_calculated.json"
    )

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        return calculate_gyro_frequency(*args, **kwargs)


class ZeroFieldFieldVecTestCase(BaseFieldVecTestCase):
    field_vec_calculations_file_name = os.path.join(
        THIS_DIR,
        "calculations/zero_field_vec_calculated.json"
    )

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        return calculate_magnetic_field_unit_vec(*args, **kwargs)
