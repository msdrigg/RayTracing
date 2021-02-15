"""
Testing the chapman layers test_atmosphere functions
"""

import os
from tests.base import BaseNumpyCalculationVerifier
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseGyroFrequencyTestCase(BaseNumpyCalculationVerifier):
    gyro_frequency_calculations_file_name = None

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented for base class")

    def test_gyro_frequency(self):
        if self.gyro_frequency_calculations_file_name is not None:
            self.run_suite_on_file(self.gyro_frequency_calculations_file_name)


class BaseFieldVecTestCase(BaseNumpyCalculationVerifier):
    field_vec_calculations_file_name = None

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented for base class")

    def test_field_vec(self):
        if self.field_vec_calculations_file_name is not None:
            self.run_suite_on_file(self.field_vec_calculations_file_name)
