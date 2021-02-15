"""
Testing the zero field functions
"""

from tests.test_magnetic.base import BaseMagFieldTestCase
from magnetic.zero import *


class TestDipoleField(BaseMagFieldTestCase):
    gyro_frequency_calculations_file_name = "zero_gyro_frequency_calculated.json.json"
    field_vec_calculations_file_name = "zero_field_vec_calculated.json.json"

    def calculate_gyro_frequency(self, *args, **kwargs):
        return calculate_gyro_frequency(*args, **kwargs)

    def calculate_field_unit_vec(self, *args, **kwargs):
        return calculate_magnetic_field_unit_vec(*args, **kwargs)
