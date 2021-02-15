"""
Testing the dipole moment field functions
"""

from tests.test_magnetic.base import BaseMagFieldTestCase
from magnetic.dipole import *


class TestDipoleField(BaseMagFieldTestCase):
    gyro_frequency_calculations_file_name = "dipole_gyro_frequency_calculated.json"
    field_vec_calculations_file_name = "dipole_field_vec_calculated.json"

    def calculate_gyro_frequency(self, *args, **kwargs):
        return calculate_gyro_frequency(*args, **kwargs)

    def calculate_field_unit_vec(self, *args, **kwargs):
        return calculate_magnetic_field_unit_vec(*args, **kwargs)
