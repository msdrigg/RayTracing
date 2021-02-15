"""
Testing the quasi parabolic test_atmosphere functions
"""

import os

from tests.test_atmosphere.base import BasePlasmaFrequencyTestCase
from atmosphere.quasi_parabolic import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestPlasmaFrequencySquaredQPInAtmosphere(BasePlasmaFrequencyTestCase):
    plasma_frequency_calculations_file_name = os.path.join(THIS_DIR, "quasi_parabolic_in_atmosphere_calculations.json")

    def calculate_plasma_frequency_squared(self, *args, **kwargs):
        return calculate_plasma_frequency_squared(*args, **kwargs)


class TestPlasmaFrequencySquaredQPAboveAtmosphere(BasePlasmaFrequencyTestCase):
    plasma_frequency_calculations_file_name = os.path.join(THIS_DIR, "quasi_parabolic_above_atmosphere_calculations.json")

    def calculate_plasma_frequency_squared(self, *args, **kwargs):
        return calculate_plasma_frequency_squared(*args, **kwargs)


class TestPlasmaFrequencySquaredQPBelowAtmosphere(BasePlasmaFrequencyTestCase):
    plasma_frequency_calculations_file_name = os.path.join(THIS_DIR, "quasi_parabolic_below_atmosphere_calculations.json")

    def calculate_plasma_frequency_squared(self, *args, **kwargs):
        return calculate_plasma_frequency_squared(*args, **kwargs)
