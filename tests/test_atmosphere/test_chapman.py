"""
Testing the chapman layers test_atmosphere functions
"""

import os

from tests.test_atmosphere.base import BasePlasmaFrequencyTestCase
from atmosphere.chapman import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestPlasmaFrequencySquaredChapman(BasePlasmaFrequencyTestCase):
    plasma_frequency_calculations_file_name = os.path.join(THIS_DIR, "calculations/chapman_layers_calculations.json")

    def calculate_plasma_frequency_squared(self, *args, **kwargs):
        return calculate_plasma_frequency_squared(*args, **kwargs)
