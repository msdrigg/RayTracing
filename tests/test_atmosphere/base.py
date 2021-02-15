"""
Testing the chapman layers test_atmosphere functions
"""

import json
from unittest import TestCase
from numpy import testing as np_test
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class BasePlasmaFrequencyTestCase(TestCase):
    plasma_frequency_calculations_file_name = None
    atmosphere_parameters_file_name = os.path.join(
        THIS_DIR,
        "calculations/atmosphere_parameter_definitions.json"
    )

    # noinspection PyTypeChecker
    def calculate_plasma_frequency_squared(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented")

    def verify_single_calculation(self, norm_value_list, params):
        if np.isnan(norm_value_list[1]):
            with self.assertWarns(RuntimeWarning):
                self.calculate_plasma_frequency_squared(
                    0, norm_value_list[0], *params
                )
            return

        np_test.assert_allclose(
            norm_value_list[1] / 1E6 ** 2,
            self.calculate_plasma_frequency_squared(
                0, norm_value_list[0], *params
            ) / 1E6 ** 2,
            atol=1E-10
        )

    def verify_vectorized_calculation(self, norm_value_list, params):
        nan_loc = np.isnan(norm_value_list[:, 1])
        null_values = norm_value_list[nan_loc]

        # Check for noisy failure
        for value in null_values[:, 0]:
            with self.assertWarns(RuntimeWarning):
                self.calculate_plasma_frequency_squared(
                    0, value, *params
                )

        # Check for correctness
        real_value_loc = np.logical_not(nan_loc)
        real_values = norm_value_list[real_value_loc]
        results = self.calculate_plasma_frequency_squared(
            0, real_values[:, 0], *params
        )
        np_test.assert_allclose(
            real_values[:, 1] / 1E6 ** 2,
            results / 1E6 ** 2,
            atol=1E-10
        )

    def test_group(self):
        if self.plasma_frequency_calculations_file_name is None:
            return

        with open(self.plasma_frequency_calculations_file_name) as f:
            tests = json.load(f)
        with open(self.atmosphere_parameters_file_name) as f:
            group_list = json.load(f)

        for group in group_list:
            params = group_list[group]
            norm_value_list = np.array(tests[group], dtype=float)
            self.verify_vectorized_calculation(norm_value_list, params)

            for i in range(0, norm_value_list.shape[0], 5):
                self.verify_vectorized_calculation(norm_value_list[i].reshape(-1, 2), params)

            for i in range(0, norm_value_list.shape[0], 5):
                self.verify_single_calculation(norm_value_list[i], params)
