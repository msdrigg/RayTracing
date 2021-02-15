"""
Testing the chapman layers test_atmosphere functions
"""

import json
from unittest import TestCase
from numpy import testing as np_test
import numpy as np
import os
from scipy import linalg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseMagFieldTestCase(TestCase):
    gyro_frequency_calculations_file_name = "zero_gyro_frequency_calculated.json.json"
    field_vec_calculations_file_name = "zero_field_vec_calculated.json.json"

    def calculate_gyro_frequency(self, *args, **kwargs):
        return self.fail("Not implemented")

    def calculate_field_unit_vec(self, *args, **kwargs):
        return self.fail("Not implemented")

    def run_single_test(self, vector_list, values):
        if vector_list.ndim == 1:
            np_test.assert_allclose(
                values,
                self.calculate_target_value(
                    vector_list, linalg.norm(vector_list)
                ),
                atol=1E-10
            )
            return
        norms = linalg.norm(vector_list, axis=1)
        nan_loc = np.isnan(values).flatten()
        null_values = vector_list[nan_loc]

        # Check for noisy failure
        for null_value in null_values[:, 0]:
            with self.assertWarns(RuntimeWarning):
                self.calculate_target_value(
                    0, null_value
                )

        # Check for correctness
        real_value_loc = np.logical_not(nan_loc)
        results = self.calculate_target_value(
            vector_list[real_value_loc], norms
        )

        np_test.assert_allclose(
            values[real_value_loc] / 1E6 ** 2,
            results / 1E6 ** 2,
            atol=1E-10
        )

    def test_group(self):
        if self.gyro_frequency_calculations_file_name == "":
            return
        with open(self.gyro_frequency_calculations_file_name) as f:
            tests = json.load(f)

        for group in tests:
            vecs = [item[0] for item in tests[group]]
            values = [item[1] for item in tests[group]]
            vector_array = np.array(vecs, dtype=float)
            value_array = np.array(values, dtype=float)

            self.run_single_test(vector_array, value_array)

            for i in range(0, vector_array.shape[0], 5):
                self.run_single_test(
                    vector_array[i].reshape(1, -1),
                    np.atleast_1d(value_array[i])
                )

            for i in range(0, vector_array.shape[0], 5):
                self.run_single_test(vector_array[i], value_array[i])
