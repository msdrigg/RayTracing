"""
General testing core
"""
import json
import math
from unittest import TestCase

import numpy as np
import numpy.testing as np_test
from scipy import linalg


class UtilityTestMixin(TestCase):
    def assert_is_close(self, num1: float, num2: float, rel_tol: float = 1E-9, abs_tol: float = 0.0):
        """
        Asserts that math.isclose returns true
        """
        self.assertTrue(
            math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol),
            f"Numbers are not within expected tolerance: "
                                 f"abs_tol={rel_tol}, rel_tol={rel_tol}\n"
                                 f"Number 1: {num1}, \nNumber 2: {num2}"
        )


class BaseNumpyCalculationVerifier(TestCase):
    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented in base class")

    def verify_single_calculation_success(self, inputs, expected, *args, **kwargs):
        if np.any(np.isnan(expected)):
            with self.assertWarns(RuntimeWarning):
                self.calculate_target_value(
                    0, inputs, *args, **kwargs
                )
            return

        np_test.assert_allclose(
            expected / 1E6 ** 2,
            self.calculate_target_value(
                0, inputs, *args, **kwargs
            ) / 1E6 ** 2,
            atol=1E-10
        )

    def verify_failure(self, *args, **kwargs):
        # Check overall for noisy warning
        with self.assertWarns(RuntimeWarning):
            self.calculate_target_value(
                *args, **kwargs
            )

    def verify_vectorized_calculation_success(self, inputs, expected, *args, **kwargs):
        """
        This function checks a list of inputs
        """
        nan_locations = np.isnan(expected)

        # Check for correctness
        real_value_loc = np.logical_not(nan_locations)
        real_inputs = inputs[real_value_loc]
        real_outputs = expected[real_value_loc]

        results = self.calculate_target_value(
            0, real_inputs, *args, **kwargs
        )
        np_test.assert_allclose(
            real_outputs / 1E6 ** 2,
            results / 1E6 ** 2,
            atol=1E-10
        )

    def run_test_suite(self, vectors, norms, expected):
        self.verify_vectorized_calculation_success(vectors, norms, expected)

        # Check individual for noisy warning
        for value in inputs:
            with self.assertWarns(RuntimeWarning):
                self.calculate_target_value(
                    0, value, *args, **kwargs
                )

        for i in range(0, vectors.shape[0], 5):
            self.verify_vectorized_calculation_success(vectors[i].reshape(1, -1), norms, expected)

        for i in range(0, vectors.shape[0], 5):
            self.verify_single_calculation_success(vectors[i], norms, expected)

    @staticmethod
    def split_test_group(test_group):
        vecs = [item[0] for item in test_group]
        values = [item[1] for item in test_group]
        vector_array = np.array(vecs, dtype=float)
        value_array = np.array(values, dtype=float)
        return vector_array, value_array

    def run_suite_on_file(self, filename):
        if filename is None:
            return

        with open(filename) as f:
            tests = json.load(f)

        for group in tests:
            vectors, calculations = self.split_test_group(tests[group])
            norms = linalg.norm(vectors, axis=1)

            self.run_test_suite(vectors, None, calculations)
            self.run_test_suite(vectors, norms, calculations)
