"""
General testing core
"""
import json
import math
from unittest import TestCase

import numpy as np
import numpy.testing as np_test
from typing import *


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


class BaseNumpyCalculationVerifier(UtilityTestMixin):
    default_atol = 1E-10
    default_rtol = 1E-9

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented in base class")

    # noinspection PyTypeChecker
    def get_vector_success_inputs_outputs(self, vectors, expected) -> Tuple:
        self.fail("Not implemented in base class")

    # noinspection PyTypeChecker
    def get_single_success_inputs_outputs(self, vectors, expected) -> List[Tuple[Any, ...]]:
        self.fail("Not implemented in base class")

    # noinspection PyTypeChecker
    def get_single_failures(self, vectors, expected) -> Tuple:
        self.fail("Not implemented in base class")

    # noinspection PyTypeChecker
    def get_vector_failures(self, vectors, expected) -> Tuple:
        self.fail("Not implemented in base class")

    # noinspection PyTypeChecker
    def get_params(self) -> Tuple:
        return ()

    def verify_single_calculation_success(self, inputs, expected):
        np_test.assert_allclose(
            expected,
            self.calculate_target_value(
                *inputs, *self.get_params()
            ),
            atol=self.default_atol,
            rtol=self.default_rtol
        )

    def verify_failure(self, inputs):
        # Check overall for noisy warning
        with self.assertWarns(RuntimeWarning):
            self.calculate_target_value(
                *inputs, *self.get_params()
            )

    def verify_vectorized_calculation_success(self, inputs, expected):
        """
        This function checks a list of inputs
        """
        results = self.calculate_target_value(
            *inputs, *self.get_params()
        )
        np_test.assert_allclose(
            expected,
            results,
            atol=self.default_atol,
            rtol=self.default_rtol
        )
    
    @staticmethod
    def format_expected(vecs):
        return vecs
    
    def run_test_suite(self, vectors, expected):
        processed_outputs = self.format_expected(expected)
        self.verify_vectorized_calculation_success(
            *self.get_vector_success_inputs_outputs(vectors, processed_outputs)
        )

        failure_vecs = self.get_vector_failures(vectors, processed_outputs)
        if len(failure_vecs) > 0 and failure_vecs[0].size > 0:
            self.verify_failure(
                failure_vecs
            )

        for inputs in self.get_single_failures(vectors, processed_outputs):
            self.verify_failure([inputs[i].reshape(1, -1) for i in range(len(inputs))])

        for vec in self.get_single_failures(vectors, processed_outputs):
            self.verify_single_calculation_success(vec, processed_outputs)

        for inputs, outputs in self.get_single_success_inputs_outputs(vectors, processed_outputs):
            new_inputs = list(inputs)
            new_inputs[0] = inputs[0].reshape(1, -1)
            self.verify_vectorized_calculation_success(
                new_inputs,
                outputs[np.newaxis]
            )

        for inputs, outputs in self.get_single_success_inputs_outputs(vectors, processed_outputs):
            self.verify_single_calculation_success(inputs, outputs)

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

            self.run_test_suite(vectors, calculations)
