"""
Tests the equations making up our algorithm
"""
from tests.base import UtilityTestMixin, get_directory_path
from typing import *
import json
import os
import numpy as np
from core import equations


class BaseEquationsTester(UtilityTestMixin):
    file_name = None
    file_dir = get_directory_path(__file__)

    @staticmethod
    def process_input(*inputs) -> Tuple:
        return inputs

    @staticmethod
    def process_output(*outputs) -> Tuple:
        return outputs

    @staticmethod
    def get_calculation(*inputs) -> Union[np.ndarray, float]:
        return 0

    def verify_failure(self, *inputs):
        with self.assertWarns(RuntimeWarning):
            self.get_calculation(inputs)

    def run_test_suite(self, inputs, expected_outputs):
        """
        We take a list of inputs 
        """
        for input_variables_preprocessed, expected_preprocessed in zip(inputs, expected_outputs):
            inputs_processed = self.process_input(*input_variables_preprocessed)
            expected_output = self.process_output(expected_preprocessed)

            if expected_output is not None:
                print("Next iteration")
                print(inputs_processed)
                print(expected_output)
                calculation = self.get_calculation(inputs_processed)
                print(calculation)
                np.testing.assert_allclose(
                    calculation,
                    expected_output
                )
                calculation = self.get_calculation([np.array(variable) for variable in inputs_processed])
                np.testing.assert_allclose(
                    calculation,
                    np.ndarray(expected_output)
                )
            else:
                self.verify_failure(inputs_processed)
                self.verify_failure([np.array(variable) for variable in inputs_processed])

        input_variables_vectorized = np.array(
            [self.process_input(*inputs_outputs_combined[0]) for inputs_outputs_combined in
             zip(inputs, expected_outputs) if inputs_outputs_combined[1] is not None]
        ).T

        expected_output_vectorized = np.array(
            [self.process_output(expected_output) for expected_output in expected_outputs
             if expected_output is not None]
        ).T

        np.testing.assert_allclose(
            self.get_calculation(*input_variables_vectorized),
            expected_output_vectorized
        )

        input_failure_vectorized = np.array(
            [self.process_input(*inputs_outputs_combined[0]) for inputs_outputs_combined in
             zip(inputs, expected_outputs) if inputs_outputs_combined[1] is None]
        ).T

        if input_failure_vectorized.size > 0:
            self.verify_failure(*input_failure_vectorized)

    def load_data(self):
        with open(os.path.join(self.file_dir, self.file_name)) as f:
            return json.load(f)

    @staticmethod
    def split_test_group(test_group):
        inputs = [item[0] for item in test_group]
        outputs = [item[1] for item in test_group]
        return inputs, outputs

    def test_target_equation(self):
        if self.file_name is not None:
            test_sets = self.load_data()

            for group in test_sets:
                inputs, expected = self.split_test_group(test_sets[group])
                try:
                    self.run_test_suite(inputs, expected)
                except Exception as e:
                    print(f"Error when running trial: {group}")
                    raise e


class TestEquation13(BaseEquationsTester):
    file_name = r'calculations/calculate_eq_13_results.json'

    @staticmethod
    def process_input(*inputs) -> Tuple:
        x, y, yp, yt = inputs
        return yp, x, y**2, yt

    @staticmethod
    def get_calculation(inputs) -> Union[np.ndarray, float]:
        return equations.equation_13(*inputs)


class TestEquation14(BaseEquationsTester):
    file_name = r'calculations/calculate_eq_14_results.json'

    @staticmethod
    def process_input(*inputs) -> Tuple:
        x, y, yp, yt = inputs
        fractions = equations.equation_16(
            yp, yp**2, x, y**2
        )
        return yp, yp**2, y**2, fractions, yt
    
    @staticmethod
    def get_calculation(inputs) -> Union[np.ndarray, float]:
        return equations.equation_14(*inputs)


class TestEquation15(BaseEquationsTester):
    file_name = r'calculations/calculate_eq_15_results.json'

    @staticmethod
    def process_input(*inputs) -> Tuple:
        x, y, yp = inputs
        return yp ** 2, x, y ** 2

    @staticmethod
    def get_calculation(inputs) -> Union[np.ndarray, float]:
        return equations.equation_15(*inputs)


class TestEquation16(BaseEquationsTester):
    file_name = r'calculations/calculate_eq_16_results.json'

    @staticmethod
    def process_input(*inputs) -> Tuple:
        x, y, yp = inputs
        return yp, yp ** 2, x, y ** 2

    @staticmethod
    def get_calculation(inputs) -> Union[np.ndarray, float]:
        return equations.equation_16(*inputs)


class TestCalculateYP(BaseEquationsTester):
    file_name = r'calculations/calculate_yp_results.json'

    @staticmethod
    def process_input(*inputs) -> Tuple:
        x, y, yt = inputs
        return x, y, y ** 2, yt

    @staticmethod
    def get_calculation(inputs) -> Union[np.ndarray, float]:
        return equations.calculate_yp(*inputs)
        

# class TestCalculateA(TestCase):
#     file_name = r'calculations/calculate_a_results.json'
    
#     def get_calculation(self, inputs) -> Union[np.ndarray, float]:
#         return equations.calculate_a(*inputs)


# class TestCalculateB(TestCase):
#     file_name = r'calculations/calculate_b_results.json'
    
#     def get_calculation(self, inputs) -> Union[np.ndarray, float]:
#         return equations.calculate_b(*inputs)


# class TestCalculateC(TestCase):
#     file_name = r'calculations/calculate_c_results.json'
    
#     def get_calculation(self, inputs) -> Union[np.ndarray, float]:
#         return equations.calculate_c(*inputs)
