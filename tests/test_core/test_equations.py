"""
Tests the equations making up our algorithm
"""
from unittest import TestCase
from tests.base import UtilityTestMixin
from typing import *
import json
import os
import numpy as np


class BaseEquationsTester(UtilityTestMixin):
    file_name = None
    file_dir = os.getcwd()

    @staticmethod
    def process_input(*inputs) -> Tuple:
        return inputs

    @staticmethod
    def process_output(*outputs) -> Tuple:
        return outputs

    def run_test_suite(self, inputs, expected_outputs):
        """
        We take a list of inputs 
        """
        processed_outputs = self.process_output(expected_outputs)

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

    def get_single_success_inputs_outputs(self, vectors, expected) -> List[Tuple[Any, ...]]:
        pass

    def load_file(self):
        with open(os.path.join(self.file_dir, self.file_name)) as f:
            return json.load(f)

    @staticmethod
    def split_test_group(test_group):
        inputs = [item["input"] for item in test_group]
        outputs = [item["output"] for item in test_group]
        return inputs, outputs

    def test_target_equation(self):
        if self.file_name is None:
            return

        test_sets = self.load_file()

        for group in test_sets:
            inputs, expected = self.split_test_group(test_sets[group])
            try:
                self.run_test_suite(inputs, expected)
            except Exception as e:
                print(f"Error when running trial: {group}")
                raise e


class TestEquation13(TestCase):
    def test_standard(self):
        self.fail()

    def test_zero_field(self):
        self.fail()
    
    def test_zero_atmosphere(self):
        self.fail()


class TestEquation13Prime(TestCase):
    def test_standard(self):
        self.fail()

    def test_zero_field(self):
        self.fail()
    
    def test_zero_atmosphere(self):
        self.fail()


class TestEquation14(TestCase):
    def test_standard(self):
        self.fail()

    def test_zero_field(self):
        self.fail()
    
    def test_zero_atmosphere(self):
        self.fail()


class TestEquation15(TestCase):
    def test_standard(self):
        self.fail()

    def test_zero_field(self):
        self.fail()
    
    def test_zero_atmosphere(self):
        self.fail()


class TestEquation16(TestCase):
    def test_standard(self):
        self.fail()

    def test_zero_field(self):
        self.fail()
    
    def test_zero_atmosphere(self):
        self.fail()
