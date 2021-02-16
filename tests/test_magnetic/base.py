"""
Testing the chapman layers test_atmosphere functions
"""

import os
from tests.base import BaseNumpyCalculationVerifier
import numpy as np
from typing import *
from scipy import linalg
from core.constants import B_FACTOR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseGyroFrequencyTestCase(BaseNumpyCalculationVerifier):
    gyro_frequency_calculations_file_name = None

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented for base class")

    def get_vector_success_inputs_outputs(self, vectors, expected) -> Tuple:
        nan_loc = np.isnan(expected)
        outputs = vectors[~nan_loc]
        reformatted = outputs
        return (reformatted, linalg.norm(reformatted, axis=-1).flatten()), expected[~nan_loc]

    def get_single_success_inputs_outputs(self, vectors, expected) -> List[Tuple[Any, ...]]:
        nan_loc = np.isnan(expected)
        successes = vectors[~nan_loc]
        inputs = []
        outputs = []
        for i in range(successes.shape[0]):
            inputs.append((successes[i], np.asarray(linalg.norm(successes[i]))))
            outputs.append(expected[i])
        return list(zip(inputs, outputs))

    def get_single_failures(self, vectors, expected) -> List:
        nan_loc = np.isnan(expected)
        failures = vectors[nan_loc]
        outputs = []
        for i in range(failures.shape[0]):
            outputs.append((failures[i], linalg.norm(failures[i])))
        return outputs

    def get_vector_failures(self, vectors, expected) -> Tuple:
        nan_loc = np.isnan(expected)
        successes = vectors[nan_loc]
        return successes, linalg.norm(successes, axis=1).flatten()

    def get_params(self) -> Tuple:
        return ()

    @staticmethod
    def format_expected(vecs):
        return vecs * B_FACTOR

    def test_gyro_frequency(self):
        if self.gyro_frequency_calculations_file_name is not None:
            self.run_suite_on_file(self.gyro_frequency_calculations_file_name)


class BaseFieldVecTestCase(BaseNumpyCalculationVerifier):
    field_vec_calculations_file_name = None

    # noinspection PyTypeChecker
    def calculate_target_value(self, *args, **kwargs) -> np.ndarray:
        self.fail("Not implemented for base class")

    def get_vector_success_inputs_outputs(self, vectors, expected) -> Tuple:
        nan_loc = np.any(np.isnan(expected), axis=-1)
        outputs = vectors[~nan_loc]
        return (outputs, linalg.norm(outputs, axis=-1)), expected[~nan_loc].reshape(outputs.shape)

    def get_single_success_inputs_outputs(self, vectors, expected) -> List[Tuple[Any, ...]]:
        nan_loc = np.any(np.isnan(expected), axis=-1)
        successes = vectors[~nan_loc]
        inputs = []
        outputs = []
        for i in range(successes.shape[0]):
            inputs.append((successes[i], np.asarray(linalg.norm(successes[i]))))
            outputs.append(expected[i])
        return list(zip(inputs, outputs))

    def get_single_failures(self, vectors, expected) -> List:
        nan_loc = np.any(np.isnan(expected), axis=-1)
        failures = vectors[nan_loc]
        outputs = []
        for i in range(failures.shape[0]):
            outputs.append((failures[i], linalg.norm(failures[i])))
        return outputs

    def get_vector_failures(self, vectors, expected) -> Tuple:
        nan_loc = np.isnan(expected)
        successes = vectors[nan_loc]
        return successes, linalg.norm(successes, axis=-1)

    def get_params(self) -> Tuple:
        return ()

    def test_field_vec(self):
        if self.field_vec_calculations_file_name is not None:
            self.run_suite_on_file(self.field_vec_calculations_file_name)
