from unittest import TestCase
import numpy as np
from utils.vector import *


class TestFlattenIfNecessary(TestCase):
    def test_array_no_flatten(self):
        self.fail()

    def test_array_flatten(self):
        self.fail()

    def test_1d_array(self):
        self.fail()


class TestAngleBetweenVectorCollections(TestCase):
    def test_90_degree_collections(self):
        self.fail()

    def test_random_degree_collections(self):
        self.fail()

    def test_1d_and_2d_mixed_vectors(self):
        self.fail()

    def test_1d_only_vectors(self):
        self.fail()


class TestNormalizeRows(TestCase):
    def test_random_vectors(self):
        self.fail()

    def test_single_vectors(self):
        self.fail()

    def test_zero_vectors(self):
        self.fail()


class TestHelpers(TestCase):
    vec1 = np.array([
        [1, 2, 3], [5, 5, 5], [2, 3, 4]
    ])
    vec2 = np.array([
        [1, 0, -1], [5, 2, 2], [2, 3, 0]
    ])
    vec3 = np.array([0, 1, 2])

    def test_row_dot_product(self):
        for vec in [self.vec1, self.vec2, self.vec3]:
            np.testing.assert_array_almost_equal(
                row_dot_product(
                    vec, vec
                ),
                np.square(np.linalg.norm(np.atleast_2d(vec), axis=1))
            )
        for vec_triple in [
            (np.array([-2, 45, 13]), self.vec1, self.vec2),
            (np.array([-2, 6, 3]), self.vec3, self.vec2)
        ]:
            np.testing.assert_array_almost_equal(
                row_dot_product(
                    vec_triple[1], vec_triple[2]
                ),
                vec_triple[0]
            )


class TestNormalizeSingleRow(TestCase):
    def test_random_vectors(self):
        self.fail()
