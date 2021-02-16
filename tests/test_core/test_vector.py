"""
Testing functions from the vector class
"""
from unittest import TestCase
from core.vector import *
from tests import base
from numpy import testing as np_test
from core import coordinates as coords
import math
import numpy as np


class TestAngleBetweenVectorCollections(base.UtilityTestMixin):
    test_vector_1d_1 = np.arange(3)
    test_vector_1d_2 = np.zeros(32)
    test_vector_1d_3 = np.ones(3)
    angle_1d_expected = 0.6847192030022829138880980697
    test_vector_no_flatten_1 = np.zeros((2, 5))
    test_vector_no_flatten_2 = np.random.random((2, 2))
    test_vector_no_flatten_3 = np.random.random((2, 1))
    test_vector_flatten_1 = np.random.random((1, 5))
    test_vector_flatten_2 = np.random.random((1, 1))
    test_vector_flatten_3 = np.random.random((1, 1, 1))

    def test_90_degree_collections(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([0, 0, 1])
        np_test.assert_allclose(
            angle_between_vector_collections(vec1, vec2), math.pi/2
        )
        np_test.assert_allclose(
            angle_between_vector_collections(vec2, vec3), math.pi/2
        )
        np_test.assert_allclose(
            angle_between_vector_collections(vec1, vec3), math.pi/2
        )

    def test_random_degree_collections(self):
        for i in range(5):
            random_spherical_vec = np.random.random(3) * math.pi
            random_cartesian_vec = coords.spherical_to_cartesian(random_spherical_vec).squeeze()

            # Force the vector into the xy plane, so we can test the phi component
            random_xy_plane_vec = random_cartesian_vec.copy()
            random_xy_plane_vec[2] = 0
            z_vec = np.array([0, 0, 1])
            x_vec = np.array([1, 0, 0])
            np_test.assert_allclose(
                angle_between_vector_collections(random_cartesian_vec, z_vec),
                random_spherical_vec[1]
            )
            np_test.assert_allclose(
                angle_between_vector_collections(random_xy_plane_vec, x_vec),
                random_spherical_vec[2]
            )

    def test_1d_and_2d_mixed_vectors(self):
        for vec1, vec2, expected in [
            (self.test_vector_1d_3, self.test_vector_1d_3.reshape(1, -1), 0),
            (np.repeat(self.test_vector_1d_1[np.newaxis], 2, axis=0), self.test_vector_1d_3,  self.angle_1d_expected)
        ]:
            angle = angle_between_vector_collections(
                vec1,
                vec2
            )
            self.assertEqual(angle.size, max(vec1.size//vec1.shape[-1], vec2.size//vec2.shape[-1]))
            np_test.assert_allclose(
                angle, expected
            )

    def test_1d_only_vectors(self):
        angle = angle_between_vector_collections(
            self.test_vector_1d_1,
            self.test_vector_1d_3
        ).item()
        self.assert_is_close(angle, self.angle_1d_expected)

    def test_failure_on_unequal_length(self):
        with self.assertRaises(ValueError):
            angle_between_vector_collections(
                self.test_vector_1d_1,
                self.test_vector_1d_2
            )
            self.fail()

    def test_failure_on_faulty_dimension(self):
        for a, b in [(self.test_vector_1d_2, self.test_vector_no_flatten_2),
                     (self.test_vector_flatten_2, self.test_vector_no_flatten_3),
                     (self.test_vector_1d_1, self.test_vector_no_flatten_1)]:
            with self.assertRaises(ValueError):
                angle_between_vector_collections(
                    a, b
                )


class TestNormalizeRows(TestCase):
    def test_random_vectors(self):
        for i in range(10):
            spherical_vec = np.random.random((10, 3))
            cartesian_vec = coords.spherical_to_cartesian(spherical_vec)
            spherical_vecs_normalized = spherical_vec
            spherical_vecs_normalized[:, 0] = 1
            cartesian_vecs_normalized = coords.spherical_to_cartesian(spherical_vecs_normalized)
            np_test.assert_allclose(
                normalize_last_axis(cartesian_vec),
                cartesian_vecs_normalized
            )

    def test_single_vectors(self):
        for i in range(10):
            spherical_vec = np.random.random(3)
            cartesian_vec = coords.spherical_to_cartesian(spherical_vec)
            spherical_vecs_normalized = spherical_vec
            spherical_vecs_normalized[0] = 1
            cartesian_vecs_normalized = coords.spherical_to_cartesian(spherical_vecs_normalized)
            normalized_rows = normalize_last_axis(cartesian_vec)
            self.assertEqual(len(normalized_rows.squeeze().shape), 1)
            np_test.assert_allclose(
                normalized_rows,
                cartesian_vecs_normalized
            )

    def test_zero_vectors(self):
        with self.assertWarns(RuntimeWarning):
            zeros_normalized = normalize_last_axis(np.zeros(4))
        np.testing.assert_equal(
            zeros_normalized, np.zeros(4)
        )
        with self.assertWarns(RuntimeWarning):
            normalized_rows = normalize_last_axis(np.arange(10).reshape((10, 1)))
        expected = np.arange(10).reshape((10, 1))
        expected[1:, :] = 1
        np.testing.assert_equal(
            normalized_rows,
            expected
        )


class TestRowDotProduct(TestCase):
    vec1 = np.array([
        [1, 2, 3], [5, 5, 5], [2, 3, 4]
    ])
    vec2 = np.array([
        [1, 0, -1], [5, 2, 2], [2, 3, 0]
    ])
    vec3 = np.array([0, 1, 2])

    def test_against_norm(self):
        for vec in [self.vec1, self.vec2, self.vec3]:
            np.testing.assert_array_almost_equal(
                row_dot_product(
                    vec, vec
                ),
                np.square(np.linalg.norm(np.atleast_2d(vec), axis=1))
            )

    def test_commutative(self):
        for vec1, vec2 in [(self.vec1, self.vec2), (self.vec2, self.vec3)]:
            np.testing.assert_equal(
                row_dot_product(
                    vec1, vec2
                ),
                row_dot_product(
                    vec2, vec1
                )
            )

    def test_mixed_dimensions(self):
        for vec_triple in [
            (np.array([-2, 45, 13]), self.vec1, self.vec2),
            (np.array([-2, 6, 3]), self.vec3, self.vec2),
            (np.array([8, 15, 11]), self.vec1, self.vec3)
        ]:
            np.testing.assert_equal(
                row_dot_product(
                    vec_triple[1], vec_triple[2]
                ),
                vec_triple[0]
            )

    def test_single_dimension(self):
        for vec1, vec2, ans in [
            (np.ones(3), np.ones(3), 3),
            (np.zeros(5), np.zeros(5), 0),
            (np.arange(2), np.arange(2), 1),
            (np.ones(1), np.ones(1), 1)
        ]:
            product = row_dot_product(
                vec1, vec2
            )
            self.assertEqual(product.size, 1)
            self.assertEqual(len(product.shape), 1)
            np.testing.assert_equal(
                product,
                ans
            )
