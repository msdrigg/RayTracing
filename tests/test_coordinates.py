from unittest import TestCase
from utils import coordinates
import math
import numpy as np


class TestCoordinates(TestCase):
    spherical_1 = np.array([coordinates.EARTH_RADIUS, 90 * math.pi / 180, 0])
    geographic_1 = np.array([0, 0, 0])
    cartesian_1 = np.array([coordinates.EARTH_RADIUS, 0, 0])
    r2 = coordinates.EARTH_RADIUS + 20
    spherical_2 = np.array([[coordinates.EARTH_RADIUS, 0, 0],
                            [r2, 210 * math.pi / 180, 11 * math.pi / 180]])
    geographic_2 = np.array([[-90, 0, 0], [120, 11, 20]])
    cartesian_2 = np.array([[0, 0, coordinates.EARTH_RADIUS],
                            [r2 * math.sin(210 * math.pi / 180) * math.cos(11 * math.pi / 180),
                             r2 * math.sin(210 * math.pi / 180) * math.sin(11 * math.pi / 180),
                             r2 * math.cos(210 * math.pi / 180)]])
    spherical_fail = np.array([-1, 0, 0])
    geographic_fail = np.array([-90, 0, - coordinates.EARTH_RADIUS - 1])

    def test_spherical_to_geographic(self):
        tested_func = coordinates.spherical_to_geographic
        np.testing.assert_array_almost_equal(tested_func(self.spherical_1), self.geographic_1)
        np.testing.assert_array_almost_equal(tested_func(self.spherical_2), self.geographic_2)
        with self.assertWarns(RuntimeWarning):
            tested_func(self.spherical_fail)

    def test_geographic_to_spherical(self):
        tested_func = coordinates.geographic_to_spherical
        np.testing.assert_array_almost_equal(tested_func(self.geographic_1), self.spherical_1)
        np.testing.assert_array_almost_equal(tested_func(self.geographic_2), self.spherical_2)
        with self.assertWarns(RuntimeWarning):
            tested_func(self.geographic_fail)

    def test_cartesian_to_spherical(self):
        tested_func = coordinates.cartesian_to_spherical
        regularize = coordinates.regularize_spherical_coordinates
        np.testing.assert_array_almost_equal(regularize(tested_func(self.cartesian_1)),
                                             regularize(self.spherical_1))
        np.testing.assert_array_almost_equal(regularize(tested_func(self.cartesian_2)),
                                             regularize(self.spherical_2))

    def test_spherical_to_cartesian(self):
        tested_func = coordinates.spherical_to_cartesian
        np.testing.assert_array_almost_equal(tested_func(self.spherical_1), self.cartesian_1)
        np.testing.assert_array_almost_equal(tested_func(self.spherical_2), self.cartesian_2)

    def test_regularize_spherical_coordinates(self):
        for test_coord in [self.spherical_1, self.spherical_2]:
            regularized = coordinates.regularize_spherical_coordinates(
                test_coord)
            np.testing.assert_array_less(np.atleast_2d(regularized)[:, 1], math.pi)
            np.testing.assert_array_less(np.atleast_2d(regularized)[:, 2], 2 * math.pi)
            np.testing.assert_array_compare(np.greater_equal, regularized, 0)
            np.testing.assert_array_almost_equal(coordinates.spherical_to_cartesian(regularized),
                                                 coordinates.spherical_to_cartesian(test_coord))

    path_start_1 = np.array([coordinates.EARTH_RADIUS, math.pi / 2, math.pi / 2])
    path_end_1 = np.array([coordinates.EARTH_RADIUS, math.pi / 2, math.pi])

    path_start_2 = np.array([coordinates.EARTH_RADIUS, 135 * math.pi / 180, math.pi / 2])
    path_end_2 = np.array([coordinates.EARTH_RADIUS + 20, math.pi / 4, 0])

    spherical_3 = np.array([
        8.108399138e6, 1.570795217, 2.356122283
    ])
    spherical_4 = np.array([
        3.678639548e6, 1.991263898, 1.106910344
    ])
    path_1_3 = np.array([
        1.7373991378634660797e6, 5.0033116672347954188e6, 7.0715561758989078771
    ])
    path_2_3 = np.array([
        1.7373991378634660797e6, -3.3350509996894406438e6, -6.086329222139008581e6
    ])
    path_1_4 = np.array([
        -2.6923604523741430244e6, -2.9554175961958576657e6, -2.6787988957139253362e6
    ])
    path_2_4 = np.array([
        -2.6923604523741430244e6, 3.3370540612300174823e6, 804.92538460834365228
    ])
    coords_paths_paths = [
        (spherical_3, (path_1_3, path_start_1, path_end_1)),
        (spherical_3, (path_2_3, path_start_2, path_end_2)),
        (spherical_4, (path_1_4, path_start_1, path_end_1)),
        (spherical_4, (path_2_4, path_start_2, path_end_2)),
    ]

    def test_reversible_spherical_path_vectorized(self):
        vecs = np.vstack((self.spherical_3, self.spherical_4, self.spherical_1))

        point_path_pairs = [
            (
                np.vstack((self.spherical_4, self.spherical_3)),
                np.vstack((self.path_1_4, self.path_1_3)),
                (self.path_start_1, self.path_end_1)
            ),
            (
                np.vstack((self.spherical_4, self.spherical_3)),
                np.vstack((self.path_2_4, self.path_2_3)),
                (self.path_start_2, self.path_end_2)
            ),
            (
                np.vstack((self.spherical_3, self.spherical_4)),
                np.vstack((self.path_1_3, self.path_1_4)),
                (self.path_start_1, self.path_end_1)
            ),
            (
                np.vstack((self.spherical_4, self.spherical_3)),
                np.vstack((self.path_2_4, self.path_2_3)),
                (self.path_start_2, self.path_end_2)
            )
        ]
        for spherical, path_comp, path in point_path_pairs:
            np.testing.assert_allclose(
                coordinates.spherical_to_path_component(
                    spherical, *path
                ),
                path_comp,
                rtol=1e-7, atol=0.01
            )

            np.testing.assert_allclose(
                coordinates.spherical_to_cartesian(coordinates.path_component_to_spherical(
                    path_comp, *path
                )),
                coordinates.spherical_to_cartesian(
                    spherical
                ),
                rtol=1e-7, atol=0.01
            )

        for path in [(self.path_start_2, self.path_end_2), (self.path_start_1, self.path_end_1)]:
            path_comp = coordinates.spherical_to_path_component(vecs, path[0], path[1])
            washed_coordinate = coordinates.path_component_to_spherical(path_comp, path[0], path[1])

            np.testing.assert_allclose(
                coordinates.spherical_to_cartesian(washed_coordinate),
                coordinates.spherical_to_cartesian(vecs),
                rtol=1e-7, atol=0.01
            )

    def test_spherical_to_path_component(self):
        for point_path_pair in self.coords_paths_paths:
            path = point_path_pair[1]
            for coordinate in [self.spherical_1, self.spherical_3, self.spherical_4, self.spherical_2[0], self.spherical_2[1]]:
                path_comp = coordinates.spherical_to_path_component(coordinate, path[1], path[2])
                washed_coordinate = coordinates.path_component_to_spherical(path_comp, path[1], path[2])

                np.testing.assert_allclose(
                    coordinates.spherical_to_cartesian(washed_coordinate),
                    coordinates.spherical_to_cartesian(coordinate),
                    rtol=1e-9, atol=0.01
                )

            np.testing.assert_allclose(
                path[0],
                coordinates.spherical_to_path_component(point_path_pair[0], path[1], path[2]),
                rtol=1e-7, atol=0.01
            )

    def test_path_component_to_spherical(self):
        for point_path_pair in self.coords_paths_paths:
            path = point_path_pair[1]

            spherical = coordinates.path_component_to_spherical(*path)
            np.testing.assert_allclose(
                point_path_pair[0],
                coordinates.regularize_spherical_coordinates(spherical)
            )


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
                coordinates.row_dot_product(
                    vec, vec
                ),
                np.square(np.linalg.norm(np.atleast_2d(vec), axis=1))
            )
        for vec_triple in [
            (np.array([-2, 45, 13]), self.vec1, self.vec2),
            (np.array([-2, 6, 3]), self.vec3, self.vec2)
        ]:
            np.testing.assert_array_almost_equal(
                coordinates.row_dot_product(
                    vec_triple[1], vec_triple[2]
                ),
                vec_triple[0]
            )
