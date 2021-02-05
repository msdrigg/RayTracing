from unittest import TestCase
import numpy as np
from utils import coordinates
import math


class TestCoordinates(TestCase):
    spherical_1 = np.array([coordinates.EARTH_RADIUS, 90 * math.pi / 180, 0])
    geographic_1 = np.array([0, 0, 0])
    cartesian_1 = np.array([coordinates.EARTH_RADIUS, 0, 0])
    r2 = coordinates.EARTH_RADIUS + 20
    spherical_2 = np.array([[coordinates.EARTH_RADIUS, 0, 0],
                            [r2, 210 * math.pi / 180, 11 * math.pi / 180]])
    geographic_2 = np.array([[-90, 0, 0], [120, 11, 20]])
    cartesian_2 = np.array([[0, 0, coordinates.EARTH_RADIUS],
                            [r2 * math.sin(210*math.pi/180)*math.cos(11*math.pi/180),
                             r2*math.sin(210*math.pi/180)*math.sin(11*math.pi/180),
                             r2*math.cos(210*math.pi/180)]])
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
        np.testing.assert_array_almost_equal(tested_func(self.cartesian_1),
                                             coordinates.regularize_spherical_coordinates(self.spherical_1))
        np.testing.assert_array_almost_equal(tested_func(self.cartesian_2),
                                             coordinates.regularize_spherical_coordinates(self.spherical_2))

    def test_spherical_to_cartesian(self):
        tested_func = coordinates.spherical_to_cartesian
        np.testing.assert_array_almost_equal(tested_func(self.spherical_1), self.cartesian_1)
        np.testing.assert_array_almost_equal(tested_func(self.spherical_2), self.cartesian_2)

    def test_regularize_spherical_coordinates(self):
        self.fail()

    def test_spherical_to_path_component(self):
        self.fail()

    def test_path_component_to_spherical(self):
        self.fail()
