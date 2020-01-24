import unittest
import numpy as np
import hyperdual
from Atmosphere import ChapmanLayers
from Field import BasicField
import Vector
from Constants import EARTH_RADIUS, PI


FIELD = BasicField()
initial = Vector.spherical_to_cartesian(
    Vector.latitude_to_spherical(
        np.array([EARTH_RADIUS, 90 + 23.5, 133.7])))
final = Vector.spherical_to_cartesian(
    Vector.latitude_to_spherical(
        np.array([EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
ATMOSPHERE = ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / PI, -1), initial)
FREQUENCY = 16E6

def mu2(pos):
    x = ATMOSPHERE.plasma_frequency(pos, using_spherical=False)/FREQUENCY
    y = FIELD.field_vec(pos)
    p = solve_for_p()
    yp = np.dot(y, p)
    y2 = np.square(FIELD.field_mag(pos))
    yp2 = np.square(yp)
    output = 1 - (2 * (-1 + x) * x) / (-2 + 2 * x + y2 + yp2 -
                                       np.sqrt(4 * np.square((-1 + x)) * yp2 + np.square((y2 - yp2))))
    return output


def dmu2(pos):
    x = ATMOSPHERE.plasma_frequency(pos, using_spherical=False)/FREQUENCY
    y = FIELD.field_vec(pos)
    p = solve_for_p()
    yp = np.dot(y, p)
    y2 = np.square(FIELD.field_mag(pos))
    yp2 = np.square(yp)
    output = (4 * (1 - x) * x * yp * (-1 + (2 + 2 * (-2 + x) * x - y2 + yp2) /
                                    np.sqrt(4 * np.square((-1 + x)) * yp2 + np.square((y2 - yp2))))) / \
        np.square((-2 + 2 * x + y2 + yp2 - np.sqrt(4 * np.square((-1 + x)) * yp2 + np.square((y2 - yp2)))))
    return output


TEST_FUNCTIONS = [mu2]
TEST_DERIVATIVES = [dmu2]
TEST_POSITIONS = []


class MyTestCase(unittest.TestCase):
    def test_hyperduals(self, args=None):
        if args is None:
            args = TEST_FUNCTIONS, TEST_DERIVATIVES, TEST_POSITIONS
        for func, der in zip(args[0], args[1]):
            for pos in args[2]:
                hyperdual = np.hyperdual(pos, np.sqrt(np.finfo(float).eps)*pos, np.sqrt(np.finfo(float).eps)*pos, 0)
                self.assertAlmostEqual(func(pos).real, der(pos))


    def test_solver(self):
        pass

    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
