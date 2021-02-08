from unittest import TestCase
from utils import coordinates as coords
from utils import testing
import math
from initialize import quasi_parabolic as qp
import numpy as np
from matplotlib import pyplot as plt


class TestQuasiParabolicBaseParams(TestCase):
    instance1 = {
        "f": 16E6,
        "fc": 8.875E6,
        "er": coords.EARTH_RADIUS,
        "rm": 350E3 + coords.EARTH_RADIUS,
        "rb": 100E3 + coords.EARTH_RADIUS,
        "beta0": math.pi/4,
        "a": 206.8310253720703125,
        "b": -2770916453.720314453,
        "c": 9291369922227116.7197,
        "xb": 21579020500000.000,
        "betab": 0.80073472391625508698
    }
    instance2 = {
        "f": 16E6,
        "fc": 17E6,
        "er": coords.EARTH_RADIUS,
        "rm": 100E3 + coords.EARTH_RADIUS,
        "rb": 50E3 + coords.EARTH_RADIUS,
        "beta0": math.pi/3,
        "a": 18617.45023281250,
        "b": -240948709217.7468750,
        "c": 779579401263770014.06,
        "xb": 31081830750000.,
        "betab": 1.051687543736184262
    }

    def test_calculate_atmosphere_params(self):
        """
        Tests calculate_param_a, calculate_param_b, calculate_param_c, calculate_param_x_b, calculate_param_beta_b
        """
        for instance in [self.instance1, self.instance2]:
            atmosphere_params = (
                instance["rm"], instance["rb"], instance["fc"] ** 2 / 80.62, instance["f"]
            )
            test_a = qp.calculate_param_a(*atmosphere_params)
            test_b = qp.calculate_param_b(*atmosphere_params)
            test_c = qp.calculate_param_c(instance["beta0"], *atmosphere_params)
            test_betab = qp.calculate_param_beta_b(instance["beta0"], instance["rb"])
            test_xb = qp.calculate_param_x_b(instance["beta0"], instance["rb"])

            testing.assert_is_close(instance["a"], test_a, rel_tol=1E-6)
            testing.assert_is_close(instance["b"], test_b, rel_tol=1E-6)
            testing.assert_is_close(instance["c"], test_c, rel_tol=1E-6)
            testing.assert_is_close(instance["betab"], test_betab, rel_tol=1E-6)
            testing.assert_is_close(instance["xb"], test_xb, rel_tol=1E-6)

    atmosphere_1 = (200E3 + coords.EARTH_RADIUS, 40E3 + coords.EARTH_RADIUS, 7E6**2 / 80.62, 10E6)
    atmosphere_2 = (500E3 + coords.EARTH_RADIUS, 350E3 + coords.EARTH_RADIUS, 7E6**2 / 80.62, 10E6)
    atmosphere_3 = (
        instance1["rm"], instance1["rb"], instance1["fc"] ** 2 / 80.62, instance1["f"]
    )
    atmosphere_4 = (
        instance2["rm"], instance2["rb"], instance2["fc"] ** 2 / 80.62, instance2["f"]
    )

    def test_get_angle_of_shortest_path(self):
        """
        Tests get_angle_of_shortest_path
        """
        for atmosphere, expected_result in [(self.atmosphere_1, 0.340804), (self.atmosphere_2, 0.613972),
                                            (self.atmosphere_3, 0.280384), (self.atmosphere_4, None)]:
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_angle_of_shortest_path(*atmosphere)
            else:
                testing.assert_is_close(
                    qp.get_angle_of_shortest_path(*atmosphere),
                    expected_result, rel_tol=1E-5
                )

    def test_get_ground_distance_derivative(self):
        """
        Tests ground_distance_derivative
        """
        test_angle_params_1 = (
            (math.pi/5., 579877.1850458072), ((3*math.pi)/20., 197445.4417837191), 
            (math.pi/10., -57934.24817360472), (math.pi/20., -922085.5304306145), 
            (0, -6.371e6)
        )
        test_angle_params_2 = (
            (math.pi/5., 222256.4290528555), ((3*math.pi)/20., -906647.621325564), 
            (math.pi/10., -1.8320712749519304e6), (math.pi/20., -3.5504613237075745e6), 
            (0, -6.371e6)
        )
        # Testing raw derivative
        for atmosphere, angle_params in [(self.atmosphere_1, test_angle_params_1),
                                         (self.atmosphere_2, test_angle_params_2)]:
            for angle, expected_derivative in angle_params:
                testing.assert_is_close(qp.ground_distance_derivative(angle, *atmosphere),
                                        expected_derivative, rel_tol=1E-6)

    def test_pedersen_angle(self):
        """
        Tests get_pedersen_angle
        """
        for atmosphere, expected_result in [(self.atmosphere_1, 0.743177), (self.atmosphere_2, 0.691971),
                                            (self.atmosphere_3, 0.502671), (self.atmosphere_4, None)]:
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_pedersen_angle(*atmosphere)
            else:
                testing.assert_is_close(qp.get_pedersen_angle(*atmosphere), expected_result, rel_tol=1E-6)

    def test_get_apogee_height(self):
        for atmosphere, launch_angle, expected_result in [
            (self.atmosphere_1, math.pi/8, 6440033), (self.atmosphere_2, math.pi/5, 6817028.),
                (self.atmosphere_3, math.pi/3, None), (self.atmosphere_4, math.pi/4, 6433896.)]:
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_apogee_height(launch_angle, *atmosphere)
            else:
                testing.assert_is_close(
                    qp.get_apogee_height(launch_angle, *atmosphere),
                    expected_result, rel_tol=1E-6
                )

    def test_get_apogee_ground_distance(self):
        """
        Tests get_apogee_ground_distance
        """
        for atmosphere, launch_angle, expected_result in [
            (self.atmosphere_1, math.pi/8, 232430.150104903),
                (self.atmosphere_2, math.pi/5, 702961.905270499),
                (self.atmosphere_3, math.pi/3, None),
                (self.atmosphere_4, math.pi/4, 75946.49213141459)]:
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_apogee_ground_distance(launch_angle, *atmosphere)
            else:
                testing.assert_is_close(
                    qp.get_apogee_ground_distance(launch_angle, *atmosphere),
                    expected_result, rel_tol=1E-6
                )

    def test_get_qp_heights(self):
        """
        Tests get_qp_heights
        """

        for atmosphere, launch_angle, expected_result in [
            (self.atmosphere_1, math.pi/8, 232430.150104903),
                (self.atmosphere_2, math.pi/5, 702961.905270499),
                (self.atmosphere_3, math.pi/3, None),
                (self.atmosphere_4, math.pi/4, 75946.49213141459)]:
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_qp_heights(launch_angle, np.zeros(10), *atmosphere)
            else:
                # TODO: Actually test this instead of just plotting here.
                # Either use mathematica and use the ground_distances -> heights to confirm backwards-forwards
                self.fail("Not implemented")
                # distances = np.linspace(0, qp.get_apogee_ground_distance(launch_angle, *atmosphere) * 2)
                # heights = qp.get_qp_heights(launch_angle, distances, *atmosphere)
                # plt.plot(distances, heights - coords.EARTH_RADIUS)
                # plt.show()
                # testing.assert_is_close(
                #     qp.get_apogee_ground_distance(launch_angle, *atmosphere),
                #     expected_result, rel_tol=1E-6
                # )

    def test_get_qp_path(self):
        # TODO Make sure path works. Use mathematica and a known angle to get a path
        self.fail()
