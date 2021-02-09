from unittest import TestCase
from utils import coordinates as coords
from utils import testing
import math
from initialize import quasi_parabolic as qp
import numpy as np
from matplotlib import pyplot as plt
from utils import plotting


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
        heights_map_1 = {
            0.: -9.313225746154785e-10,
            59512.138892608666: 43750.69657665491,
            119024.27778521733: 79357.1327750124,
            178536.41667782597: 102476.00323323254,
            238048.55557043466: 115454.88882118836,
            297560.6944630433: 119635.83955823164,
            357072.83335565194: 115454.88882118836,
            416584.9722482606: 102476.00323322974,
            476097.1111408693: 79357.13277501334,
            535609.2500334779: 43750.69657665398,
            595121.3889260866: -9.313225746154785e-10
        }
        heights_map_2 = {
            0.: -9.313225746154785e-10,
            140592.38105404895: 105404.73810752388,
            281184.7621080979: 217618.22637726553,
            421777.1431621468: 337112.9730895199,
            562369.5242161958: 423427.95128732733,
            702961.9052702447: 446028.0773610296,
            843554.2863242936: 423427.95128732733,
            984146.6673783425: 337112.9730895199,
            1.1247390484323916e6: 217618.22637726553,
            1.2653314294864405e6: 105404.73810752202,
            1.4059238105404894e6: -9.313225746154785e-10
        }
        heights_map_3 = None
        heights_map_4 = {
            0.: -1.862645149230957e-9,
            15189.298426254249: 15243.777116704732,
            30378.596852508497: 30597.150661764666,
            45567.89527876274: 46061.088167474605,
            60757.193705016994: 58815.94639489055,
            75946.49213127124: 62896.03693717811,
            91135.79055752548: 58815.94639489055,
            106325.08898377973: 46061.088167474605,
            121514.38741003399: 30597.150661764666,
            136703.68583628823: 15243.777116704732,
            151892.98426254248: -1.862645149230957e-9

        }
        i = 1
        for atmosphere, launch_angle, expected_result in [
            (self.atmosphere_1, math.pi/5, heights_map_1),
                (self.atmosphere_2, math.pi/5, heights_map_2),
                (self.atmosphere_3, math.pi/4, heights_map_3),
                (self.atmosphere_4, math.pi/4, heights_map_4)]:
            print("Doing test " + str(i))
            i += 1
            if expected_result is None:
                with self.assertRaises(ValueError):
                    qp.get_qp_heights(launch_angle, np.zeros(10), *atmosphere)
            else:
                distances = list(expected_result.keys())
                heights = [expected_result[key] for key in distances]
                dists_numpy = np.array(distances)
                expected_heights_numpy = np.array(heights)
                result_heights_numpy = qp.get_qp_heights(launch_angle, dists_numpy, *atmosphere) - coords.EARTH_RADIUS
                # plt.plot(dists_numpy, expected_heights_numpy, color='green')
                # plt.plot(dists_numpy, result_heights_numpy, color='blue')
                # plt.show()
                np.testing.assert_allclose(
                    expected_heights_numpy, result_heights_numpy, rtol=1E-7, atol=1E-5
                )

    def test_get_qp_path(self):
        """
        Tests python code against mathematica.
        All distances are equally spaced.
        atm_n_result_m signifies the m 'th ray in atmosphere n.
        m is 1 or 2 (for high/low ray) and n is anything
        """
        atm_1_result_1 = {
            0.: 0.,
            33358.47799351321: 30438.376457925886,
            66716.95598702642: 59857.23787973542,
            100075.43398053964: 84257.37434461899,
            133433.91197405284: 104224.97853034455,
            166792.38996756607: 120518.83398114704,
            200150.8679610793: 133769.23422350734,
            233509.34595459249: 144496.1483021425,
            266867.8239481057: 153125.40623837803,
            300226.3019416189: 160002.7517724987,
            333584.77993513213: 165405.77521852683,
            366943.2579286453: 169553.8306521261,
            400301.7359221586: 172616.0832423549,
            433660.2139156718: 174717.84173964616,
            467018.69190918497: 175945.31996173225,
            500377.1699026982: 176348.9475973323,
            533735.6478962114: 175945.31996161956,
            567094.1258897246: 174717.84173953626,
            600452.6038832378: 172616.08324242663,
            633811.081876751: 169553.83065223414,
            667169.5598702643: 165405.77521841507,
            700528.0378637775: 160002.75177248754,
            733886.5158572906: 153125.4062383417,
            767244.9938508039: 144496.14830209687,
            800603.4718443172: 133769.23422353156,
            833961.9498378304: 120518.83398114704,
            867320.4278313436: 104224.97853036039,
            900678.9058248567: 84257.37434462178,
            934037.3838183699: 59857.23787973914,
            967395.8618118832: 30438.37645792868,
            1.0007543398053964e6: 1.862645149230957e-9
        }
        atm_1_result_2 = {
            0.: 0.,
            33358.47799319029: 1772.4751942036673,
            66716.95598638058: 3720.751727901399,
            100075.43397957088: 5845.0977685963735,
            133433.91197276115: 8145.805942695588,
            166792.38996595144: 10623.193434099667,
            200150.86795914176: 13277.60209117271,
            233509.34595233205: 16109.398542195559,
            266867.8239455223: 19118.974319351837,
            300226.3019387126: 22306.74599134829,
            333584.7799319029: 25673.155304719694,
            366943.2579250932: 29218.669333944097,
            400301.7359182835: 32943.78064044751,
            433660.2139114738: 36849.007440580055,
            467018.6919046641: 40844.1935769571,
            500377.1698978544: 42526.72510015592,
            533735.6478910446: 40844.19357693009,
            567094.1258842349: 36849.00744058099,
            600452.6038774252: 32943.780640449375,
            633811.0818706155: 29218.66933394596,
            667169.5598638058: 25673.155304720625,
            700528.0378569961: 22306.74599135015,
            733886.5158501863: 19118.9743193537,
            767244.9938433768: 16109.398542195559,
            800603.471836567: 13277.602091173641,
            833961.9498297573: 10623.193434099667,
            867320.4278229476: 8145.805942695588,
            900678.9058161379: 5845.0977685963735,
            934037.3838093282: 3720.7517279023305,
            967395.8618025185: 1772.4751942036673,
            1.0007543397957088e6: 0.
        }
        atm_2_result_1 = {
            0.: 0.,
            46764.29999999596: 33321.832566644065,
            93528.59999999191: 67342.76892844401,
            140292.8999999879: 102077.70582551323,
            187057.19999998383: 137542.0360449273,
            233821.49999997977: 173751.667826741,
            280585.7999999758: 210723.04521456547,
            327350.0999999717: 248473.16940409876,
            374114.39999996766: 287019.6211465737,
            420878.6999999636: 326380.5842677923,
            467642.99999995955: 365473.50382417906,
            514407.29999995546: 395529.5108364066,
            561171.5999999515: 416582.626917826,
            607935.8999999474: 430433.0291420156,
            654700.1999999434: 438284.7006113287,
            701464.4999999393: 440827.1409868179,
            748228.7999999353: 438284.7006113343,
            794993.0999999313: 430433.0291420147,
            841757.3999999271: 416582.626917826,
            888521.6999999231: 395529.5108364066,
            935285.9999999191: 365473.50382417627,
            982050.2999999151: 326380.5842677923,
            1.0288145999999109e6: 287019.6211465737,
            1.075578899999907e6: 248473.16940409876,
            1.122343199999903e6: 210723.04521456547,
            1.169107499999899e6: 173751.667826741,
            1.2158717999998948e6: 137542.0360449273,
            1.262636099999891e6: 102077.70582551323,
            1.3094003999998868e6: 67342.76892844401,
            1.3561646999998828e6: 33321.832566644065,
            1.4029289999998787e6: 0.
        }
        atm_2_result_2 = {
            0.: 0.,
            46764.29999998558: 33290.6602136381,
            93528.59999997116: 67279.76209731586,
            140292.89999995675: 101982.1778243687,
            187057.19999994233: 137413.27463261224,
            233821.49999992788: 173588.93418130744,
            280585.7999999135: 210525.57284984458,
            327350.09999989904: 248240.1630313974,
            374114.39999988466: 286750.2554783011,
            420878.6999998702: 326074.00275960285,
            467642.99999985576: 365172.28296222445,
            514407.2999998414: 395284.9122392973,
            561171.599999827: 416379.3499962259,
            607935.8999998126: 430257.79629387613,
            654700.1999997981: 438125.7034950899,
            701464.4999997837: 440673.45725638233,
            748228.7999997693: 438125.7034950899,
            794993.0999997548: 430257.79629387986,
            841757.3999997404: 416379.3499962194,
            888521.699999726: 395284.9122392591,
            935285.9999997115: 365172.28296221886,
            982050.2999996971: 326074.00275960285,
            1.0288145999996827e6: 286750.2554783011,
            1.0755788999996684e6: 248240.1630313974,
            1.122343199999654e6: 210525.57284984458,
            1.1691074999996396e6: 173588.93418130744,
            1.2158717999996252e6: 137413.27463261224,
            1.2626360999996108e6: 101982.1778243687,
            1.3094003999995962e6: 67279.7620973168,
            1.3561646999995818e6: 33290.6602136381,
            1.4029289999995674e6: 0.
        }
        atm_4_result_1 = {
            0.: 9.313225746154785e-10,
            13343.391197384837: 3330.7080525411293,
            26686.782394769674: 6692.890776319429,
            40030.17359215452: 10086.627522166818,
            53373.56478953935: 13511.998486295342,
            66716.95598692418: 16969.08471505996,
            80060.34718430904: 20457.96810980141,
            93403.73838169387: 23978.731431737542,
            106747.1295790787: 27531.458306915127,
            120090.52077646353: 31116.23323122412,
            133433.91197384836: 34733.141575477086,
            146777.30317123322: 38382.26959054358,
            160120.69436861807: 42063.704412550665,
            173464.0855660029: 45777.53406814765,
            186807.47676338773: 49523.847479837015,
            200150.86796077256: 51636.269840589724,
            213494.2591581574: 49523.847479837015,
            226837.65035554222: 45777.53406814765,
            240181.04155292705: 42063.704412550665,
            253524.4327503119: 38382.26959054358,
            266867.8239476967: 34733.14157547802,
            280211.2151450816: 31116.23323122412,
            293554.60634246643: 27531.458306915127,
            306897.9975398513: 23978.731431737542,
            320241.38873723615: 20457.96810980141,
            333584.77993462095: 16969.08471505996,
            346928.1711320058: 13511.998486295342,
            360271.5623293906: 10086.627522166818,
            373614.95352677547: 6692.890776319429,
            386958.34472416027: 3330.7080525411293,
            400301.7359215451: 9.313225746154785e-10
        }
        should_plot = True

        for atmosphere, ground_distance, expected_results in [
            (self.atmosphere_1, math.pi/20*coords.EARTH_RADIUS, (atm_1_result_1, atm_1_result_2)),
                (self.atmosphere_2, 1402928 + 1, (atm_2_result_1, atm_2_result_2)),
                (self.atmosphere_3, math.pi/30*coords.EARTH_RADIUS, None),
                (self.atmosphere_4, math.pi/50*coords.EARTH_RADIUS, (atm_4_result_1,))]:
            if expected_results is None:
                with self.assertRaises(ValueError):
                    qp.get_quasi_parabolic_path(ground_distance, *atmosphere)
            else:
                point_count = len(expected_results[0])

                result_paths = qp.get_quasi_parabolic_path(ground_distance, *atmosphere,
                                                           step_size_horizontal=ground_distance/(point_count - 1))
                self.assertTrue(len(result_paths) == len(expected_results), "Gotten {} paths but expected {}"
                                .format(len(result_paths), len(expected_results)))

                fig, ax = None, None
                if should_plot:
                    semi_width = atmosphere[0] - atmosphere[1]
                    max_height = max(
                        (atmosphere[1] + 2 * semi_width - coords.EARTH_RADIUS) * 1.05 / 1000,
                        np.amax(result_paths[0][:, 1] - coords.EARTH_RADIUS) / 1000
                    )

                    fig, ax = plotting.visualize_atmosphere(
                        lambda a: np.sqrt(qp.calculate_e_density(a[:, 0], *atmosphere) * 80.62),
                        np.array([coords.EARTH_RADIUS, 0, 0]),
                        np.array([coords.EARTH_RADIUS, result_paths[0][-1, 0] / coords.EARTH_RADIUS, 0]),
                        show=False,
                        max_height=max_height
                    )
                
                for expected, gotten in zip(expected_results, result_paths):
                    distances = list(expected.keys())
                    heights = [expected[key] for key in distances]
                    expected_distances_numpy = np.array(list(distances))
                    expected_heights_numpy = np.array(heights)
                    expected_combined = np.column_stack((
                        expected_distances_numpy,
                        expected_heights_numpy
                    ))
                    gotten[:, 1] = gotten[:, 1] - coords.EARTH_RADIUS
                    np.testing.assert_allclose(
                        expected_combined, gotten, rtol=1E-7, atol=1E-4
                    )
                    if should_plot and fig is not None:
                        plotting.visualize_path(gotten, show=False, fig=fig, ax=ax, color='white')
                        # plotting.visualize_path(expected_combined, show=True, fig=fig, ax=ax, color='black')

                if should_plot:
                    plt.show()
