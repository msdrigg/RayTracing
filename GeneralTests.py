from Constants import EARTH_RADIUS
from numpy import array, arange
from math import atan
from numpy.linalg import norm
from matplotlib import pyplot as plt
import Atmosphere
import Constants as Const
import Vector as Vec
import Paths as Path
import Tracer as Trace
import Field


def test_integration():
    frequency = 10E6
    qp = Atmosphere.QuasiParabolic(9.5E6, 350E3 + EARTH_RADIUS, 100E3)
    initial = Vec.spherical_to_cartesian(
        Vec.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vec.spherical_to_cartesian(
        Vec.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    path = Path.QuasiParabolic(initial, final, qp, frequency)
    atmosphere = Atmosphere.ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / Const.PI, -1), initial)
    test_fig, ax1 = plt.subplots(figsize=(6, 4.5))
    atmosphere.visualize(initial, final, ax=ax1, fig=test_fig, point_number=400, show=False)
    path.visualize(fig=test_fig, ax=ax1, color="black")

    tracer = Trace.Tracer(frequency, qp, Field.ZeroField(), Path.QuasiParabolic)
    tracer.parameters = 100, 0
    tracer.parameter_number = 100
    tracer.initial_coordinates, tracer.final_coordinates = initial, final
    tracer.compile_initial_path()
    params = tracer.calculated_paths[0].parameters
    indexes = arange(tracer.parameters[0])

    # tracer.calculated_paths[0].adjust_parameters(
    #     indexes,
    #     -(params[:, 1] - EARTH_RADIUS)/250,
    #     mutate=True
    # )
    integration = tracer.integrate_parameter(tracer.calculated_paths[0], h=0.00001)
    print(f"Real final: {final}")
    analytic_result = 1.24598E6
    print(f"Calculated result: {integration}")
    print(f"Analytic Result: {analytic_result}")
    print(f"Percent Error: {(integration - analytic_result)/integration*100}%")
    print(f"Raw Error: {(integration - analytic_result)/integration}")


def test_qp_model():
    frequency = 18E6
    qp = Atmosphere.QuasiParabolic(9.5E6, 350E3 + EARTH_RADIUS, 100E3)
    initial = Vec.spherical_to_cartesian(
        Vec.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vec.spherical_to_cartesian(
        Vec.latitude_to_spherical(
            array([EARTH_RADIUS, 90 + 23.5 - 16, 133.7])))
            
    tracer_parameters = 100, 0
    tracer = Trace.Tracer(frequency, qp, Field.ZeroField(), Path.QuasiParabolic)
    tracer.parameters = tracer_parameters
    tracer.parameter_number = sum(tracer_parameters)
    tracer.initial_coordinates, tracer.final_coordinates = initial, final
    tracer.compile_initial_path()
    params = tracer.calculated_paths[0].parameters
    indexes = arange(tracer.parameters[0])
    initial_path = tracer.calculated_paths[0]
    # new_path = tracer.calculated_paths[0].adjust_parameters(
    #     indexes,
    #     (params[:, 1] - EARTH_RADIUS)/3,
    #     mutate=False
    # )
    # tracer.calculated_paths.append(new_path)
    paths = tracer.trace(steps=30, h=100, debug_while_calculating='save')
    print(f"Snells final: {tracer.integrate_parameter(paths[-1])}")
    print(f"Analytic Result for QP Path: {tracer.integrate_parameter(paths[0])}")
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    paths[-1].visualize(fig=fig, ax=ax, color='red', show=False)
    initial_path.visualize(fig=fig, ax=ax, color='green')
    print(f"Starting angle of final path: {atan((norm(paths[-1](0.05))-norm(paths[-1](0)))/(0.05*Const.EARTH_RADIUS*paths[-1].total_angle))}")


if __name__ == "__main__":
    test_qp_model()
    # test_integration()
