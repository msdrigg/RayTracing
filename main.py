import math
import numpy as np

import Field
import Vector
from Atmosphere import ChapmanLayers
from Constants import EARTH_RADIUS
from Paths import QuasiParabolic
from Tracer import Tracer


if __name__ == "__main__":
    field = Field.ZeroField()
    path_start_point = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            np.array([EARTH_RADIUS, 90 + 23.5, 133.7])))
    path_end_point = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            np.array([EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    operating_frequency = 10E6
    atmosphere_critical_frequency = 7E6
    atmosphere_altitude_of_max = 350E3
    atmosphere_semi_width = 100E3
    atmosphere_gradient = (0.1E6 * 180 / math.pi, -1)
    atmosphere = ChapmanLayers(
        atmosphere_critical_frequency,
        atmosphere_altitude_of_max,
        atmosphere_semi_width,
        atmosphere_gradient,
        path_start_point
    )
    # atmosphere = ChapmanLayers(7E6, 350E3, 100E3, None, initial)
    path_generator = QuasiParabolic(path_start_point, path_end_point, atmosphere, operating_frequency)
    frequency = 10E6  # Hz
    # atmosphere.visualize(initial, final, ax=None, fig=None, point_number=400, show=True)
    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.parameters = (50, 0)
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = path_start_point, path_end_point

    basic_tracer.compile_initial_path()

    basic_tracer.trace(debug_while_calculating=False, h=10, high_ray=True)
    basic_tracer.visualize(show_history=True)
    basic_tracer.trace(debug_while_calculating=False, h=10, high_ray=True, is_extraordinary_ray=True)
    basic_tracer.visualize(show_history=True)
    basic_tracer.trace(debug_while_calculating=False, h=10, high_ray=False)
    basic_tracer.visualize(show_history=True)
    basic_tracer.trace(debug_while_calculating=False, h=10, high_ray=False, is_extraordinary_ray=True)
    basic_tracer.visualize(show_history=True)
    basic_tracer.cleanup()  # Should call after we are done with tracer
