import math
import numpy as np

import Field
import Vector
from Atmosphere import ChapmanLayers
from Constants import EARTH_RADIUS
from Paths import QuasiParabolic, GreatCircleDeviation
from Tracer import Tracer

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500

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
    atmosphere_params = (atmosphere_critical_frequency, atmosphere_altitude_of_max, atmosphere_semi_width)
    path_generator = QuasiParabolic(path_start_point, path_end_point, atmosphere_params, operating_frequency)
    path_start = GreatCircleDeviation.from_path(np.linspace(0, 1, 52), np.linspace(0, 1, 2), other_path=path_generator)
    basic_tracer = Tracer(operating_frequency, atmosphere, field, path_start)

    basic_tracer.trace()
    basic_tracer.visualize(show_history=True)

    path_generator.using_high_ray = False
    path_generator.compile_points()
    basic_tracer.replace_path(GreatCircleDeviation.from_path(
        np.linspace(0, 1, 52), np.linspace(0, 1, 2), path_generator
    ))
    basic_tracer.trace()
    basic_tracer.visualize()

    basic_tracer.cleanup()  # Should call after we are done with tracer
