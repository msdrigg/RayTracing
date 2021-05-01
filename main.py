import os
import time

import math
import numpy as np
from matplotlib.lines import Line2D

import Field
import Vector
from Atmosphere import ChapmanLayers
from Constants import EARTH_RADIUS
from Paths import QuasiParabolic
from Tracer import Tracer

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500

if __name__ == "__main__":
    field = Field.DipoleField()
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
    atmosphere_gradient = (0.375E6 * 180 / math.pi, -1)
    atmosphere = ChapmanLayers(
        atmosphere_critical_frequency,
        atmosphere_altitude_of_max,
        atmosphere_semi_width,
        atmosphere_gradient,
        path_start_point
    )
    # atmosphere = ChapmanLayers(7E6, 350E3, 100E3, None, initial)
    atmosphere_params = (atmosphere_critical_frequency, atmosphere_altitude_of_max, atmosphere_semi_width)
    path_generator = QuasiParabolic(path_start_point, path_end_point, atmosphere_params, operating_frequency)
    frequency = 10E6  # Hz
    # atmosphere.visualize(initial, final, ax=None, fig=None, point_number=400, show=True)
    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.parameters = (50, 0)
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = path_start_point, path_end_point

    basic_tracer.compile_initial_path()

    basic_tracer.trace(h=10, high_ray=True)
    t0 = time.time()
    basic_tracer.trace(h=10, high_ray=True)
    print(time.time() - t0)
    atmosphere.visualize()
    basic_tracer.trace(h=10, high_ray=True)
    fig, ax = basic_tracer.visualize(show=False)
    basic_tracer.trace(h=10, high_ray=False)
    basic_tracer.visualize(fig=fig, ax=ax)
    basic_tracer.visualize(show_history=True)
    basic_tracer.trace(h=10, high_ray=False)
    basic_tracer.visualize(show_history=True)
    basic_tracer.trace(h=10, high_ray=False, is_extraordinary_ray=True)
    fig, ax = basic_tracer.visualize(fig=fig, ax=ax, show=False, color='white')

    custom_lines = [Line2D([0], [0], color='black', lw=4),
                    Line2D([0], [0], color='white', lw=4)]
    ax.legend(custom_lines, ['Ordinary Ray', 'Extraordinary Ray'])
    ax.set_title(f"Various Ray Traces with a {int(operating_frequency / 1E6)} MHz frequency")
    fig.savefig(os.path.join('saved_plots', 'test_path_all_grad'))
    plt.close(fig)

    basic_tracer.cleanup()  # Should call after we are done with tracer
