"""
Example: trace the high and low, ordinary and extraordinary rays between two
points 10 degrees of latitude apart, through a Chapman layer with a horizontal
gradient and a dipole magnetic field.

    python3 main.py                 # show the figures
    python3 main.py --save plots    # write the figures to ./plots instead
"""
import argparse
import math
import os
import time

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import magnetic_fields
from utilities import Vector
from atmospheres import ChapmanLayers
from utilities.Constants import EARTH_RADIUS
from paths import QuasiParabolic, GreatCircleDeviation
from tracing.tracer import Tracer


def geographic_point(latitude: float, longitude: float, altitude: float = 0.0) -> np.ndarray:
    """Cartesian coordinates of a point given in degrees (latitude positive north) and meters"""
    return Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(np.array([EARTH_RADIUS + altitude, latitude, longitude]))
    )


def main(save_directory=None, gradient_mhz_per_degree=0.1, use_field=True):
    # A path heading north from Alice Springs
    path_start_point = geographic_point(-23.5, 133.7)
    path_end_point = geographic_point(-23.5 + 10, 133.7)

    operating_frequency = 10E6
    atmosphere_critical_frequency = 7E6
    atmosphere_altitude_of_max = 350E3
    atmosphere_semi_width = 100E3
    # (Hz per radian, direction). -1 means the peak plasma frequency increases to the north
    atmosphere_gradient = (gradient_mhz_per_degree * 1E6 * 180 / math.pi, -1)
    atmosphere = ChapmanLayers(
        atmosphere_critical_frequency,
        atmosphere_altitude_of_max,
        atmosphere_semi_width,
        atmosphere_gradient,
        path_start_point
    )
    field = magnetic_fields.DipoleField() if use_field else magnetic_fields.ZeroField()

    # The analytic ray through a quasi-parabolic layer starts us off. As Coleman suggests, the
    # layer approximates the ionosphere above the middle of the path.
    midpoint = 0.5 * (path_start_point + path_end_point)
    atmosphere_params = (
        float(atmosphere.peak_frequency(midpoint)[0]), atmosphere_altitude_of_max, atmosphere_semi_width
    )
    path_generator = QuasiParabolic(path_start_point, path_end_point, atmosphere_params, operating_frequency)

    # 50 adjustable heights and 10 adjustable lateral deviations
    radial_locations, lateral_locations = np.linspace(0, 1, 52), np.linspace(0, 1, 12)

    results = {}
    for ray_name, use_high_ray in (("high", True), ("low", False)):
        path_generator.using_high_ray = use_high_ray
        path_generator.compile_points()
        seed = GreatCircleDeviation.from_path(radial_locations, lateral_locations, path_generator)
        for mode_name, is_extraordinary in (("ordinary", False), ("extraordinary", True)):
            print(f"\n=== {ray_name} ray, {mode_name} ===")
            tracer = Tracer(operating_frequency, atmosphere, field, seed)
            start_time = time.perf_counter()
            tracer.trace(is_extraordinary_ray=is_extraordinary)
            print(f"Took {time.perf_counter() - start_time:.2f} s")
            if not tracer.converged:
                print("No ray of this type was found. The last iteration is plotted but it is NOT a ray.")

            solution = tracer.calculated_paths[-1]
            fractions = np.linspace(0, 1, 2001)
            altitudes = np.linalg.norm(solution(fractions), axis=1) - EARTH_RADIUS
            lateral = solution.poly_fit_angular(fractions)
            print(f"Apogee {altitudes.max() / 1000:.2f} km, "
                  f"largest lateral deviation {lateral[np.argmax(np.abs(lateral))] / 1000:.3f} km")
            results[(ray_name, mode_name)] = tracer

            fig, ax = tracer.visualize(show_history=True, show=False)
            ax.set_title(f"{ray_name.capitalize()} {mode_name} ray, {int(operating_frequency / 1E6)} MHz")
            if save_directory is not None:
                fig.savefig(os.path.join(save_directory, f"{ray_name}_{mode_name}.png"), dpi=200)
                plt.close(fig)
            else:
                plt.show()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", metavar="DIRECTORY", help="save figures here instead of showing them")
    parser.add_argument("--gradient", type=float, default=0.1,
                        help="northward gradient of the peak plasma frequency in MHz per degree (default 0.1)")
    parser.add_argument("--no-field", action="store_true", help="use a zero magnetic field")
    arguments = parser.parse_args()
    if arguments.save is not None:
        matplotlib.use("Agg")
        os.makedirs(arguments.save, exist_ok=True)
    main(arguments.save, arguments.gradient, not arguments.no_field)
