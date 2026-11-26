"""
Reproduces the experiments behind Tables 1 and 2 of Coleman (2011).

The ionosphere is a Chapman layer (foF2 = 7 MHz, hmF2 = 350 km, ymF2 = 100 km)
with a gradient of 0.375 MHz per degree, the frequency is 10 MHz and the rays
leave Alice Springs heading north. As in the paper, a ray is first traced
with the Haselgrove equations from a given launch elevation, and then the
direct variational method is asked for the ray between the same end points
using different numbers of segments.

The absolute numbers are a few percent away from the paper's because it
doesn't give the exact Chapman formula, where the 7 MHz is anchored, or the
magnetic field model. What is being reproduced is the agreement between the
two methods and its convergence with the number of segments.

    python3 benchmarks/coleman_tables.py
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atmospheres
import magnetic_fields
from paths import QuasiParabolic, GreatCircleDeviation
from tracing.tracer import Tracer
from utilities import Vector
from utilities.Constants import EARTH_RADIUS
from validation import haselgrove

FREQUENCY, FC, HM, YM = 10E6, 7E6, 350E3, 100E3
GRADIENT = 0.375E6 * 180 / np.pi  # Hz per radian
SEGMENTS = (200, 100, 50, 24)


def launch_direction(start, elevation_degrees):
    """Unit vector pointing north from start at the given elevation"""
    up = start / np.linalg.norm(start)
    north = np.array([0, 0, 1.0]) - up[2] * up
    north /= np.linalg.norm(north)
    elevation = np.radians(elevation_degrees)
    return np.cos(elevation) * north + np.sin(elevation) * up


def direct_solution(atmosphere, field, start, end, segments, is_extraordinary, use_high_ray):
    generator = QuasiParabolic(start, end, (FC, HM, YM), FREQUENCY, use_high_ray=use_high_ray)
    # As in the paper, a tenth as many lateral samples as height samples
    seed = GreatCircleDeviation.from_path(
        np.linspace(0, 1, segments + 1), np.linspace(0, 1, max(segments // 10, 3) + 1), generator
    )
    tracer = Tracer(FREQUENCY, atmosphere, field, seed)
    start_time = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracer.trace(is_extraordinary_ray=is_extraordinary, verbose=False)
    elapsed = time.perf_counter() - start_time
    solution = tracer.calculated_paths[-1]
    fractions = np.linspace(0, 1, 4001)
    heights = np.linalg.norm(solution(fractions), axis=1) - EARTH_RADIUS
    lateral = np.abs(solution.poly_fit_angular(fractions))
    return tracer.converged, tracer.phase_path, heights.max(), lateral.max(), elapsed


def lateral_deviation(points, start, end):
    normal = Vector.unit_vector(np.cross(start, end))
    return np.max(np.abs(points @ normal))


def run_case(title, atmosphere, field, start, rays):
    print(f"\n{title}")
    print(f"{'':24s}" + "".join(f"{name + ' PP':>14s}{name + ' hmax':>13s}{name + ' dmax':>13s}" for name, *_ in rays))
    rows = {"Haselgrove": []}
    rows.update({f"direct, {n} segments": [] for n in SEGMENTS})
    times = []
    for name, elevation, is_extraordinary, use_high_ray in rays:
        reference = haselgrove.trace_ray(
            atmosphere, field, FREQUENCY, start, launch_direction(start, elevation), is_extraordinary=is_extraordinary
        )
        if not reference.landed:
            raise RuntimeError(f"The Haselgrove ray for {name} did not return to the ground")
        end = reference.landing_point
        rows["Haselgrove"].append(
            (True, reference.phase_path, reference.apogee_height, lateral_deviation(reference.points, start, end))
        )
        for n in SEGMENTS:
            *result, elapsed = direct_solution(atmosphere, field, start, end, n, is_extraordinary, use_high_ray)
            rows[f"direct, {n} segments"].append(result)
            times.append((n, elapsed))
    for label, results in rows.items():
        cells = "".join(
            f"{pp / 1000:14.3f}{hmax / 1000:13.3f}{dmax / 1000:13.3f}" if converged else f"{'no convergence':>40s}"
            for converged, pp, hmax, dmax in results
        )
        print(f"{label:24s}{cells}")
    print("  (km) direct solve times: " + ", ".join(
        f"{n} segments {np.mean([t for m, t in times if m == n]):.2f} s" for n in SEGMENTS))


if __name__ == "__main__":
    start_point = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(np.array([EARTH_RADIUS, -23.5, 133.7])))
    dipole = magnetic_fields.DipoleField()

    run_case(
        "Table 1: gradient along the path (peak plasma frequency increasing 0.375 MHz per degree to the north)",
        atmospheres.ChapmanLayers(FC, HM, YM, (GRADIENT, -1), start_point), dipole, start_point,
        [("O low", 25, False, False), ("X low", 25, True, False)]
    )
    run_case(
        "Table 2: gradient across the path (peak plasma frequency increasing 0.375 MHz per degree to the west)",
        atmospheres.ChapmanLayers(FC, HM, YM, (GRADIENT, -2), start_point), dipole, start_point,
        # The paper launches its high ray at 43 degrees. In this ionosphere rays above 39 degrees
        # penetrate, so the high ray is launched at 38 degrees instead.
        [("O low", 30, False, False), ("O high", 38, False, True)]
    )
