"""
Checks of the physics and of the solver. Run with `python3 -m pytest tests` or
`python3 tests/test_tracing.py`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atmospheres
import magnetic_fields
from paths import QuasiParabolic, GreatCircleDeviation
from tracing import coleman_equations as ce
from tracing.tracer import Tracer, integrate_parameter
from utilities import Vector
from utilities.Constants import EARTH_RADIUS, B_FIELD


def geographic_point(latitude, longitude):
    return Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(np.array([EARTH_RADIUS, latitude, longitude])))


def random_magnetoionic_parameters(sign, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.01, 0.6, n)
    # Stay below the reflection level of each mode (X = 1 for O, X = 1 - Y for X)
    x = rng.uniform(0, 0.95, n) * (1 if sign == 1 else 1 - y)
    yp = y * rng.uniform(-1, 1, n)
    return x, y, yp


def test_refractive_index_solves_booker_quartic():
    for sign in (1, -1):
        x, y, yp = random_magnetoionic_parameters(sign)
        mu2 = ce.equation_15(yp, x, y ** 2, sign)
        a = 1 - x - y ** 2 + x * yp ** 2
        b = (x - 1) * (1 - x - y ** 2) + x * y ** 2 / 2 - x * yp ** 2 / 2
        c = (1 - x) * ((1 - x) ** 2 - y ** 2)
        assert np.max(np.abs(a * mu2 ** 2 + 2 * b * mu2 + c)) < 1E-12
        # Coleman's form of eq. 16 in terms of A and B
        assert np.allclose(ce.equation_16(yp, x, y ** 2, sign), -x * yp * (mu2 - 1) / (a * mu2 + b), rtol=1E-8)


def test_yp_pt_solver():
    for sign in (1, -1):
        x, y, yp_true = random_magnetoionic_parameters(sign)
        yt = ce.equation_14(yp_true, x, y ** 2, sign)
        assert np.all(np.abs(yt) <= y * (1 + 1E-12))
        yp, pt = ce.solve_yp_pt(x, y, yt, sign)
        assert np.max(np.abs(yp - yp_true)) < 1E-12
        assert np.all((pt > 0) & (pt <= 1))
        # Both of the paper's equations hold at the solution
        assert np.max(np.abs(ce.equation_13(yp, x, y ** 2, yt, sign))) < 1E-12


def test_ray_direction_is_normal_to_refractive_index_surface():
    """mu * pt must be stationary with respect to the wave normal direction (Fermat)"""
    sign = 1
    x, y, yt = 0.4, 0.3, 0.3 * np.cos(0.7)
    # Wave normals in the plane of Y and t, at angle a from Y. t is at angle 0.7 from Y.
    angles = np.linspace(0.3, 1.1, 20001)
    values = np.sqrt(ce.equation_15(y * np.cos(angles), x, y ** 2, sign)) * np.cos(angles - 0.7)
    stationary_angle = angles[np.argmax(values)]
    yp, pt = ce.solve_yp_pt(np.array([x]), np.array([y]), np.array([yt]), sign)
    assert abs(yp[0] - y * np.cos(stationary_angle)) < 1E-4
    assert abs(ce.ray_refractive_index(x, y, yt, sign) - values.max()) < 1E-8


def test_dipole_field():
    field = magnetic_fields.DipoleField()
    equator, north_pole = geographic_point(0, 20), geographic_point(90, 0)
    south = geographic_point(-23.5, 133.7)
    # Horizontal and northward at the equator, twice as strong and downward at the north pole
    assert np.allclose(field.field_vec(equator), [0, 0, 1])
    assert np.isclose(field.field_mag(equator), B_FIELD)
    assert np.allclose(field.field_vec(north_pole), [0, 0, -1])
    assert np.isclose(field.field_mag(north_pole), 2 * B_FIELD)
    # Points up in the southern hemisphere, with tan(dip) = 2 tan(latitude)
    b = field.field_vec(south)
    up = south / EARTH_RADIUS
    assert np.isclose(np.arcsin(b @ up), np.arctan(2 * np.tan(np.radians(23.5))))
    assert field.field_vec(np.array([equator, south])).shape == (2, 3)


def test_geographic_conversions():
    assert np.allclose(geographic_point(90, 0) / EARTH_RADIUS, [0, 0, 1])
    assert geographic_point(-23.5, 133.7)[2] < 0


def perturbed_seed(path_generator, radial_number=42, lateral_number=2, amplitude=15E3):
    locations = np.linspace(0, 1, radial_number)
    seed = GreatCircleDeviation.from_path(locations, np.linspace(0, 1, lateral_number), path_generator)
    parameters = seed.adjustable_parameters
    bump = np.zeros_like(parameters)
    bump[:radial_number - 2] = amplitude * np.sin(np.pi * locations[1:-1]) ** 2
    return seed.adjust_parameters(np.arange(len(parameters)), bump)


def test_analytic_quasi_parabolic_ray_is_stationary():
    """
    Without a field, in a QP layer, the ray is known in closed form (Croft & Hoogasian 1968),
    so the solver must find (almost) nothing to change when it is started on that ray.

    The QP layer has a kink in its gradient at its base. That makes the discretised problem
    converge only to first order in the segment length, and can even stall the iteration (which
    the tracer reports as non convergence), so this test looks at the size of the first Newton
    step rather than at a fully converged path. Smooth layers are tested against Haselgrove
    ray tracing below.
    """
    start, end = geographic_point(-23.5, 133.7), geographic_point(-13.5, 133.7)
    f, fc, hm, ym = 10E6, 7E6, 350E3, 100E3
    atmosphere = atmospheres.QuasiParabolic(fc, hm, ym)
    for use_high_ray in (True, False):
        exact = QuasiParabolic(start, end, (fc, hm, ym), f, use_high_ray=use_high_ray)
        steps = []
        for segments in (80, 320):
            seed = GreatCircleDeviation.from_path(np.linspace(0, 1, segments + 1), np.linspace(0, 1, 2), exact)
            tracer = Tracer(f, atmosphere, magnetic_fields.ZeroField(), seed)
            matrix, gradient, _ = tracer.calculate_derivatives()
            steps.append(np.max(np.abs(np.linalg.solve(matrix, gradient))))
        # For scale: the ray climbs about 300 km
        assert steps[0] < 2000 and steps[1] < 500, steps


def test_same_ray_from_different_starting_paths():
    start, end = geographic_point(-23.5, 133.7), geographic_point(-13.5, 133.7)
    f, fc, hm, ym = 10E6, 7E6, 350E3, 100E3
    atmosphere = atmospheres.ChapmanLayers(fc, hm, ym, None, start)
    for use_high_ray in (True, False):
        generator = QuasiParabolic(start, end, (fc, hm, ym), f, use_high_ray=use_high_ray)
        solutions = []
        for amplitude in (0, 15E3, -15E3):
            tracer = Tracer(f, atmosphere, magnetic_fields.ZeroField(), perturbed_seed(generator, amplitude=amplitude))
            tracer.trace(verbose=False)
            assert tracer.converged
            solutions.append(tracer.calculated_paths[-1].adjustable_parameters)
        assert np.allclose(solutions[0], solutions[1], atol=0.1) and np.allclose(solutions[0], solutions[2], atol=0.1)


def test_agrees_with_haselgrove_ray_tracing():
    """
    Trace a ray with the (completely independent) Haselgrove equations, then ask the direct method
    for the ray between the same end points. Gradients along and across the path, dipole field.
    """
    from validation import haselgrove
    f = 10E6
    start = geographic_point(-23.5, 133.7)
    up = start / EARTH_RADIUS
    north = np.array([0, 0, 1.0]) - up[2] * up
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    horizontal = np.cos(np.radians(35)) * north + np.sin(np.radians(35)) * east
    direction = np.cos(np.radians(28)) * horizontal + np.sin(np.radians(28)) * up
    field = magnetic_fields.DipoleField()
    for gradient_direction in (-1, -2):
        atmosphere = atmospheres.ChapmanLayers(7E6, 350E3, 100E3, (0.2E6 * 180 / np.pi, gradient_direction), start)
        for is_extraordinary in (False, True):
            reference = haselgrove.trace_ray(atmosphere, field, f, start, direction, is_extraordinary=is_extraordinary)
            assert reference.landed
            end = reference.landing_point
            generator = QuasiParabolic(start, end, (7E6, 350E3, 100E3), f, use_high_ray=False)
            seed = GreatCircleDeviation.from_path(np.linspace(0, 1, 101), np.linspace(0, 1, 11), generator)
            tracer = Tracer(f, atmosphere, field, seed)
            tracer.trace(is_extraordinary_ray=is_extraordinary, verbose=False)
            assert tracer.converged
            solution = tracer.calculated_paths[-1]
            assert abs(tracer.phase_path - reference.phase_path) < 5
            points = solution(np.linspace(0, 1, 401))
            assert abs(np.max(np.linalg.norm(points, axis=1)) - EARTH_RADIUS - reference.apogee_height) < 20
            assert haselgrove.compare_to_path(reference.points, points)["max_distance"] < 100
            launch = solution(np.array([0.0]), nu=1)[0]
            assert np.degrees(np.arccos(launch @ direction / np.linalg.norm(launch))) < 0.02


def test_derivatives_match_brute_force():
    """The chain rule gradient and Hessian must agree with finite differences of P itself"""
    start, end = geographic_point(-23.5, 133.7), geographic_point(-15, 140)
    f = 10E6
    atmosphere = atmospheres.ChapmanLayers(7E6, 350E3, 100E3, (0.1E6 * 180 / np.pi, -1), start)
    generator = QuasiParabolic(start, end, (7E6, 350E3, 100E3), f, use_high_ray=False)
    path = GreatCircleDeviation.from_path(np.linspace(0, 1, 12), np.linspace(0, 1, 5), generator)
    path = path.adjust_parameters(np.arange(13), np.linspace(-3E3, 3E3, 13))
    tracer = Tracer(f, atmosphere, magnetic_fields.DipoleField(), path)
    for is_extraordinary in (False, True):
        state = tracer.get_system_state(is_extraordinary)
        matrix, gradient, _ = tracer.calculate_derivatives(is_extraordinary_ray=is_extraordinary)

        def phase(indexes, changes):
            return integrate_parameter(state, path.adjust_parameters(indexes, changes))

        h = 50.0
        for i in (0, 4, 9, 10, 12):
            brute = (phase(i, h) - phase(i, -h)) / (2 * h)
            assert abs(brute - gradient[i]) < 1E-6 * max(1, abs(gradient[i])) + 1E-9, (i, brute, gradient[i])
        for i, j in ((3, 3), (3, 4), (2, 11), (10, 12), (11, 11)):
            if i == j:
                brute = (phase(i, h) - 2 * phase(i, 0.0) + phase(i, -h)) / h ** 2
            else:
                brute = (phase([i, j], [h, h]) - phase([i, j], [h, -h]) -
                         phase([i, j], [-h, h]) + phase([i, j], [-h, -h])) / (4 * h ** 2)
            scale = np.sqrt(abs(matrix[i, i] * matrix[j, j]))
            assert abs(brute - matrix[i, j]) < 2E-3 * scale, (i, j, brute, matrix[i, j])


def test_magnetic_field_splits_the_modes():
    start, end = geographic_point(-23.5, 133.7), geographic_point(-15, 140)
    f = 10E6
    atmosphere = atmospheres.ChapmanLayers(7E6, 350E3, 100E3, None, start)
    generator = QuasiParabolic(start, end, (7E6, 350E3, 100E3), f, use_high_ray=False)
    seed = GreatCircleDeviation.from_path(np.linspace(0, 1, 42), np.linspace(0, 1, 10), generator)
    apogees, laterals = [], []
    for is_extraordinary in (False, True):
        tracer = Tracer(f, atmosphere, magnetic_fields.DipoleField(), seed)
        tracer.trace(is_extraordinary_ray=is_extraordinary, verbose=False)
        assert tracer.converged
        solution = tracer.calculated_paths[-1]
        fractions = np.linspace(0, 1, 501)
        apogees.append(np.max(np.linalg.norm(solution(fractions), axis=1)) - EARTH_RADIUS)
        # The deviation out of the great circle plane is S shaped, so sample it on the way up
        laterals.append(float(solution.poly_fit_angular(0.3)))
    # The extraordinary low ray reflects lower, and the two modes deviate to opposite sides
    assert apogees[1] < apogees[0] - 1E3
    assert laterals[0] * laterals[1] < 0 and min(np.abs(laterals)) > 100

    # Without a field the two modes are the same ray
    tracers = [Tracer(f, atmosphere, magnetic_fields.ZeroField(), seed) for _ in range(2)]
    paths = [t.trace(is_extraordinary_ray=flag, verbose=False)[-1] for t, flag in zip(tracers, (False, True))]
    assert np.allclose(paths[0].adjustable_parameters, paths[1].adjustable_parameters, atol=1E-3)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok  ", name)
