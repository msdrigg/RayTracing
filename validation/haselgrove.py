"""
Independent initial-value (Haselgrove / Hamiltonian) HF ray tracer.

This module exists to *validate* the variational point-to-point solver in
``tracing/``.  It deliberately shares no code and no maths with it: the ray
equations below are derived directly from the collisionless Appleton-Hartree
dispersion relation and integrated as an initial value problem.

Conventions
-----------
* SI units.  Positions are cartesian metres from the Earth's centre.
* X = (f_p / f)^2,  Y_vec = (f_H / f) * b_hat,  Y_p = Y_vec . p  where p is the
  unit wave normal.
* Appleton-Hartree (no collisions)::

      mu^2 = 1 - 2X(1-X) / D
      D    = 2(1-X) - Y_T^2 +/- sqrt(Y_T^4 + 4 (1-X)^2 Y_p^2),   Y_T^2 = Y^2 - Y_p^2

  "+" is the ordinary mode, "-" the extraordinary mode.
* Hamiltonian, as a function of position r and refractive-index vector n
  (|n| = mu on the ray, direction of n = wave normal)::

      H(r, n) = 1/2 [ n.n - mu^2(r, n/|n|) ]

      dr/dtau =  dH/dn = n - 1/2 (dmu^2/dY_p) (Y_vec - Y_p p) / |n|
      dn/dtau = -dH/dr = 1/2 [ mu2_X grad X + mu2_Y2 grad(Y^2) + mu2_Yp grad(Y_vec) . p ]

  The ray (energy) direction is dr/dtau, which differs from n when Y != 0.
  Because mu^2 is homogeneous of degree zero in n, n . dH/dn = n.n, hence

      d(phase path)/dtau = n . dr/dtau = mu^2
      d(group path)/dtau = n.n + (f/2) d(mu^2)/df
                         = n.n - X mu2_X - Y^2 mu2_Y2 - 1/2 Y_p mu2_Yp

  (X ~ f^-2, Y ~ f^-1; for Y = 0 this reduces to dP'/ds = 1/mu as it must).

* Spatial gradients of X and Y_vec are taken by central finite differences of
  the supplied models, so any duck-typed model works:

      atmosphere.plasma_frequency(points (N,3)) -> (N,) Hz
      field.gyro_frequency(points (N,3))        -> (N,) Hz
      field.field_vec(points (N,3))             -> (N,3) unit vectors

  ``field`` may be ``None`` for an isotropic (no magnetic field) trace.

Public API: :func:`trace_ray`, :func:`home_ray`, :func:`compare_to_path`,
plus the helper :func:`ground_range`.

Run ``python validation/haselgrove.py`` for the self-tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

EARTH_RADIUS = 6371e3

__all__ = [
    "EARTH_RADIUS",
    "RayResult",
    "HomingResult",
    "trace_ray",
    "home_ray",
    "compare_to_path",
    "ground_range",
    "appleton_hartree_mu2",
]


# ----------------------------------------------------------------------------
# Appleton-Hartree refractive index and its partial derivatives
# ----------------------------------------------------------------------------
def _mu2_and_partials(X, Y2, Yp, sign):
    """
    mu^2(X, Y^2, Y_p) and its partial derivatives with respect to X, Y^2 and
    Y_p (each taken with the other two held fixed).

    ``sign`` = +1 ordinary, -1 extraordinary.  Works on scalars or arrays.
    Returns (mu2, dmu2_dX, dmu2_dY2, dmu2_dYp).
    """
    X = np.asarray(X, dtype=float)
    Y2 = np.asarray(Y2, dtype=float)
    Yp = np.asarray(Yp, dtype=float)

    U = 1.0 - X
    T = np.maximum(Y2 - Yp * Yp, 0.0)  # Y_T^2 (clamped against round-off)
    S = np.sqrt(T * T + 4.0 * U * U * Yp * Yp)
    D = 2.0 * U - T + sign * S

    with np.errstate(divide="ignore", invalid="ignore"):
        # 1/S appears only multiplied by things that vanish with S.
        inv_S = np.where(S > 0.0, 1.0 / np.where(S > 0.0, S, 1.0), 0.0)

        mu2 = 1.0 - 2.0 * X * U / D
        G = 2.0 * X * U / (D * D)  # d(mu2)/dD

        dD_dX = -2.0 - sign * 4.0 * U * Yp * Yp * inv_S
        dD_dT = -1.0 + sign * T * inv_S
        dD_dYp_fixed_T = sign * 4.0 * U * U * Yp * inv_S

        dmu2_dX = -2.0 * (1.0 - 2.0 * X) / D + G * dD_dX
        dmu2_dY2 = G * dD_dT
        dmu2_dYp = G * (dD_dT * (-2.0 * Yp) + dD_dYp_fixed_T)

    # Exactly unmagnetised plasma: mu^2 = 1 - X (also removes the 0/0 at X = 1).
    iso = Y2 == 0.0
    if np.any(iso):
        mu2 = np.where(iso, 1.0 - X, mu2)
        dmu2_dX = np.where(iso, -1.0, dmu2_dX)
        dmu2_dY2 = np.where(iso, 0.0, dmu2_dY2)
        dmu2_dYp = np.where(iso, 0.0, dmu2_dYp)
    return mu2, dmu2_dX, dmu2_dY2, dmu2_dYp


def appleton_hartree_mu2(X, Y2, Yp, is_extraordinary=False):
    """Collisionless Appleton-Hartree mu^2 for given X, Y^2 and Y_p = Y_vec.p."""
    return _mu2_and_partials(X, Y2, Yp, -1.0 if is_extraordinary else 1.0)[0]


# ----------------------------------------------------------------------------
# Model evaluation helpers
# ----------------------------------------------------------------------------
def _eval_X(atmosphere, frequency, points):
    fp = np.atleast_1d(np.asarray(atmosphere.plasma_frequency(points), dtype=float))
    if fp.shape[0] != points.shape[0]:
        fp = np.broadcast_to(fp.reshape(-1)[:1], (points.shape[0],))
    return (fp / frequency) ** 2


def _eval_Yvec(field, frequency, points):
    n = points.shape[0]
    if field is None:
        return np.zeros((n, 3))
    fh = np.atleast_1d(np.asarray(field.gyro_frequency(points), dtype=float))
    if fh.shape[0] != n:
        fh = np.broadcast_to(fh.reshape(-1)[:1], (n,))
    if not np.any(fh):
        return np.zeros((n, 3))
    b = np.asarray(field.field_vec(points), dtype=float).reshape(-1, 3)
    if b.shape[0] != n:
        b = np.broadcast_to(b[:1], (n, 3))
    b = np.nan_to_num(b)
    return (fh / frequency)[:, None] * b


def _hamiltonian(atmosphere, field, frequency, sign, points, n_vecs):
    """H = 1/2 (n.n - mu^2) at many points (vectorised)."""
    points = np.atleast_2d(points)
    n_vecs = np.atleast_2d(n_vecs)
    X = _eval_X(atmosphere, frequency, points)
    Yv = _eval_Yvec(field, frequency, points)
    nn2 = np.einsum("ij,ij->i", n_vecs, n_vecs)
    p = n_vecs / np.sqrt(nn2)[:, None]
    Yp = np.einsum("ij,ij->i", Yv, p)
    Y2 = np.einsum("ij,ij->i", Yv, Yv)
    mu2 = _mu2_and_partials(X, Y2, Yp, sign)[0]
    return 0.5 * (nn2 - mu2)


_FD_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)


def _make_rhs(atmosphere, field, frequency, sign, fd_step):
    """
    Right-hand side for the state y = [r(3), n(3), phase_path, group_path, arc_length].
    """
    offsets = _FD_OFFSETS * fd_step
    inv_2h = 1.0 / (2.0 * fd_step)

    def rhs(_tau, y):
        r = y[0:3]
        n = y[3:6]
        pts = r[None, :] + offsets

        X_all = _eval_X(atmosphere, frequency, pts)
        Y_all = _eval_Yvec(field, frequency, pts)

        X = X_all[0]
        Yv = Y_all[0]
        grad_X = (X_all[1::2] - X_all[2::2]) * inv_2h  # (3,)  dX/dr_i
        jac_Y = (Y_all[1::2] - Y_all[2::2]) * inv_2h  # (3,3) [i, j] = dY_j/dr_i

        nn2 = n @ n
        nn = np.sqrt(nn2)
        p = n / nn
        Yp = Yv @ p
        Y2 = Yv @ Yv

        _, m_X, m_Y2, m_Yp = _mu2_and_partials(X, Y2, Yp, sign)
        m_X = float(m_X)
        m_Y2 = float(m_Y2)
        m_Yp = float(m_Yp)

        dr = n - 0.5 * m_Yp * (Yv - Yp * p) / nn
        dn = 0.5 * (m_X * grad_X + 2.0 * m_Y2 * (jac_Y @ Yv) + m_Yp * (jac_Y @ p))

        out = np.empty(9)
        out[0:3] = dr
        out[3:6] = dn
        out[6] = n @ dr  # phase path:  n . dr
        out[7] = nn2 - X * m_X - Y2 * m_Y2 - 0.5 * Yp * m_Yp  # group path: c dt
        out[8] = np.sqrt(dr @ dr)  # geometric arc length
        return out

    return rhs


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
class _DictAccess:
    """Lets the result dataclasses also be used like dicts: result['points']."""

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return list(self.__dataclass_fields__.keys())


@dataclass
class RayResult(_DictAccess):
    points: np.ndarray  # (M,3) ray positions, densely sampled (launch -> end)
    wave_normals: np.ndarray  # (M,3) refractive-index vectors n at the samples
    tau: np.ndarray  # (M,) integration parameter at the samples
    landing_point: Optional[np.ndarray]  # (3,) or None
    landed: bool
    phase_path: float  # integral of n . dr  [m]
    group_path: float  # integral of c dt    [m]
    geometric_path: float  # arc length          [m]
    max_abs_H: float  # max |H| over the samples (should be ~0)
    apogee_height: float  # max(|r|) - ground radius over the samples [m]
    termination: str  # 'landed' | 'escaped' | 'max_path_length' | 'integration_failed'
    launch_direction: np.ndarray  # (3,) unit launch direction (wave normal at launch)
    landing_direction: Optional[np.ndarray]  # (3,) unit ray direction at landing
    nfev: int
    sol: Any = _dc_field(default=None, repr=False)  # dense output: sol(tau) -> state(9)


@dataclass
class HomingResult(_DictAccess):
    ray: RayResult  # the homed ray (fully sampled)
    miss_distance: float  # |landing_point - end_point| [m]
    miss_vector: np.ndarray  # (2,) [along-track, cross-track] miss [m]
    launch_direction: np.ndarray  # (3,) unit cartesian launch direction
    elevation: float  # launch elevation above local horizontal [rad]
    azimuth_offset: float  # launch azimuth relative to the great circle
    #                        start->end, positive towards cross(up, track) [rad]
    converged: bool
    iterations: int
    n_traces: int
    message: str = ""


# ----------------------------------------------------------------------------
# 1. Initial value ray trace
# ----------------------------------------------------------------------------
def trace_ray(
    atmosphere,
    field,
    frequency,
    start_point,
    direction,
    is_extraordinary=False,
    max_path_length=2.0e7,
    rtol=1e-9,
    atol=None,
    *,
    method="DOP853",
    fd_step=10.0,
    max_step=10e3,
    sample_spacing=250.0,
    ground_radius=EARTH_RADIUS,
    escape_height=1500e3,
    sample=True,
):
    """
    Trace one ray as an initial value problem.

    Parameters
    ----------
    atmosphere, field : duck-typed models (see module docstring); ``field`` may be None.
    frequency : wave frequency [Hz].
    start_point : (3,) cartesian metres.  Normally on the ground.
    direction : (3,) launch direction of the wave normal (need not be unit).
        On the ground X = 0, so wave normal == ray direction there.  If the
        start point is inside the plasma, |n| is set to mu(start, direction).
    is_extraordinary : False -> O mode ("+"), True -> X mode ("-").
    max_path_length : stop after this much geometric arc length [m].
    rtol : relative tolerance for the integrator.
    atol : absolute *position* tolerance [m] (default 1e-4 m).  The absolute
        tolerance for n is atol / EARTH_RADIUS.  A length-9 array is passed
        straight to solve_ivp.
    method : 'DOP853' (default) or any solve_ivp method with dense output.
    fd_step : step [m] of the central differences for model gradients.
        Note: where the model has a gradient discontinuity (e.g. the edge of a
        quasi-parabolic layer) the differences smooth it over +/- fd_step, so
        a sample that falls inside that zone can show |H| ~ |grad X| fd_step / 8
        (~1e-5) even though the ray is fine; for smooth models max|H| ~ 1e-8.
    max_step : largest step in tau (~metres) so thin layers are not skipped.
    sample_spacing : approximate spacing [m] of the returned ``points``.
    ground_radius : radius of the "ground" sphere [m].
    escape_height : stop if the ray climbs above this height [m].
    sample : if False skip the dense sampling/H check (used while homing).

    Returns
    -------
    RayResult (dataclass, also indexable like a dict) -- see its fields.
    """
    sign = -1.0 if is_extraordinary else 1.0
    r0 = np.asarray(start_point, dtype=float).reshape(3)
    d0 = np.asarray(direction, dtype=float).reshape(3)
    d0 = d0 / np.linalg.norm(d0)

    # |n| = mu at the start (== 1 on the ground where X = 0).
    X0 = _eval_X(atmosphere, frequency, r0[None, :])[0]
    Yv0 = _eval_Yvec(field, frequency, r0[None, :])[0]
    mu2_0 = float(_mu2_and_partials(X0, Yv0 @ Yv0, Yv0 @ d0, sign)[0])
    if not np.isfinite(mu2_0) or mu2_0 <= 0.0:
        raise ValueError("wave is evanescent at the start point (mu^2 = %r)" % mu2_0)
    y0 = np.concatenate([r0, np.sqrt(mu2_0) * d0, [0.0, 0.0, 0.0]])

    if atol is None:
        atol = 1e-4
    atol_arr = np.asarray(atol, dtype=float)
    if atol_arr.ndim == 0:
        a = float(atol_arr)
        atol_arr = np.array([a, a, a] + [a / EARTH_RADIUS] * 3 + [a, a, a])

    def ev_ground(_t, y):
        return np.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) - ground_radius

    ev_ground.terminal = True
    ev_ground.direction = -1.0

    def ev_escape(_t, y):
        return np.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) - (ground_radius + escape_height)

    ev_escape.terminal = True
    ev_escape.direction = 1.0

    def ev_length(_t, y):
        return y[8] - max_path_length

    ev_length.terminal = True
    ev_length.direction = 1.0

    rhs = _make_rhs(atmosphere, field, frequency, sign, fd_step)

    # tau advances no faster than arc length is consumed only when |dr/dtau|<=1,
    # which is not guaranteed for the X mode, so give tau a generous span and
    # let the arc-length event do the stopping.
    t_end = 50.0 * max_path_length
    with np.errstate(all="ignore"):
        sol = solve_ivp(
            rhs,
            (0.0, t_end),
            y0,
            method=method,
            rtol=rtol,
            atol=atol_arr,
            dense_output=True,
            events=(ev_ground, ev_escape, ev_length),
            max_step=max_step,
            first_step=min(100.0, max_step),
        )

    y_end = sol.y[:, -1]
    t_final = sol.t[-1]
    landed = False
    landing_point = None
    landing_direction = None
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        landed = True
        termination = "landed"
        y_end = sol.y_events[0][-1]
        t_final = sol.t_events[0][-1]
        landing_point = y_end[0:3] * (ground_radius / np.linalg.norm(y_end[0:3]))
        v = rhs(t_final, y_end)[0:3]
        landing_direction = v / np.linalg.norm(v)
    elif sol.status == 1 and len(sol.t_events[1]) > 0:
        termination = "escaped"
    elif sol.status == 1 and len(sol.t_events[2]) > 0:
        termination = "max_path_length"
    elif sol.status == 0:
        termination = "max_path_length"
    else:
        termination = "integration_failed"

    if sample and t_final > 0.0 and np.all(np.isfinite(y_end)):
        m = int(np.clip(np.ceil(y_end[8] / sample_spacing), 2, 2_000_000)) + 1
        taus = np.linspace(0.0, t_final, m)
        ys = sol.sol(taus)
        points = ys[0:3].T.copy()
        normals = ys[3:6].T.copy()
        if landed:
            points[-1] = landing_point
        with np.errstate(all="ignore"):
            H = _hamiltonian(atmosphere, field, frequency, sign, points, normals)
        max_abs_H = float(np.nanmax(np.abs(H))) if np.any(np.isfinite(H)) else float("nan")
        apogee = float(np.max(np.linalg.norm(points, axis=1)) - ground_radius)
    else:
        taus = np.array([0.0, t_final])
        points = np.vstack([r0, y_end[0:3]])
        normals = np.vstack([y0[3:6], y_end[3:6]])
        max_abs_H = float("nan")
        apogee = float("nan")

    return RayResult(
        points=points,
        wave_normals=normals,
        tau=taus,
        landing_point=landing_point,
        landed=landed,
        phase_path=float(y_end[6]),
        group_path=float(y_end[7]),
        geometric_path=float(y_end[8]),
        max_abs_H=max_abs_H,
        apogee_height=apogee,
        termination=termination,
        launch_direction=d0,
        landing_direction=landing_direction,
        nfev=int(sol.nfev),
        sol=sol.sol,
    )


# ----------------------------------------------------------------------------
# 2. Homing
# ----------------------------------------------------------------------------
def _track_frame(start_point, end_point):
    """Orthonormal (up, along-track, cross-track) frame at the start point."""
    r0 = np.asarray(start_point, dtype=float).reshape(3)
    r1 = np.asarray(end_point, dtype=float).reshape(3)
    up = r0 / np.linalg.norm(r0)
    t = r1 - (r1 @ up) * up
    nt = np.linalg.norm(t)
    if nt < 1e-9 * np.linalg.norm(r1):
        raise ValueError("start and end points are (anti)parallel: track is undefined")
    t = t / nt
    c = np.cross(up, t)
    return up, t, c


def _direction_from_angles(frame, elevation, azimuth):
    up, t, c = frame
    ce = np.cos(elevation)
    return ce * (np.cos(azimuth) * t + np.sin(azimuth) * c) + np.sin(elevation) * up


def ground_range(point_a, point_b, radius=EARTH_RADIUS):
    """Great-circle distance [m] between the directions of two cartesian points."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return radius * np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)


def home_ray(
    atmosphere,
    field,
    frequency,
    start_point,
    end_point,
    initial_direction,
    is_extraordinary=False,
    *,
    tol=0.1,
    max_iterations=40,
    angle_fd_step=2e-6,
    max_angle_step=np.radians(2.0),
    rtol=1e-9,
    atol=None,
    max_path_length=2.0e7,
    **trace_kwargs,
):
    """
    Adjust the two launch angles until the ray from ``start_point`` lands on
    ``end_point``.

    The unknowns are (elevation, azimuth offset from the start->end great
    circle).  The residual is the ground miss vector [along-track, cross-track]
    in metres.  A damped Newton iteration is used: the 2x2 Jacobian comes from
    forward differences with a tiny angular step (``angle_fd_step`` rad), each
    Newton step is limited to ``max_angle_step`` and back-tracked until the
    ray still lands and the miss actually decreases.  This converges to the
    solution *branch* (low ray / high ray, hop count) nearest to
    ``initial_direction``; it does not search for other branches.

    Returns a HomingResult with fields: ray (RayResult), miss_distance [m],
    miss_vector [along, cross] [m], launch_direction (3,), elevation [rad],
    azimuth_offset [rad], converged, iterations, n_traces, message.
    """
    r0 = np.asarray(start_point, dtype=float).reshape(3)
    r1 = np.asarray(end_point, dtype=float).reshape(3)
    ground_radius = trace_kwargs.get("ground_radius", EARTH_RADIUS)
    frame = _track_frame(r0, r1)
    up, t, c = frame
    e_hat = r1 / np.linalg.norm(r1)
    target_angle = np.arctan2(e_hat @ t, e_hat @ up)

    d = np.asarray(initial_direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    x = np.array([np.arcsin(np.clip(d @ up, -1.0, 1.0)), np.arctan2(d @ c, d @ t)])

    el_min, el_max = np.radians(0.01), np.radians(89.99)
    counter = {"n": 0}

    def run(xv, full=False):
        counter["n"] += 1
        kw = dict(trace_kwargs)
        kw["sample"] = full
        return trace_ray(
            atmosphere,
            field,
            frequency,
            r0,
            _direction_from_angles(frame, xv[0], xv[1]),
            is_extraordinary,
            max_path_length=max_path_length,
            rtol=rtol,
            atol=atol,
            **kw,
        )

    def miss(xv):
        ray = run(xv)
        if not ray.landed:
            return None
        l_hat = ray.landing_point / np.linalg.norm(ray.landing_point)
        along = ground_radius * (np.arctan2(l_hat @ t, l_hat @ up) - target_angle)
        along = (along + np.pi * ground_radius) % (2 * np.pi * ground_radius) - np.pi * ground_radius
        cross = ground_radius * np.arcsin(np.clip(l_hat @ c, -1.0, 1.0))
        return np.array([along, cross])

    def jacobian(xv, res):
        J = np.zeros((2, 2))
        for j in range(2):
            ok = False
            for h in (angle_fd_step, -angle_fd_step):
                xp = xv.copy()
                xp[j] += h
                rp = miss(xp)
                if rp is not None:
                    J[:, j] = (rp - res) / h
                    ok = True
                    break
            if not ok:
                return None
        return J

    message = ""
    converged = False
    iterations = 0
    res = miss(x)
    if res is None:
        message = "initial ray does not return to the ground"
    else:
        for iterations in range(1, max_iterations + 1):
            if np.linalg.norm(res) < tol:
                converged = True
                iterations -= 1
                break
            J = jacobian(x, res)
            if J is None:
                message = "could not form Jacobian (neighbouring rays do not land)"
                break
            dx = np.linalg.lstsq(J, -res, rcond=None)[0]
            biggest = np.max(np.abs(dx))
            if biggest > max_angle_step:
                dx = dx * (max_angle_step / biggest)
            accepted = False
            alpha = 1.0
            for _ in range(25):
                xn = x + alpha * dx
                xn[0] = np.clip(xn[0], el_min, el_max)
                rn = miss(xn)
                if rn is not None and np.linalg.norm(rn) < np.linalg.norm(res):
                    x, res = xn, rn
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                message = "line search stalled (near a skip-distance fold or no such ray)"
                break
        else:
            converged = np.linalg.norm(res) < tol
            if not converged:
                message = "maximum iterations reached"
        if np.linalg.norm(res) < tol:
            converged = True
            message = ""

    ray = run(x, full=True)
    if ray.landed:
        miss_distance = float(np.linalg.norm(ray.landing_point - r1 * (ground_radius / np.linalg.norm(r1))))
    else:
        miss_distance = float("inf")
    return HomingResult(
        ray=ray,
        miss_distance=miss_distance,
        miss_vector=res if res is not None else np.array([np.nan, np.nan]),
        launch_direction=_direction_from_angles(frame, x[0], x[1]),
        elevation=float(x[0]),
        azimuth_offset=float(x[1]),
        converged=bool(converged),
        iterations=int(iterations),
        n_traces=int(counter["n"]),
        message=message,
    )


# ----------------------------------------------------------------------------
# 3. Comparison with a solver path
# ----------------------------------------------------------------------------
def compare_to_path(ray_points, path_points):
    """
    Distance from each of ``path_points`` (K,3) (e.g. the variational solver's
    path) to the polyline through ``ray_points`` (M,3).

    The nearest polyline vertices are found with a KD-tree and the distance is
    then refined to the exact point-to-segment distance on the neighbouring
    segments, so the result does not depend on the ray sampling density.

    Returns a dict with:
        max_distance  [m], rms_distance [m], mean_distance [m],
        distances (K,) [m], index_of_max (index into path_points),
        endpoint_distances (2,): |path[0]-ray[0]|, |path[-1]-ray[-1]| [m].
    """
    ray = np.asarray(ray_points, dtype=float).reshape(-1, 3)
    path = np.asarray(path_points, dtype=float).reshape(-1, 3)
    if ray.shape[0] < 2:
        raise ValueError("need at least two ray points")
    m = ray.shape[0]

    k = min(3, m)
    _, idx = cKDTree(ray).query(path, k=k)
    idx = np.asarray(idx).reshape(path.shape[0], k)

    # segments on either side of each of the k nearest vertices
    seg_start = np.clip(np.concatenate([idx - 1, idx], axis=1), 0, m - 2)

    a = ray[seg_start]  # (K, 2k, 3)
    b = ray[seg_start + 1]
    ab = b - a
    ap = path[:, None, :] - a
    denom = np.einsum("ijk,ijk->ij", ab, ab)
    s = np.einsum("ijk,ijk->ij", ap, ab) / np.where(denom > 0.0, denom, 1.0)
    s = np.clip(s, 0.0, 1.0)
    closest = a + s[..., None] * ab
    dist = np.min(np.linalg.norm(path[:, None, :] - closest, axis=2), axis=1)

    return {
        "max_distance": float(np.max(dist)),
        "rms_distance": float(np.sqrt(np.mean(dist**2))),
        "mean_distance": float(np.mean(dist)),
        "distances": dist,
        "index_of_max": int(np.argmax(dist)),
        "endpoint_distances": np.array(
            [np.linalg.norm(path[0] - ray[0]), np.linalg.norm(path[-1] - ray[-1])]
        ),
    }


# ============================================================================
# Self-tests
# ============================================================================
if __name__ == "__main__":
    import time

    np.set_printoptions(precision=6, suppress=False, linewidth=140)

    # ---------------- stub models (independent of the repo classes) ---------
    class QuasiParabolicLayer:
        """f_p^2 = f_c^2 [1 - ((r - r_m)/y_m)^2 (r_b/r)^2] inside the layer, else 0."""

        def __init__(self, f_c, h_m, y_m):
            self.f_c = f_c
            self.r_m = EARTH_RADIUS + h_m
            self.y_m = y_m
            self.r_b = self.r_m - y_m
            self.r_t = self.r_m * self.r_b / (self.r_b - y_m)

        def plasma_frequency(self, points):
            r = np.linalg.norm(np.atleast_2d(points), axis=1)
            fp2 = self.f_c**2 * (1.0 - ((r - self.r_m) / self.y_m) ** 2 * (self.r_b / r) ** 2)
            fp2 = np.where((r > self.r_b) & (r < self.r_t), fp2, 0.0)
            return np.sqrt(np.maximum(fp2, 0.0))

    class ChapmanLayer:
        """Smooth spherically symmetric alpha-Chapman layer."""

        def __init__(self, f_c, h_m, scale_height):
            self.f_c, self.h_m, self.H = f_c, h_m, scale_height

        def plasma_frequency(self, points):
            h = np.linalg.norm(np.atleast_2d(points), axis=1) - EARTH_RADIUS
            z = (h - self.h_m) / self.H
            return self.f_c * np.exp(0.25 * (1.0 - z - np.exp(-z)))

    class ZeroField:
        def gyro_frequency(self, points):
            return np.zeros(np.atleast_2d(points).shape[0])

        def field_vec(self, points):
            return np.tile([0.0, 0.0, 1.0], (np.atleast_2d(points).shape[0], 1))

    class DipoleField:
        """Earth-centred axis-aligned dipole: B = -B0 (R_E/r)^3 [3 cos(theta) r_hat - z_hat]."""

        E_CHARGE = 1.602176634e-19
        E_MASS = 9.1093837015e-31

        def __init__(self, b0=3.12e-5):
            self.b0 = b0

        def _b(self, points):
            pts = np.atleast_2d(points)
            r = np.linalg.norm(pts, axis=1)
            r_hat = pts / r[:, None]
            cos_theta = r_hat[:, 2]
            z_hat = np.array([0.0, 0.0, 1.0])
            return -self.b0 * (EARTH_RADIUS / r)[:, None] ** 3 * (3.0 * cos_theta[:, None] * r_hat - z_hat)

        def gyro_frequency(self, points):
            return self.E_CHARGE * np.linalg.norm(self._b(points), axis=1) / (2 * np.pi * self.E_MASS)

        def field_vec(self, points):
            b = self._b(points)
            return b / np.linalg.norm(b, axis=1)[:, None]

    def geo_to_cart(lat_deg, lon_deg, radius=EARTH_RADIUS):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        return radius * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def enu(point):
        up = point / np.linalg.norm(point)
        east = np.cross([0.0, 0.0, 1.0], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        return east, north, up

    def launch_dir(point, elevation_deg, azimuth_deg):
        """Azimuth clockwise from north."""
        east, north, up = enu(point)
        el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
        return np.cos(el) * (np.cos(az) * north + np.sin(az) * east) + np.sin(el) * up

    def out_of_plane(points, start, direction):
        """Max signed-magnitude distance of points from the plane (centre, start, direction)."""
        normal = np.cross(start, direction)
        normal /= np.linalg.norm(normal)
        d = points @ normal
        return d[np.argmax(np.abs(d))]

    failures = []

    def check(name, ok):
        print("    [%s] %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            failures.append(name)

    # ---------------- (0) analytic partials of mu^2 vs finite differences ---
    print("=" * 78)
    print("(0) Appleton-Hartree partial derivatives vs finite differences")
    rng = np.random.default_rng(1)
    worst = 0.0
    for sgn in (+1.0, -1.0):
        Xs = rng.uniform(0.02, 0.6, 2000)
        Ys = rng.uniform(0.05, 0.3, 2000)
        cs = rng.uniform(-1, 1, 2000)
        Y2s, Yps = Ys**2, Ys * cs
        _, aX, aY2, aYp = _mu2_and_partials(Xs, Y2s, Yps, sgn)
        e = 1e-6
        nX = (_mu2_and_partials(Xs + e, Y2s, Yps, sgn)[0] - _mu2_and_partials(Xs - e, Y2s, Yps, sgn)[0]) / (2 * e)
        nY2 = (_mu2_and_partials(Xs, Y2s + e, Yps, sgn)[0] - _mu2_and_partials(Xs, Y2s - e, Yps, sgn)[0]) / (2 * e)
        nYp = (_mu2_and_partials(Xs, Y2s, Yps + e, sgn)[0] - _mu2_and_partials(Xs, Y2s, Yps - e, sgn)[0]) / (2 * e)
        ok = np.abs(Y2s - Yps**2) > 1e-4  # keep the FD stencil away from the clamp
        worst = max(worst, np.max(np.abs(aX - nX)[ok]), np.max(np.abs(aY2 - nY2)[ok]), np.max(np.abs(aYp - nYp)[ok]))
    print("    max |analytic - FD| over 4000 random (X, Y, angle), both modes: %.2e" % worst)
    check("mu^2 partial derivatives", worst < 1e-6)

    # ---------------- (a) QP layer, no field, vs closed form ----------------
    print("=" * 78)
    print("(a) Quasi-parabolic layer, zero field, f = 10 MHz, f_c = 7 MHz, h_m = 350 km, y_m = 100 km")
    f = 10e6
    qp = QuasiParabolicLayer(7e6, 350e3, 100e3)

    def qp_closed_form(beta0):
        """Croft & Hoogasian (1968) ground range, phase path and group path."""
        R = EARTH_RADIUS
        F = f / qp.f_c
        rb, rm, ym = qp.r_b, qp.r_m, qp.y_m
        cb = np.cos(beta0)
        A = 1.0 - 1.0 / F**2 + (rb / (F * ym)) ** 2
        B = -2.0 * rm * rb**2 / (F * ym) ** 2
        C = (rb * rm / (F * ym)) ** 2 - R**2 * cb**2
        disc = B * B - 4 * A * C
        if disc <= 0:
            return None
        beta_b = np.arccos(R * cb / rb)
        Xb = rb**2 - R**2 * cb**2
        D = 2 * (
            R * (beta_b - beta0)
            + R**2 * cb / np.sqrt(C) * np.log((2 * C + B * rb + 2 * np.sqrt(C * Xb)) / (rb * np.sqrt(disc)))
        )
        # group path (same reference): P' = 2 { rb sin(beta_b) - R sin(beta0)
        #    + (1/A) [ -rb sin(beta_b) - B/(4 sqrt(A)) ln( (B^2-4AC) / (2A rb + B + 2 rb sqrt(A) sin(beta_b))^2 ) ] }
        sb = np.sin(beta_b)
        Pg = 2 * (
            rb * sb
            - R * np.sin(beta0)
            + (1.0 / A)
            * (-rb * sb - B / (4 * np.sqrt(A)) * np.log(disc / (2 * A * rb + B + 2 * rb * np.sqrt(A) * sb) ** 2))
        )
        return D, Pg

    start = geo_to_cart(0.0, 0.0)
    zero = ZeroField()
    print("    elev   closed-form D [m]     traced D [m]      diff [m]   closed P' [m]     traced P' [m]   dP' [m]"
          "   max|H|     apogee[km]  time[s]")
    worst_diff = 0.0
    worst_H = 0.0
    n_reflected = 0
    for el_deg in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        cf = qp_closed_form(np.radians(el_deg))
        t0 = time.perf_counter()
        ray = trace_ray(qp, zero, f, start, launch_dir(start, el_deg, 37.0))
        dt = time.perf_counter() - t0
        if cf is None:
            print("    %4.1f   (penetrates per closed form)   traced: %s" % (el_deg, ray.termination))
            check("elevation %.0f deg penetrates in both" % el_deg, not ray.landed)
            continue
        n_reflected += 1
        D_cf, Pg_cf = cf
        D_tr = ground_range(start, ray.landing_point) if ray.landed else float("nan")
        diff = D_tr - D_cf
        worst_diff = max(worst_diff, abs(diff))
        worst_H = max(worst_H, ray.max_abs_H)
        print(
            "    %4.1f   %16.3f  %16.3f  %+10.3f  %16.3f  %16.3f  %+8.3f  %.1e  %9.3f  %6.2f"
            % (el_deg, D_cf, D_tr, diff, Pg_cf, ray.group_path, ray.group_path - Pg_cf, ray.max_abs_H,
               ray.apogee_height / 1e3, dt)
        )
    check("at least 3 reflected elevations tested", n_reflected >= 3)
    check("ground range agrees with closed form to < 30 m (worst %.3f m)" % worst_diff, worst_diff < 30.0)
    # The QP layer has a gradient discontinuity at its base; samples that fall within
    # +/- fd_step of it see |H| ~ |grad X| fd_step / 8 ~ 1e-5 (see trace_ray docstring).
    check("max |H| < 1e-4 across the layer-edge kink (worst %.1e)" % worst_H, worst_H < 1e-4)

    # field=None must be equivalent to the zero-field stub
    r_a = trace_ray(qp, None, f, start, launch_dir(start, 20.0, 37.0))
    r_b = trace_ray(qp, zero, f, start, launch_dir(start, 20.0, 37.0))
    check("field=None equals zero-field stub", np.allclose(r_a.landing_point, r_b.landing_point, atol=1e-6))
    oop = out_of_plane(r_b.points, start, r_b.launch_direction)
    print("    zero-field out-of-plane deviation: %.2e m" % oop)
    check("zero-field ray stays in its great-circle plane (< 0.1 m)", abs(oop) < 0.1)

    # isotropic homing against the closed form
    el_true = 20.0
    D_cf = qp_closed_form(np.radians(el_true))[0]
    east, north, up = enu(start)
    az = np.radians(37.0)
    horiz = np.cos(az) * north + np.sin(az) * east
    ang = D_cf / EARTH_RADIUS
    target = EARTH_RADIUS * (np.cos(ang) * up + np.sin(ang) * horiz)
    t0 = time.perf_counter()
    hom = home_ray(qp, zero, f, start, target, launch_dir(start, 17.0, 38.5))
    dt = time.perf_counter() - t0
    print(
        "    homing to closed-form range for %.0f deg: elevation found %.6f deg (error %.2e deg), "
        "miss %.4f m, %d iterations, %d traces, %.2f s"
        % (el_true, np.degrees(hom.elevation), np.degrees(hom.elevation) - el_true, hom.miss_distance,
           hom.iterations, hom.n_traces, dt)
    )
    check("isotropic homing converged (miss < 1 m)", hom.converged and hom.miss_distance < 1.0)
    check("homed elevation matches closed form to 1e-3 deg", abs(np.degrees(hom.elevation) - el_true) < 1e-3)

    # ---------------- (b) dipole + Chapman layer ----------------------------
    print("=" * 78)
    print("(b) Axis-aligned dipole (B0 = 3.12e-5 T) + Chapman layer (f_c = 7 MHz, h_m = 300 km, H = 50 km), f = 10 MHz")
    chap = ChapmanLayer(7e6, 300e3, 50e3)
    dip = DipoleField()
    start = geo_to_cart(-23.5, 133.7)
    fh0 = dip.gyro_frequency(start[None, :])[0]
    print("    gyro frequency at start: %.4f MHz (Y = %.4f);  X at ground: %.2e"
          % (fh0 / 1e6, fh0 / f, (chap.plasma_frequency(start[None, :])[0] / f) ** 2))
    el_deg = 30.0
    rays = {}
    for az_deg, label in ((0.0, "north"), (60.0, "az=60")):
        d0 = launch_dir(start, el_deg, az_deg)
        for name, field_model, is_x in (("none", None, False), ("O", dip, False), ("X", dip, True)):
            t0 = time.perf_counter()
            ray = trace_ray(chap, field_model, f, start, d0, is_extraordinary=is_x)
            dt = time.perf_counter() - t0
            rays[(label, name)] = ray
            if not ray.landed:
                print("    %-6s %-4s did not land: %s" % (label, name, ray.termination))
                check("%s %s ray lands" % (label, name), False)
                continue
            lateral_ground = out_of_plane(ray.landing_point[None, :], start, d0)
            lateral_max = out_of_plane(ray.points, start, d0)
            # angle between wave normal and ray direction, max along the ray
            v = ray.sol(ray.tau)  # state at samples
            rhs_dbg = _make_rhs(chap, field_model, f, -1.0 if is_x else 1.0, 10.0)
            sub = np.linspace(0, len(ray.tau) - 1, 400).astype(int)
            angs = []
            for i in sub:
                dy = rhs_dbg(ray.tau[i], v[:, i])
                cosang = (v[3:6, i] @ dy[0:3]) / (np.linalg.norm(v[3:6, i]) * np.linalg.norm(dy[0:3]))
                angs.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            print(
                "    %-6s %-4s range %13.3f m  apogee %8.3f km  P %13.3f m  P' %13.3f m  max|H| %.1e  "
                "lateral: landing %+10.3f m, max along ray %+10.3f m  max(n^ray) %5.2f deg  nfev %5d  %5.2f s"
                % (label, name, ground_range(start, ray.landing_point), ray.apogee_height / 1e3, ray.phase_path,
                   ray.group_path, ray.max_abs_H, lateral_ground, lateral_max, max(angs), ray.nfev, dt)
            )
            check("%s %s lands, max|H| < 1e-7" % (label, name), ray.landed and ray.max_abs_H < 1e-7)

    if all(r.landed for r in rays.values()):
        for label in ("north", "az=60"):
            ro, rx, rn = rays[(label, "O")], rays[(label, "X")], rays[(label, "none")]
            d_ox = ground_range(start, ro.landing_point) - ground_range(start, rx.landing_point)
            print("    %-6s O-X range difference: %+.3f m ; O - no-field: %+.3f m ; X - no-field: %+.3f m"
                  % (label, d_ox, ground_range(start, ro.landing_point) - ground_range(start, rn.landing_point),
                     ground_range(start, rx.landing_point) - ground_range(start, rn.landing_point)))
            check("%s: O and X land at different ranges (> 1 km apart)" % label, abs(d_ox) > 1e3)
            check("%s: phase path < geometric path < group path (O)" % label,
                  ro.phase_path < ro.geometric_path < ro.group_path)
        # Exactly northward in an axis-aligned dipole the problem is mirror-symmetric
        # about the meridian plane (mu^2 is even in B), so the lateral deviation must vanish.
        lat_n = [abs(out_of_plane(rays[("north", m)].points, start, rays[("north", m)].launch_direction)) for m in "OX"]
        lat_o = [abs(out_of_plane(rays[("az=60", m)].points, start, rays[("az=60", m)].launch_direction)) for m in "OX"]
        check("northward (in magnetic meridian): lateral deviation ~ 0 by symmetry (%.1e, %.1e m)" % tuple(lat_n),
              max(lat_n) < 1e-2)
        check("az=60: nonzero lateral deviation for O and X (%.1f, %.1f m)" % tuple(lat_o), min(lat_o) > 10.0)
        lo = out_of_plane(rays[("az=60", "O")].points, start, rays[("az=60", "O")].launch_direction)
        lx = out_of_plane(rays[("az=60", "X")].points, start, rays[("az=60", "X")].launch_direction)
        check("az=60: O and X deviate to opposite sides", lo * lx < 0)

        # ---- homing: recover the az=60 rays from a perturbed launch direction
        for name, is_x in (("O", False), ("X", True)):
            ref = rays[("az=60", name)]
            guess = launch_dir(start, el_deg + 2.5, 60.0 - 1.5)
            t0 = time.perf_counter()
            hom = home_ray(chap, dip, f, start, ref.landing_point, guess, is_extraordinary=is_x)
            dt = time.perf_counter() - t0
            dir_err = np.degrees(np.arccos(np.clip(hom.launch_direction @ ref.launch_direction, -1, 1)))
            cmp_ = compare_to_path(hom.ray.points, ref.points[::40])
            print(
                "    home %s: converged=%s  miss %.4f m  launch-direction error %.2e deg  iterations %d  traces %d  "
                "%.2f s  | compare_to_path vs reference ray: max %.4f m, rms %.4f m"
                % (name, hom.converged, hom.miss_distance, dir_err, hom.iterations, hom.n_traces, dt,
                   cmp_["max_distance"], cmp_["rms_distance"])
            )
            check("home %s: miss < 1 m and launch direction recovered" % name,
                  hom.converged and hom.miss_distance < 1.0 and dir_err < 1e-4)
            check("home %s: compare_to_path max distance < 1 m" % name, cmp_["max_distance"] < 1.0)

        # ---- homing the X ray onto the O landing point (different launch angles needed)
        ref = rays[("az=60", "O")]
        t0 = time.perf_counter()
        hom = home_ray(chap, dip, f, start, ref.landing_point, ref.launch_direction, is_extraordinary=True)
        dt = time.perf_counter() - t0
        print(
            "    home X onto O landing point: converged=%s miss %.4f m, elevation %.5f deg, azimuth offset %+.5f deg, "
            "group path X %.3f m vs O %.3f m, %d traces, %.2f s"
            % (hom.converged, hom.miss_distance, np.degrees(hom.elevation), np.degrees(hom.azimuth_offset),
               hom.ray.group_path, ref.group_path, hom.n_traces, dt)
        )
        check("home X onto O landing point", hom.converged and hom.miss_distance < 1.0)

    # ---- compare_to_path sanity: a path offset by a known amount
    ref = rays[("az=60", "O")]
    if ref.landed:
        normal = np.cross(start, ref.launch_direction)
        normal /= np.linalg.norm(normal)
        shifted = ref.points[100:-100:57] + 123.0 * normal
        cmp_ = compare_to_path(ref.points, shifted)
        print("    compare_to_path on a path shifted 123 m sideways: max %.4f m, rms %.4f m"
              % (cmp_["max_distance"], cmp_["rms_distance"]))
        check("compare_to_path recovers a known 123 m offset",
              abs(cmp_["max_distance"] - 123.0) < 0.5 and abs(cmp_["rms_distance"] - 123.0) < 0.5)

    print("=" * 78)
    if failures:
        print("FAILED checks (%d):" % len(failures))
        for name in failures:
            print("   - " + name)
        raise SystemExit(1)
    print("All self-tests passed.")
