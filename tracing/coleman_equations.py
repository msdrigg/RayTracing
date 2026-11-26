"""
The magneto-ionic relations used by the direct variational method of
Coleman (2011), "Point-to-point ionospheric ray tracing by a direct
variational method", Radio Science 46, RS5016.

Notation (all dimensionless):
    X   = (plasma frequency / wave frequency)^2
    Y   = gyro frequency / wave frequency, Y_vec = Y * (unit vector along B)
    t   = unit tangent of the ray path,       Yt = Y_vec . t   (known from the path)
    p   = unit wave normal,                   Yp = Y_vec . p   (unknown)
    pt  = p . t                                               (unknown)
    mu  = phase refractive index (Appleton-Hartree, no collisions)

Fermat's principle for an anisotropic medium is  delta  int mu * pt ds = 0.

Where the equations come from
-----------------------------
The ray (group) direction is the normal of the refractive index surface.
Writing L = d(ln mu^2)/dYp this is

    t = pt * [ p - (L / 2) * (Y_vec - Yp * p) ]                          (*)

Dotting (*) with t and with Y_vec gives the two coupled equations of the
paper (numbered 13 and 14 there, 3 and 4 in Report.pdf):

    1  = pt * [ pt - (L / 2) * (Yt - Yp * pt) ]                          (13)
    Yt = pt * [ Yp - (L / 2) * (Y^2 - Yp^2) ]                            (14)

Taking the squared norm of (*) instead gives pt explicitly in terms of Yp,

    pt = 1 / sqrt(1 + (L^2 / 4) * (Y^2 - Yp^2))                          (**)

so the pair (13), (14) collapses to ONE scalar equation in Yp:

    G(Yp) = pt(Yp) * [ Yp - (L / 2) * (Y^2 - Yp^2) ] - Yt = 0

G(-Y) = -Y - Yt <= 0 and G(+Y) = Y - Yt >= 0 because |Yt| <= Y, so a root is
always bracketed by [-Y, Y] and a bisection can never fail. This is what
`solve_yp_pt` does. (Solving (13) with (14) substituted, as the original code
did, has spurious roots and poles, and Yp does not have to have the sign of
Yt, so bracketing on [0, Y] is wrong.)

Everything here is vectorised over numpy arrays.
"""
from typing import Tuple

import numpy as np

# sign = +1 -> ordinary ray, sign = -1 -> extraordinary ray
ORDINARY = 1
EXTRAORDINARY = -1


def _denominator_terms(yp, x, y_squared, sign):
    s = y_squared - np.square(yp)
    one_minus_x = 1 - x
    radical = np.sqrt(np.square(s) + 4 * np.square(one_minus_x) * np.square(yp))
    denominator = 2 * one_minus_x - s + sign * radical
    return s, one_minus_x, radical, denominator


def equation_15(yp, x, y_squared, sign=1):
    """
    Appleton-Hartree refractive index squared (Coleman eq. 15).
    sign=+1 is the ordinary ray, sign=-1 the extraordinary ray.
    """
    yp, x, y_squared = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (yp, x, y_squared)))
    s, one_minus_x, radical, denominator = _denominator_terms(yp, x, y_squared, sign)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu2 = 1 - 2 * x * one_minus_x / denominator
    # Where the denominator vanishes (only possible at X = 1) the limit is used
    mu2 = np.where(denominator == 0, 1.0 if sign == -1 else 0.0, mu2)
    return mu2


def equation_16(yp, x, y_squared, sign=1):
    """
    L = d(ln mu^2)/dYp (Coleman eq. 16), obtained by differentiating eq. 15.
    Algebraically identical to  -X Yp (mu^2 - 1) / (A mu^2 + B)  but free of
    the 0/0 that form has when Y -> 0.
    """
    yp, x, y_squared = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (yp, x, y_squared)))
    s, one_minus_x, radical, denominator = _denominator_terms(yp, x, y_squared, sign)
    safe_radical = np.where(radical > 0, radical, 1.0)
    d_denominator = 2 * yp * (1 + sign * (2 * np.square(one_minus_x) - s) / safe_radical)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu2 = 1 - 2 * x * one_minus_x / denominator
        d_mu2 = 2 * x * one_minus_x / np.square(denominator) * d_denominator
        return d_mu2 / mu2


def pt_from_yp(yp, x, y_squared, sign=1):
    """p . t as an explicit function of Yp (equation (**) in the module docstring)."""
    log_derivative = equation_16(yp, x, y_squared, sign)
    s = np.maximum(y_squared - np.square(yp), 0)
    return 1 / np.sqrt(1 + 0.25 * np.square(log_derivative) * s)


def equation_14(yp, x, y_squared, sign=1):
    """
    Right hand side of Coleman eq. 14: the value of Yt that corresponds to a
    given Yp. `solve_yp_pt` inverts this function.
    """
    log_derivative = equation_16(yp, x, y_squared, sign)
    s = np.maximum(y_squared - np.square(yp), 0)
    pt = 1 / np.sqrt(1 + 0.25 * np.square(log_derivative) * s)
    return pt * (yp - 0.5 * log_derivative * s)


def equation_13(yp, x, y_squared, yt, sign=1):
    """
    Residual of Coleman eq. 13 with pt taken from (**). It vanishes at the
    solution; used in tests as an independent check on `solve_yp_pt`.
    """
    log_derivative = equation_16(yp, x, y_squared, sign)
    pt = pt_from_yp(yp, x, y_squared, sign)
    return pt * (pt - 0.5 * log_derivative * (yt - yp * pt)) - 1


def solve_yp_pt(x, y, yt, sign=1, iterations=60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Coleman's eqs. 13 and 14 for Yp and pt.

    :param x: X, shape (N,)
    :param y: Y (magnitude, >= 0), shape (N,)
    :param yt: Y_vec . t, shape (N,), |yt| <= y
    :param sign: +1 ordinary, -1 extraordinary
    :param iterations: bisection steps. 60 halvings of [-Y, Y] reach double precision.
    :returns: (Yp, pt), each of shape (N,)
    """
    x, y, yt = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, yt)))
    y_squared = np.square(y)
    low, high = -y, y.copy()
    with np.errstate(invalid='ignore'):
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            too_small = equation_14(mid, x, y_squared, sign) < yt
            low = np.where(too_small, mid, low)
            high = np.where(too_small, high, mid)
    yp = 0.5 * (low + high)
    return yp, pt_from_yp(yp, x, y_squared, sign)


def calculate_yp_pt_isotropic(yt) -> Tuple[np.ndarray, np.ndarray]:
    """
    The isotropic approximation p = t (Yp = Yt, pt = 1). Exact when Y = 0.
    This was the "cheater" solver of the original code.
    """
    yt = np.asarray(yt, dtype=float)
    return yt.copy(), np.ones_like(yt)


def ray_refractive_index(x, y, yt, sign=1, isotropic_approximation=False) -> np.ndarray:
    """
    The integrand of Fermat's principle per unit arc length, mu * pt, as a
    function of the path direction (through Yt). NaN where the wave is
    evanescent (mu^2 <= 0).
    """
    x, y, yt = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, yt)))
    if isotropic_approximation:
        yp, pt = calculate_yp_pt_isotropic(yt)
    else:
        yp, pt = solve_yp_pt(x, y, yt, sign)
    mu2 = equation_15(yp, x, np.square(y), sign)
    with np.errstate(invalid='ignore'):
        return np.where(mu2 > 0, np.sqrt(np.abs(mu2)) * pt, np.nan)
