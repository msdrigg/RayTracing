"""
This file holds equations that come from Coleman 2011.
I will cite each equation as it comes from that paper.
"""

import numpy as np
from numpy.typing import *


def equation_13(yp: float, x: float, y_squared: float, yt: float) -> float:
    """
    Calculating the output of equation 13 from Coleman 2011
    This generates pt using equation 14
    """
    yp2 = np.square(yp)

    fractions = equation_16(yp, yp2, x, y_squared)
    pt = equation_14(yp, yp2, y_squared, fractions, yt)

    output = -1 + pt * (pt - yt * (1 - pt) * fractions * 0.5)
    return output


def equation_13_prime(yp: float, x: float, y_squared: float, yt: float) -> float:
    """
    Calculates the derivative of equation 13 in Coleman 2011
    with respect to yp
    """
    yp_squared = np.square(yp)
    radical = np.sqrt(np.square(y_squared - yp_squared) + 4 * yp_squared * np.square(1 - x))
    a = 1 - x - y_squared + x * yp_squared

    # Choosing ordinary ray
    alpha = 2*(1-x)/((2 + yp_squared + radical) - (2 * x + y_squared))
    d_alpha = -4*(1-x)*yp*(-1 + (-(4 * x + y_squared) + (yp_squared + 2 + 2*np.square(x)))/radical) / \
        np.square(-(2 * x + y_squared) + (2 + yp_squared + radical))

    frac = 2*x*yp*alpha/(-(2 + yp_squared) + (y_squared + 2 * a * alpha + 2 * x))
    d_frac = 2 * x * yp * d_alpha * (-2 + 2 * x + y_squared - yp_squared) / \
        np.square(-(2 * x + y_squared + 2 * a * alpha) + (2 + yp_squared))

    pt = yt/(yp - 0.5 * frac * (y_squared - yp_squared))
    d_pt = -2 * yt * (2 + 2 * yp * frac - (y_squared - yp_squared) * d_frac) / \
        np.square(2 * yp - (y_squared - yp_squared) * frac)

    output = 0.5*(pt*(frac*pt + (-yt + yp*pt)*d_frac) + (-yt*frac + 2*(2 + yp*frac)*pt)*d_pt)

    return output


def equation_14(yp: ArrayLike,
                yp_squared: ArrayLike,
                y_squared: ArrayLike,
                fractions: ArrayLike,
                yt: ArrayLike) -> ArrayLike:
    """
    Equation 14 from Coleman 2011
    """
    return yt / (yp - (y_squared - yp_squared) * fractions * 0.5)


def equation_15(yp_squared: np.ndarray, x: np.ndarray, y_squared: np.ndarray) -> np.ndarray:
    """
    Equation 15 from Coleman 2011
    """
    # We choose ordinary ray in our calculation of mu2
    one_minus_x = 1 - x
    y_squared_minus_yp_squared = y_squared - yp_squared

    denominator = 2 * one_minus_x - y_squared_minus_yp_squared + \
        np.sqrt(np.square(y_squared_minus_yp_squared) + 4 * np.square(one_minus_x) * yp_squared)

    return 1 - 2 * x * one_minus_x / denominator


def equation_16(yp: ArrayLike, yp_squared: ArrayLike, x: ArrayLike, y_squared: ArrayLike) -> ArrayLike:
    """
    Equation 16 from Coleman 2011.
    """
    a = 1 - x - y_squared + x * yp_squared
    beta = 2 * (1 - x) / (2 * (1 - x) - (y_squared - yp_squared) +
                          np.sqrt(np.square((y_squared - yp_squared)) + 4 * np.square(1 - x) * yp_squared))

    return x * yp * beta / (1 + 0.5 * (yp_squared - y_squared) - a * beta - x)
