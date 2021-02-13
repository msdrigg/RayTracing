"""
This file holds equations that come from Coleman 2011.
I will cite each equation as it comes from that paper.
"""

import numpy as np


def equation_13(yp, x, y2, yt):
    """
    Calculating the output of equation 13 from Coleman 2011
    This generates pt using equation 14
    """
    yp2 = np.square(yp)

    fractions = equation_16(yp, yp2, x, y2)
    pt = equation_14(yp, yp2, y2, fractions, yt)

    output = -1 + pt * (pt - yt * (1 - pt) * fractions * 0.5)
    return output


def equation_13_prime(yp, x, y2, yt):
    """
    Calculates the derivative of equation 13 in Coleman 2011
    with respect to yp
    """
    yp2 = np.square(yp)
    radical = np.sqrt(np.square(y2 - yp2) + 4*yp2*np.square(1 - x))
    a = 1 - x - y2 + x*yp2

    # Choosing ordinary ray
    alpha = 2*(1-x)/((2 + yp2 + radical) - (2*x + y2))
    d_alpha = -4*(1-x)*yp*(-1 + (-(4*x + y2) +
                                 (yp2 + 2 + 2*np.square(x)))/radical)/np.square(-(2*x + y2) + (2 + yp2 + radical))

    frac = 2*x*yp*alpha/(-(2 + yp2) + (y2 + 2*a*alpha + 2*x))
    d_frac = 2*x*yp*d_alpha*(-2 + 2*x + y2 - yp2)/np.square(-(2*x + y2 + 2*a*alpha) + (2 + yp2))

    pt = yt/(yp - 0.5*frac*(y2 - yp2))
    d_pt = -2*yt*(2 + 2*yp*frac - (y2 - yp2)*d_frac)/np.square(2*yp - (y2 - yp2)*frac)

    output = 0.5*(pt*(frac*pt + (-yt + yp*pt)*d_frac) + (-yt*frac + 2*(2 + yp*frac)*pt)*d_pt)

    return output


def equation_14(yp, yp_squared, y_squared, fractions, yt):
    """
    Equation 14 from Coleman 2011
    """
    return yt / (yp - (y_squared - yp_squared) * fractions * 0.5)


def equation_15(yp_squared, x, y_squared):
    """
    Equation 15 from Coleman 2011
    """
    # We choose ordinary ray in our calculation of mu2
    one_minus_x = 1 - x
    y_squared_minus_yp_squared = y_squared - yp_squared

    denominator = 2 * one_minus_x - y_squared_minus_yp_squared + \
        np.sqrt(np.square(y_squared_minus_yp_squared) + 4 * np.square(one_minus_x) * yp_squared)

    return 1 - 2 * x * one_minus_x / denominator


def equation_16(yp, yp2, x, y2):
    a = 1 - x - y2 + x * yp2
    beta = 2 * (1 - x) / (2 * (1 - x) - (y2 - yp2) +
                          np.sqrt(np.square((y2 - yp2)) + 4 * np.square(1 - x) * yp2))

    return x * yp * beta / (1 + 0.5 * (yp2 - y2) - a*beta - x)
