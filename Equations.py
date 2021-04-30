
# I wanted to make these static methods of the Tracer class but I think this is faster,
#   and we really need the speed
import numpy as np


def equation_13_new(yp, *args):
    yp2 = np.square(yp)
    x, y2, yt = args

    fractions = equation_16(yp, yp2, x, y2)
    pt = equation_14(yp, yp2, y2, fractions, yt)

    output = -1 + pt * (pt - yt * (1 - pt) * fractions * 0.5)
    return output


def equation_13_prime_new(yp, *args):
    x, y2, yt = args
    yp2 = np.square(yp)
    radical = np.sqrt(np.square(y2 - yp2) + 4 * yp2 * np.square(1 - x))
    a = 1 - x - y2 + x * yp2

    # Choosing ordinary ray
    alpha = 2 * (1 - x) / ((2 + yp2 + radical) - (2 * x + y2))
    d_alpha = -4 * (1 - x) * yp * (-1 + (-(4 * x + y2) +
                                         (yp2 + 2 + 2 * np.square(x))) / radical) / np.square(
        -(2 * x + y2) + (2 + yp2 + radical))

    frac = 2 * x * yp * alpha / (-(2 + yp2) + (y2 + 2 * a * alpha + 2 * x))
    d_frac = 2 * x * yp * d_alpha * (-2 + 2 * x + y2 - yp2) / np.square(-(2 * x + y2 + 2 * a * alpha) + (2 + yp2))

    pt = yt / (yp - 0.5 * frac * (y2 - yp2))
    d_pt = -2 * yt * (2 + 2 * yp * frac - (y2 - yp2) * d_frac) / np.square(2 * yp - (y2 - yp2) * frac)

    output = 0.5 * (pt * (frac * pt + (-yt + yp * pt) * d_frac) + (-yt * frac + 2 * (2 + yp * frac) * pt) * d_pt)
    return output


def equation_14(yp, yp2, y2, fractions, yt):
    return yt / (yp - (y2 - yp2) * fractions * 0.5)


def equation_15(yp, x, y2, sign=1):
    # We choose ordinary ray in our calculation of mu2
    yp2 = np.square(yp)

    return 1 - 2 * x * (1 - x) / (
            2 * (1 - x) - (y2 - yp2) + sign *
            np.sqrt(np.square(y2 - yp2) + 4 * np.square(1 - x) * yp2)
    )


def equation_16(yp, yp2, x, y2):
    a = 1 - x - y2 + x * yp2
    beta = 2 * (1 - x) / (
            2 * (1 - x) - (y2 - yp2) +
            np.sqrt(np.square((y2 - yp2)) + 4 * np.square(1 - x) * yp2)
    )
    return x * yp * beta / (1 + 0.5 * (yp2 - y2) - a * beta - x)
