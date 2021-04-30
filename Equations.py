
# I wanted to make these static methods of the Tracer class but I think this is faster,
#   and we really need the speed
import warnings
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize


epsilon = np.finfo(float).eps * 10


def equation_13_new(yp, *args):
    yp2 = np.square(yp)
    x, y_squared, yt, sign = args

    fractions = equation_16(yp, yp2, x, y_squared)
    pt = equation_14(yp, yp2, y_squared, fractions, yt, sign=sign)

    output = -1 + pt * (pt - yt * (1 - pt) * fractions * 0.5)
    return output


def equation_14(
        yp_in: Union[float, ArrayLike],
        yp_squared_in: Union[float, ArrayLike],
        y_squared_in: Union[float, ArrayLike],
        x_in: Union[float, ArrayLike],
        yt_in: Union[float, ArrayLike],
        sign=1
) -> Union[float, np.ndarray]:
    yp = np.asarray(yp_in)
    yp_squared = np.asarray(yp_squared_in)
    y_squared = np.asarray(y_squared_in)
    x = np.asarray(x_in)
    yt = np.asarray(yt_in)
    fractions = np.zeros_like(yp)
    defined_mask = y_squared > epsilon
    if not np.any(defined_mask):
        return np.ones_like(yp)

    fractions[defined_mask] = equation_16(
        yp[defined_mask],
        yp_squared[defined_mask],
        x[defined_mask],
        y_squared[defined_mask],
        sign=sign
    )
    output = np.ones_like(yp)
    output[defined_mask] = yt[defined_mask] / (
            yp[defined_mask] -
            (y_squared[defined_mask] - yp_squared[defined_mask]) *
            fractions[defined_mask] * 0.5
    )
    if np.isscalar(yp):
        return output.item()
    else:
        return output


def equation_15(yp, x, y2, sign=1):
    # We choose ordinary ray in our calculation of mu2
    yp2 = np.square(yp)

    return 1 - 2 * x * (1 - x) / (
            2 * (1 - x) - (y2 - yp2) + sign *
            np.sqrt(np.square(y2 - yp2) + 4 * np.square(1 - x) * yp2)
    )


def equation_16(yp, yp2, x, y2, sign=1):
    a = 1 - x - y2 + x * yp2
    beta = 2 * (1 - x) / (
            2 * (1 - x) - (y2 - yp2) + sign *
            np.sqrt(np.square((y2 - yp2)) + 4 * np.square(1 - x) * yp2)
    )
    return x * yp * beta / (1 + 0.5 * (yp2 - y2) - a * beta - x)


def calculate_yp_pt_cheating(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    solved_yp = np.zeros(r.shape[0])
    current_pt = np.repeat(1, r.shape[0])
    return solved_yp, current_pt


# TODO: Iron out wrinkles
def calculate_yp_pt_real(
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        y_squared: Union[float, ArrayLike],
        yt: Union[float, ArrayLike],
        sign=1
) -> Tuple[np.ndarray, np.ndarray]:
    solved_yp = calculate_yp(x, y, y_squared, yt, sign=sign)
    current_pt = equation_14(solved_yp, np.square(solved_yp), y_squared, x, yt, sign=sign)
    return solved_yp, current_pt


def calculate_yp(
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        y_squared: Union[float, ArrayLike],
        yt: Union[float, ArrayLike],
        sign=1
):
    """
    Given the parameters, return the solution to yp
    :return: yp solved
    """
    if isinstance(x, float):
        return calculate_yp_float(x, y, y_squared, yt)
    else:
        outputs = y.copy()
        for i in range(len(x)):
            outputs[i] = calculate_yp_float(x[i], y[i], y_squared[i], yt[i], sign=sign)
        return outputs


def calculate_yp_float(
        x: float,
        y: float,
        y_squared: float,
        yt: float,
        sign=1
) -> float:
    if y_squared < epsilon:
        return 0

    function_args = x, y_squared, yt, sign

    # noinspection PyTypeChecker
    yp_solved, details = optimize.brentq(
        equation_13_new, -y - epsilon, y + epsilon,
        args=function_args, xtol=1E-15, rtol=1E-15, full_output=True
    )
    if not details.converged:
        warnings.warn(
            f"Error solving for yp using equation 13. Attempted {details.iterations} iterations "
            f"and resulted in {details.root}, "
            f"stopping with reason: {details.flag}."
        )
    return yp_solved
