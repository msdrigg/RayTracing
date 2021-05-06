
# I wanted to make these static methods of the Tracer class but I think this is faster,
#   and we really need the speed
import warnings
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize


epsilon = np.finfo(float).eps * 10


def equation_13(yp, x, y_squared, yt, sign=1):
    yp_squared = np.square(yp)

    fractions = equation_16(yp, yp_squared, x, y_squared)
    pt = equation_14(yp, yp_squared, y_squared, x, yt, sign=sign)

    output = -1 + pt * (pt - (yt - yp*pt) * fractions * 0.5)
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
        output = np.ones_like(yp)
    else:
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


def equation_15(yp_in, x_in, y_squared_in, sign=1):
    yp = np.asarray(yp_in)
    y_squared = np.asarray(y_squared_in)
    x = np.asarray(x_in)
    yp_squared = np.square(yp)

    defined_mask = np.abs(x - 1) > epsilon

    if sign == 1:
        output = np.zeros_like(yp)
    else:
        output = np.ones_like(yp)
    output[defined_mask] = 1 - 2 * x[defined_mask] * (1 - x[defined_mask]) / (
            2 * (1 - x[defined_mask]) - (y_squared[defined_mask] - yp_squared[defined_mask]) +
            sign * np.sqrt(np.square(y_squared[defined_mask] - yp_squared[defined_mask]) +
                           4 * np.square(1 - x[defined_mask]) * yp_squared[defined_mask])
    )

    if np.isscalar(yp_in):
        return output.item()
    else:
        return output


def equation_16(yp, yp_squared, x, y_squared, sign=1):
    a = 1 - x - y_squared + x * yp_squared
    beta = 2 * (1 - x) / (
            2 * (1 - x) - (y_squared - yp_squared) + sign *
            np.sqrt(np.square((y_squared - yp_squared)) + 4 * np.square(1 - x) * yp_squared)
    )
    return x * yp * beta / (1 + 0.5 * (yp_squared - y_squared) - a * beta - x)


def calculate_yp_pt_cheating(yt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    solved_yp = np.copy(yt)
    current_pt = np.ones_like(yt)
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
    if np.isscalar(x):
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
    if abs(x - 1) < epsilon:
        return 1
    if x > 1:
        warnings.warn(
            f"Cannot solve for yp when x > 1, and x={x}."
            "We will assume yp = 1"
        )
        return 1
    if abs(yt) < epsilon:
        return 0

    function_args = x, y_squared, yt, sign

    # noinspection PyTypeChecker
    yp_solved, details = optimize.brentq(
        equation_13, epsilon, y,
        args=function_args, xtol=1E-15, rtol=1E-15, full_output=True
    )
    if not details.converged:
        warnings.warn(
            f"Error solving for yp using equation 13. Attempted {details.iterations} iterations "
            f"and resulted in {details.root}, "
            f"stopping with reason: {details.flag}. "
            f"Returning yp = {yt} as a good guess."
        )
        return yt
    return yp_solved * np.sign(yt)
