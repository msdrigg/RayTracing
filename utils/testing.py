"""
General testing utils
"""
import math


def assert_is_close(num1: float, num2: float, rel_tol: float = 1E-9, abs_tol: float = 0.0):
    """
    Asserts that math.isclose returns true
    """
    if not math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AssertionError(f"Numbers are not within expected tolerance: "
                             f"abs_tol={rel_tol}, rel_tol={rel_tol}\n"
                             f"Number 1: {num1}, \nNumber 2: {num2}")
