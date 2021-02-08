import math


def assert_is_close(num1, num2, rel_tol=1E-9, abs_tol=0.0):
    """
    Asserts that math.isclose returns true
    """
    if not math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AssertionError(f"Numbers are not within expected tolerance: abs_tol={rel_tol}, rel_tol={rel_tol}\n"
                             f"Number 1: {num1}, \nNumber 2: {num2}")
