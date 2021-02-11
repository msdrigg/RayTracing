EARTH_RADIUS = 6371E3  # m
B_FIELD = 3.12E-5  # T
E_CHARGE = 1.6022E-19  # Coulombs
E_MASS = 9.10938356E-31  # kg
C = 2.998E8  # m / s
PI = 3.141592653589793  # unit-less
EPSILON_0 = 8.854187817E-12  # F / m
B_FACTOR = E_CHARGE / (2 * PI * E_MASS)  # Used to calculate gyro frequency
TYPE_ABBREVIATION = {'QuasiParabolic': 'QP',
                     'GreatCircleDeviation': "GCD"}

error_codes = {
    0: "Improper input parameters were entered.",
    1: "The solution converged.",
    2: "The number of calls to function has "
       "reached max_fev = %d." % 200,
    3: "x_tol=%f is too small, no further improvement "
       "in the approximate\n  solution "
       "is possible." % 1E-7,
    4: "The iteration is not making good progress, as measured "
       "by the \n  improvement from the last five "
       "Jacobian evaluations.",
    5: "The iteration is not making good progress, "
       "as measured by the \n  improvement from the last "
       "ten iterations.",
       'unknown': "An error occurred."
}

DEFAULT_TRACER_PARALLEL_PARAMS = 50
DEFAULT_TRACER_NORMAL_PARAMS = 5
DEFAULT_TRACER_PARAMETERS = (
    DEFAULT_TRACER_PARALLEL_PARAMS,
    DEFAULT_TRACER_NORMAL_PARAMS
)
