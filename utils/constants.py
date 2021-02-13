EARTH_RADIUS = 6371E3  # m
EARTH_RADIUS_CUBED = 258596602811000000000 # m^3
STANDARD_MAGNETIC_FIELD_MAXIMUM = 3.12E-5  # T
E_CHARGE = 1.6022E-19  # Coulombs
E_MASS = 9.10938356E-31  # kg
C = 2.998E8  # m / s
PI = 3.141592653589793  # unit-less
EPSILON_0 = 8.854187817E-12  # F / m
B_FACTOR = E_CHARGE / (2 * PI * E_MASS)  # Used to calculate gyro frequency
TYPE_ABBREVIATION = {'QuasiParabolic': 'QP',
                     'GreatCircleDeviation': "GCD"}

INTEGRATION_STEP_SIZE_FACTOR = 1E4

DEFAULT_TRACER_PARALLEL_PARAMS = 50
DEFAULT_TRACER_NORMAL_PARAMS = 5
DEFAULT_TRACER_PARAMETERS = (
    DEFAULT_TRACER_PARALLEL_PARAMS,
    DEFAULT_TRACER_NORMAL_PARAMS
)
