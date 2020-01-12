import numpy as np


def cartesian(spherical_vector):
    """Returns the cartesian form of the spherical vector."""
    return np.array([spherical_vector[0]*np.sin(spherical_vector[1])*np.cos(spherical_vector[2]),
                     spherical_vector[0]*np.sin(spherical_vector[1])*np.sin(spherical_vector[2]),
                     spherical_vector[0]*np.cos(spherical_vector[1])])


def spherical(cartesian_vector):
    """Returns the cartesian form of the spherical vector."""
    r = np.linalg.norm(cartesian_vector)
    return np.array([r, np.arccos(cartesian_vector[2]/r),
                     np.arctan2(cartesian_vector[1], cartesian_vector[0])])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(s1, s2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1, v2 = cartesian(s1), cartesian(s2)
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotate(v_i, axis, angle):
    """Returns the vector found by rotating v_i by angle radians CCW about the given axis"""
    pass


if __name__ == "__main__":
    test1 = np.array([1, .5, .9])
    test1_cart = cartesian(test1)
    test2 = np.array([1, .8, .7])
    test2_cart = cartesian(test2)
    shift = np.array([0, 2, 3])
    print(f"Angle Between: {angle_between(s1=test1, s2=test2)}")
    print(f"Difference Vector: {test1-test2}")

    print(f"Difference Norm: {np.linalg.norm(np.array([0.3, 0.2]))}")
