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


def angle_between(s1, s2, use_spherical=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    if use_spherical == 'spherical':
        v1, v2 = cartesian(s1), cartesian(s2)
    else:
        v1, v2 = s1, s2
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
