import numpy as np


def cartesian(spherical_vector):
    """Returns the cartesian form of the spherical vector."""
    spherical_vector = spherical_vector.reshape(-1, 3)
    output = np.vstack([(spherical_vector[:, 0]*np.sin(spherical_vector[:, 1])*np.cos(spherical_vector[:, 2])),
                        spherical_vector[:, 0]*np.sin(spherical_vector[:, 1])*np.sin(spherical_vector[:, 2]),
                        spherical_vector[:, 0]*np.cos(spherical_vector[:, 1])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def spherical(cartesian_vector):
    """Returns the cartesian form of the spherical vector."""
    cartesian_vector = cartesian_vector.reshape(-1, 3)
    r = np.linalg.norm(cartesian_vector, axis=1)
    output = np.vstack([r, np.arccos(cartesian_vector[:, 2]/r),
                        np.arctan2(cartesian_vector[:, 1], cartesian_vector[:, 0])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norms = np.linalg.norm(vector.reshape(-1, 3), axis=1)
    output = vector / norms.reshape(-1, 1)
    if len(output) == 1:
        return output[0]
    else:
        return output


def angle_between(s1, s2, use_spherical=False):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    if use_spherical:
        v1, v2 = cartesian(s1), cartesian(s2)
    else:
        v1, v2 = s1, s2
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    output = np.arccos(np.clip(np.sum(v1_u.reshape((-1, 3))*v2_u.reshape((-1, 3)), axis=1), -1.0, 1.0))
    if len(output) == 1:
        return output[0]
    else:
        return output
