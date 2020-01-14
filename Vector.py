from numpy.linalg import norm
from numpy import cos, arccos, sin, arctan2, vstack, clip, sum
from Constants import PI, EARTH_RADIUS


def spherical_to_cartesian(spherical_vector):
    """Returns the cartesian form of the spherical vector."""
    spherical_vector = spherical_vector.reshape(-1, 3)
    output = vstack([(spherical_vector[:, 0]*sin(spherical_vector[:, 1])*cos(spherical_vector[:, 2])),
                    spherical_vector[:, 0]*sin(spherical_vector[:, 1])*sin(spherical_vector[:, 2]),
                    spherical_vector[:, 0]*cos(spherical_vector[:, 1])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def cartesian_to_spherical(cartesian_vector):
    """Returns the cartesian form of the spherical vector."""
    cartesian_vector = cartesian_vector.reshape(-1, 3)
    r = norm(cartesian_vector, axis=1)
    output = vstack([r, arccos(cartesian_vector[:, 2]/r),
                     arctan2(cartesian_vector[:, 1], cartesian_vector[:, 0])]).T
    if len(output) == 1:
        return output[0]
    else:
        return output


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    norms = norm(vector.reshape(-1, 3), axis=1)
    output = vector / norms.reshape(-1, 1)
    if len(output) == 1:
        return output[0]
    else:
        return output


def angle_between(s1, s2, use_spherical=False):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    if use_spherical:
        v1, v2 = spherical_to_cartesian(s1), spherical_to_cartesian(s2)
    else:
        v1, v2 = s1, s2
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    output = arccos(clip(sum(v1_u.reshape((-1, 3))*v2_u.reshape((-1, 3)), axis=1), -1.0, 1.0))
    if len(output) == 1:
        return output[0]
    else:
        return output


def unit_radius(position):
    position = cartesian_to_spherical(position).reshape(-1, 3)
    sin_phi = sin(position[:, 2])
    unit_radii = vstack([cos(position[:, 1])*sin_phi,
                         sin(position[:, 1])*sin_phi,
                         cos(position[:, 2])]).T
    if len(unit_radii) == 1:
        return unit_radii[0]
    else:
        return unit_radii


def unit_theta(position):
    position = cartesian_to_spherical(position).reshape(-1, 3)
    cos_phi = cos(position[:, 2])
    unit_thetas = vstack([cos(position[:, 1])*cos_phi,
                         sin(position[:, 1])*cos_phi,
                         -sin(position[:, 2])]).T
    if len(unit_thetas) == 1:
        return unit_thetas[0]
    else:
        return unit_thetas


def latitude_to_spherical(vector):
    vector = vector.reshape(-1, 3)
    vector[:, 1] = 90 - vector[:, 1]
    vector[:, 1:] *= PI / 180.0

    if len(vector) == 1:
        return vector[0]
    else:
        return vector
