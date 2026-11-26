import numpy as np

from utilities.Constants import B_FIELD, EARTH_RADIUS
from magnetic_fields import BaseField


class DipoleField(BaseField):
    """
    Implementation of BaseField that follows the dipole model of earth's magnetic field.
    The dipole is at the earth's centre and aligned with the rotation (z) axis:

        B = -B0 (R_E / r)^3 [ 2 cos(theta) r_hat + sin(theta) theta_hat ]
          = -B0 (R_E / r)^3 [ 3 cos(theta) r_hat - z_hat ]

    (theta is the colatitude) so the field points up/outward in the southern
    hemisphere, down in the northern hemisphere and north at the equator.
    """
    def __init__(self, b_max: float = B_FIELD):
        # Field magnitude at the equator on the ground
        self.b_max = b_max

    @staticmethod
    def _dipole_direction(position):
        position = np.asarray(position, dtype=float).reshape(-1, 3)
        r = np.linalg.norm(position, axis=1)
        r_hat = position / r[:, np.newaxis]
        cos_theta = r_hat[:, 2]
        # -(3 cos(theta) r_hat - z_hat), whose norm is sqrt(1 + 3 cos^2(theta))
        direction = -3 * cos_theta[:, np.newaxis] * r_hat
        direction[:, 2] += 1
        return r, cos_theta, direction

    def field_vec(self, position):
        _, cos_theta, direction = self._dipole_direction(position)
        b_vec = direction / np.sqrt(1 + 3 * np.square(cos_theta))[:, np.newaxis]
        if np.ndim(position) == 1:
            return b_vec[0]
        return b_vec

    def field_mag(self, position):
        r, cos_theta, _ = self._dipole_direction(position)
        b_mags = self.b_max * np.power(EARTH_RADIUS / r, 3) * np.sqrt(1 + 3 * np.square(cos_theta))
        if np.ndim(position) == 1:
            return b_mags[0]
        return b_mags


class ZeroField(BaseField):
    """
    Implementation of Base field that always returns 0
    """
    def field_vec(self, position):
        # The direction of a zero field is arbitrary, but it must still be a unit vector
        position = np.asarray(position, dtype=float)
        output = np.zeros_like(position)
        output[..., 2] = 1
        return output

    def field_mag(self, position):
        return np.zeros(np.shape(position)[:-1])

    def gyro_frequency(self, position):
        return np.zeros(np.shape(position)[:-1])
