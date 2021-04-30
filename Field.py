from abc import abstractmethod, ABC

from Vector import cartesian_to_spherical, unit_radius, unit_theta, unit_vector, spherical_to_cartesian
from Constants import B_FIELD, EARTH_RADIUS, B_FACTOR
from numpy import power, cos, sin, sqrt, square, array, asarray, repeat


class Field(ABC):
    @abstractmethod
    def field_vec(self, position, using_spherical=False):
        """
        This function returns a normalized magnetic field vector using the given position vector
        :param position - 3-vec that describes the position in either a cartesian or spherical coordinate system
        :param using_spherical - Boolean deciding the system of the position vector and return vector
        :returns The normalized magnetic field in the same coordinate system as the position vector
        """
        raise NotImplementedError("Inheriting classes must override the field_vec function")

    @abstractmethod
    def field_mag(self, position, using_spherical=False):
        """
        This function returns the magnitude of the magnetic field given the parameters described above
        """
        raise NotImplementedError("Inheriting classes must override the field_mag function")

    def gyro_frequency(self, position, using_spherical=False):
        """
        This function returns the gyro frequency at the given point using the class's defined functions for
        b_factor and field_mag
        """
        if using_spherical:
            position = spherical_to_cartesian(position)
        return B_FACTOR * self.field_mag(position)


class DipoleField(Field):
    def __init__(self):
        self.b_max = B_FIELD
        self.re3 = power(EARTH_RADIUS, 3)  # E_Radius^3

    def field_vec(self, position, using_spherical=False):
        if not using_spherical:
            position = cartesian_to_spherical(position).reshape(-1, 3)
        # radii cubed is 1/r^3 for all r
        radii_cubed = power(position[:, 0], -3)
        cos_thetas = cos(position[:, 1])
        b_vec = -2*self.b_max*radii_cubed.reshape(-1, 1)*self.re3*cos_thetas.reshape(-1, 1)*unit_radius(position) - \
            self.b_max*self.re3*radii_cubed.reshape(-1, 1)*sin(position[:, 1]).reshape(-1, 1)*unit_theta(position)
        b_vec = unit_vector(b_vec)

        if using_spherical:
            b_vec = cartesian_to_spherical(b_vec)

        if len(b_vec) == 1:
            return b_vec[0]
        else:
            return b_vec

    def field_mag(self, position, using_spherical=False):
        if not using_spherical:
            position = cartesian_to_spherical(position).reshape(-1, 3)
        # radii cubed is 1/r^3 for all r
        radii_cubed = power(position[:, 0], -3)
        cos_thetas = cos(position[:, 1])
        b_mags = self.b_max * self.re3 * radii_cubed * sqrt(1 + 3 * square(cos_thetas))
        if len(b_mags) == 1:
            return b_mags[0]
        else:
            return b_mags


class ZeroField(Field):
    def field_vec(self, position, using_spherical=False):
        b_vec = array([1, 0, 0])
        if not using_spherical:
            b_vec = spherical_to_cartesian(b_vec)

        return b_vec

    def field_mag(self, position, using_spherical=False):
        return repeat(0, len(position))

    def gyro_frequency(self, position, using_spherical=False):
        return repeat(0, len(position))
