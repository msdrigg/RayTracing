class BasicField:
    def field_vec(self, position, using_spherical=False):
        """
        This function returns a normalized magnetic field vector using the given position vector
        :param position - 3-vec that describes the position in either a cartesian or spherical coordinate system
        :param using_spherical - Boolean deciding the system of the position vector and return vector
        :returns The normalized magnetic field in the same coordinate system as the position vector
        """
        # TODO: Implement this
        #   This must be vectorized i.e. coordinates may be a stack of vectors then plasma_frequency will return
        #   a stack of values
        pass

    def gyro_frequency(self, position, using_spherical=False):
        # TODO: Implement this
        #         # This must be vectorized i.e. coordinates may be a stack of vectors then plasma_frequency will return
        #         # a stack of values
        pass
