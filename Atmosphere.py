from Vector import cartesian, spherical


class ChapmanLayers:
    def __init__(self):
        self._parameters = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_params):
        self._parameters = new_params
        self.compile()

    def electron_density(self, coordinate, use_spherical=False):
        # TODO: Do I even need this?
        #   I dont need unless its necessary to calculate gyro or plasma frequency
        pass

    def plasma_frequency(self, coordinate, use_spherical=False):
        if use_spherical:
            coordinate = cartesian(coordinate)
        # TODO: Implement this
        #   This must be vectorized i.e. coordinates may be a stack of vectors then plasma_frequency will return
        #   a stack of values

        # Must convert back to spherical for return
        output = coordinate + self._parameters
        if use_spherical:
            output = spherical(output)
        if len(output) == 1:
            return output[0]
        else:
            return output

    def compile(self):
        # TODO: Implement this
        #   This will initialize the backend including any poly_fits to return the atmospheric components
        pass
