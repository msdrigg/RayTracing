class ChapmanLayers:
    # TODO: Implement this
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
        # TODO: Implement this
        pass

    def plasma_frequency(self, coordinate, use_spherical=False):
        # TODO: Implement this
        pass

    def compile(self):
        # TODO: Implement this
        pass
