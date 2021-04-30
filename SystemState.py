from Atmosphere import Atmosphere
from Field import Field


class SystemState:
    def __init__(
            self, field: Field,
            atmosphere: Atmosphere,
            operating_frequency: float,
            is_extraordinary_ray: bool
    ):
        self.field, self.atmosphere = field, atmosphere
        self.operating_frequency = operating_frequency
        self.is_extraordinary_ray = is_extraordinary_ray
