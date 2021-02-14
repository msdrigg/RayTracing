"""
Use this file by changing imports
Set the atmosphere by 'from atmosphere import <chosen atmosphere> as atmosphere'
Set the magnetic field by 'from magnetic field import <chosen field> as magnetic field'
You can also change the atmosphere used by manually passing the correct 'calculate_plasma_frequency' function
to the trace() call. The required atmosphere functions are described in atmosphere.base and magnetic.base
"""
import numpy as np
from core import tracing, coordinates as coords

# List of module names for magnetic fields
IMPLEMENTED_MAGNETIC_FIELDS = [
    'dipole',
    'zero'
]

# List of module names for atmospheres
IMPLEMENTED_ATMOSPHERES = [
    'chapman',
    'quasi_parabolic'
]

if __name__ == "__main__":
    atmosphere = IMPLEMENTED_ATMOSPHERES[0]
    atmosphere_parameters = (350E3 + coords.EARTH_RADIUS, 100E3 + coords.EARTH_RADIUS, 7E6)
    magnetic_field = IMPLEMENTED_MAGNETIC_FIELDS[1]  #

    operating_frequency = 10E6  # MHZ
    parameter_definitions = 50, 0

    path_start_point_latitude_longitude = np.array([coords.EARTH_RADIUS, 90 + 23.5, 133.7])  # Degrees
    path_start_point = coords.geographic_to_spherical(path_start_point_latitude_longitude)  # Spherical/radians
    path_end_point_latitude_longitude = np.array([coords.EARTH_RADIUS, 90 + 23.5 - 10, 133.7])  # Degrees
    path_end_point = coords.geographic_to_spherical(path_end_point_latitude_longitude)  # Spherical/radians

    calculated_paths = tracing.trace(
        path_start_point,
        path_end_point,
        atmosphere,
        magnetic_field,
        operating_frequency,
        atmosphere_parameters,
        arrows=True
    )
