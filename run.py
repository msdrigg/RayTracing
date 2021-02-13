"""
Use this file by changing imports
Set the atmosphere by 'from atmosphere import xxxxx as atmosphere'
Set the magnetic field by 'from magnetic field import xxxxx as magnetic field'
You can also change the atmosphere used by manually passing the correct 'calculate_plasma_frequency' function
to the trace() call. The required atmosphere functions are described in atmosphere.base and magnetic.base
"""

if __name__ == "__main__":
    field = BasicField()
    initial = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5, 133.7])))
    final = Vector.spherical_to_cartesian(
        Vector.latitude_to_spherical(
            array([coords.EARTH_RADIUS, 90 + 23.5 - 10, 133.7])))
    atmosphere = ChapmanLayers(7E6, 350E3, 100E3, (0.375E6 * 180 / math.pi, -1), initial)
    path_generator = QuasiParabolic
    frequency = 10E6  # Hz
    atmosphere.visualize(initial, final, ax=None, fig=None, point_number=400, show=True)
    basic_tracer = Tracer(frequency, atmosphere, field, path_generator)
    basic_tracer.parameters = (50, 0)
    basic_tracer.parameter_number = 50
    basic_tracer.initial_coordinates, basic_tracer.final_coordinates = initial, final

    basic_tracer.compile_initial_path()
    # basic_tracer.visualize(plot_all=True)

    paths = basic_tracer.trace()
