# RayTracing
 Python implementiation of Coleman's 2011 Paper

Documentation:
# Tracer.py
 File that contains the core code to trace generalized rays.
	:inputs
		Known Parameters: The ray's starting and ending points
		Ionosphere Model: Function that returns the electron density given a position in space in a corrdinate system
		Magnetic Field: Function that returns the earth's magnetic field given a position in a coordinate system
		Initial Path: Function that returns the ion's path given the starting and ending points (maybe the atmosphere parameters too?)
		Other stuff:??

# Layers.py
  File that contains the code to generate the chapman layers given whatever input parameters necessary
	:inputs
		Whatever Chapman Layers need to Generate?

# Field.py
  File that contains the code to calculate the earth's magnetic field given a point in space
	:inputs
		Should be a fixed function or a zero function

# Parabolic.py
  File that contains the code to calculate the initial parabolic path that is our guess for a good ray tracing path