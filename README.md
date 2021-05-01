# RayTracing

Python implementation of Coleman's 2011 Paper. 
The structure of the folder is defined below.

All vectors will be stored and used in spherical coordinates 
in the ISO convention of (r, theta, phi) = (radial, polar, azimuthal)

## Tracer.py

File that contains the core code to trace generalized rays.

### Inputs

* **Known Parameters**: 
  The ray's starting and ending points

* **Ionosphere Model**: 
  Function that returns the electron density given 
  a position in space in a coordinate system

* **Magnetic Field**: 
  Class that returns the earth's magnetic field 
  when called with a position in space in a coordinate system

* **Initial Path**: 
  Path Class representing the ion's path given 
  the starting and ending points and the atmosphere parameters too

## Layers.py

File that contains the code to generate the chapman layers 
given whatever input parameters necessary

### Inputs

* **f_0**: Maximum magnetic field
* **hm**: Height of maximum magnetic field
* **ym**: Field width parameter

## Field.py

  File that contains the code to calculate the earth's magnetic field given a point in space

## Constants.py

  File that contains known constants for geometry and physical constants