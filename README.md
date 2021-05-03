# RayTracing

[![CC BY 4.0][cc-by-shield]][cc-by]

## Introduction

Python implementation of a direct variational ray tracing method. 
This algorithm is outlined in 
the paper by Christopher Coleman in his 
[2011 paper](https://doi.org/10.1029/2011RS004748)
in the Radio Science journal.

For this codebase, spherical vectors
will be stored and used in the ISO convention of 
(r, theta, phi) = (radial, polar, azimuthal). 
However, cartesian vectors will be preferred where 
possible.

## Setup

To set up this repository, perform the following steps
  1. Install Python on your computer
  2. Clone the contents of this repository onto your computer.
  3. Open a terminal in the project's root directory
  4. (Optional) Setup a virtual environment to 
  store this repositories dependencies.
  [Checkout the docs!](https://docs.python.org/3/library/venv.html)
  5. Run `python3 -m pip install -r requirements.txt`
  to install the required packages.
  6. You're all set!

## Entrypoint

[main.py](main.py) is a file contains that a working example of how 
to initialize a `Tracer` object and trace 
different rays within an atmosphere-magnetic field 
system.

To execute this file, run `python3 main.py` with no 
arguments. This file will run the code and generate 
figures showing the convergence.

## Standardized Components
Atmospheres, Magnetic Fields, and Paths are all standardized
with easy ways to create new implementations. To achieve this 
standardization, I have a base class for each of these types. 
For example, `BaseAtmosphere` is the base class for 
atmospheres, and it can be found in the 
[atmospheres/base.py](atmospheres/base.py) file. 
Look in the `magnetic_fields` and `paths` directories for 
similar file structure. 

All implementations of these components need to be in the
`implementations.py` file in their respective directories. For 
example, the `ChapmanLayers` class, which subclasses 
`BaseAtmosphere`, is in the 
[atmospheres/implementations.py](atmospheres/implementations.py) file.

## Tracing Core

 [tracing/tracer.py](tracing/tracer.py) is a file that contains the core code to trace generalized rays.

 This file primarily contains the `Tracer` class
 which contains information about the path 
 and system. The method `Tracer.trace` can 
 be used to find a valid path given the necessary 
 inputs describing the system. For each trace, 
 the following inputs must be provided.

### Inputs

* **Known Path Parameters**: The ray's starting 
  and ending points, and the 
path's variable parameters.
for the path.

* **Ionosphere Model**: Object responsible for 
  calculating the plasma frequency at any position in space. 
  This object needs to subclass the `BaseAtmosphere` class.

* **Magnetic Field**: Object responsible for
calculating the earth's magnetic field at any given 
position in space. This object needs to subclass 
the `BaseField` class.

* **Initial Path**: Path object representing 
an approximate ray path to begin iterating over. 
This object must subclass the `BasePath` class.

## Conclusion
Please review the [supporting paper](Report.pdf), 
and read the comments provided within the code. Good luck tracing!

## Citing this Work
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by]. Please use [this doi link](https://doi.org/10.5281/zenodo.4734289) to cite this software.

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
