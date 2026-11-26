# RayTracing

[![CC BY 4.0][cc-by-shield]][cc-by]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4734289.svg)](https://doi.org/10.5281/zenodo.4734289)



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
  1. Install Python on your computer (only tested with Python versions 3.9.4 and 3.8.5,
    but any version 3.7 or later should work)
  3. Clone the contents of this repository onto your computer.
  4. Open a terminal in the project's root directory
  5. (Optional) Setup a virtual environment to 
  store this repositories dependencies.
  [Checkout the docs!](https://docs.python.org/3/library/venv.html)
  5. Run `python3 -m pip install -r requirements.txt`
  to install the required packages.
  6. You're all set!

## Entrypoint

[main.py](main.py) is a working example of how to initialize a `Tracer` object and trace
the high and low, ordinary and extraordinary rays within an atmosphere-magnetic field
system.

    python3 main.py                 # show the figures
    python3 main.py --save plots    # write the figures to ./plots instead
    python3 main.py --gradient 0.375 --no-field

Each ray takes a few tenths of a second.

## Checking the results

* `python3 -m pytest tests` runs the checks of the physics and of the solver.
* [validation/haselgrove.py](validation/haselgrove.py) is an independent, conventional
  (Haselgrove equation) ray tracer with homing. It shares no code with the variational
  solver and is used to check its answers.
* `python3 benchmarks/coleman_tables.py` repeats the experiments of Tables 1 and 2 of
  Coleman's paper: a ray is traced with the Haselgrove equations and the direct method is
  then asked for the ray between the same end points with 24 to 200 segments. Phase path,
  apogee and lateral deviation agree to within meters.

## How it works

The phase path of a trial path is `P = integral of mu * pt ds` where `mu` is the
Appleton-Hartree refractive index and `pt` the cosine of the angle between the ray and the
wave normal. Both depend on the direction of the wave normal, which is found at every point
of the path from Coleman's equations 13 and 14
(see [tracing/coleman_equations.py](tracing/coleman_equations.py) for how these are reduced
to a single equation that always has a bracketed root). A ray is a path that makes `P`
stationary, which is found by Newton-Raphson iteration on the path parameters starting from
the analytic ray in a quasi-parabolic layer.

The gradient and Hessian of `P` are assembled with the chain rule from derivatives of the
integrand at each quadrature point (see [tracing/tracer.py](tracing/tracer.py)), so a Newton
step costs 73 vectorized evaluations of the integrand no matter how many parameters the
path has.

Things to know:

* Coleman's Figure 2 uses an IRI90 based ionosphere at 16 MHz and can't be reproduced with
  the simple layers here. The Chapman layer with gradients (foF2 = 7 MHz, hmF2 = 350 km,
  ymF2 = 100 km, 0.375 MHz per degree, 10 MHz) is the one behind his Tables 1 to 3.
* The paper doesn't state its exact Chapman formula, where the gradient is anchored or its
  magnetic field model, so absolute numbers differ from its tables by a few percent. See
  the `profile_exponent` argument of `ChapmanLayers`.
* If no ray of the requested type exists near the starting path (for example a high ray
  when the ionosphere becomes too dense along the path), `Tracer.trace` warns and
  `Tracer.converged` is False. The last path is then not a ray.
* Layers with kinks (the quasi-parabolic layer) converge slowly with the number of
  segments. Smooth layers converge to meters with 24 to 50 segments.

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

## Acknowledgements
This work was made possible with the help and support of Dr. Stephen Kaeppler. Thank you.

## Citing this Work
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by]. Please use [this doi link](https://doi.org/10.5281/zenodo.4734289) to cite this software.

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
