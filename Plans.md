# Future Changes
Outline of future changes.
This will serve as a checklist for the project
# Ongoing
1. Test current algorithms
    1. Repair test_quasi_parabolic
        1. We changed the order of parameters, so some may need adjusting
        2. We also switched to solely gyro_frequency^2 not e_density
    2. Test all new utils
    3. Test new plotting features
    4. Test tracer
## Testing
1. Retest the core functionality around the zero-field implementation
    1. Test solving perturbations of 
       QP solutions in QP atmosphere
    2. Test solving QP solutions in Chapman layers atmosphere
    3. Test using generated results with PHARLARP
2. Write tests for small functions
    1. Plotting functions for points and paths
    2. Functions in utils.algorithms
    3. Functions in utils.paths
3. Write tests for magnetic field case
    1. Ask Dr. Kaeppler for baseline results
    2. Generate baseline results with PHARLARP
## Optimizations
1. Profile one newton raphson step 
to determine where the most important optimizations need to occur.
   1. Generate profile outputs, and flame diagrams
2. Investigate whether other implementations would improve some core functionality
    1. Rust + open-blas for low level processing
    2. Julia for something
    3. Python+Ray to replace raw multiprocessing for 
    sending large numpy arrays between processes
        1. DONE, NEEDS TESTING: I am using multiprocessing shared_memory now. I don't know if this is better
        It is supposed to be faster, but I'm not sure
    4. See https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html?highlight=scipy%20integrate%20simps
    for how to pass C functions to scipy integration
    5. Switch to rompberg integration -- Need 2^k + 1 equally spaced points
        1. DONE, NEEDS TESTING
    6. Profile different solvers for y_p, p_t
        1. Notice that y_p and p_t are both constrained to be between -1 and 1
        2. We also note that a good guess for y_p is y_t
        3. Currently trying toms748, which is fast, but written in python
        4. Could use a version of brent's which is written in C
        5. Need this HEAVILY optimized because it is called nearly 5 million times
        each iteration
3. Investigate algorithms for solving the hessian linear system
    1. DONE, NEEDS TESTING
    2. Use scipy.linalg.lstqs or scipy.linalg.pinvh
        1. TEST RESULTS: lstqs does not work. pinvh does, and 
        pinvh is marginally faster as well.
        2. Unless pinvh fails in practice or can be majorly
        then we will settle on this method
    3. Dynamic parameter positions depending on their variability