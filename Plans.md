# Future Changes
Outline of future changes.
This will serve as a checklist for the project
# Ongoing
1. Finish implementing current efforts
    1. paths.generate_cartesian_callable
    2. tracing.integrate_over_path
    3. tracing main function for quick testing
    4. tracing.trace generate the path_components evenly spaced
    5. magnetic.dipole
    6. magnetic.zero (make sure it uses cartesian and r_norm)
    7. All callable's -- make sure they accept and use r_norm
2. Test current algorithms
    1. Test new plotting features
    2. Test tracer
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
## Implementation
1. Implement solution in magnetic field case
    1. Re-profile with this new algorithm
    2. Optimize where necessary
    3. Sanity tests and quantitative tests
## Restructuring
1. Rewrite the class-based implementation using functions
2. Reorganize the code to make sense. I currently rely too heavily on utils.*
3. Write a high level wrapper that runs the backend code with the least amount of user input
## Optimizations
1. Profile one newton raphson step 
to determine where the most important optimizations need to occur.
   1. Generate profile outputs, and flame diagrams
2. Investigate whether other implementations would improve some core functionality
    1. Rust + open-blas for low level processing
    2. Julia for something
    3. Python+Ray or Multiprocess.Array to replace raw multiprocessing for 
    sending large numpy arrays between processes
    4. See https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html?highlight=scipy%20integrate%20simps
    for how to pass C functions to scipy integration
    5. Switch to rompberg integration -- Need 2^k + 1 equally spaced points
        1. DONE, NEEDS TESTING
    6. Profile different solvers for y_p, p_t
        1. Notice that y_p and p_t are both constrained to be between -1 and 1
3. Investigate algorithms for solving the hessian linear system
    1. Use scipy.linalg.lstqs or scipy.linalg.pinvh
        1. TEST RESULTS: lstqs does not work. pinvh does, and 
        pinvh is marginally faster as well.
        2. Unless pinvh fails in practice or can be majorly
        then we will settle on this method
    2. Dynamic parameter positions depending on their variability