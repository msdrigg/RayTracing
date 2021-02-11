# Future Changes
Outline of future changes.
This will serve as a checklist for the project
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
2. Investigate whether rust/julia would improve some core functionality
3. Investigate algorithms for solving the hessian linear system
    1. Conditioning the matrix
    2. Dynamic parameter positions depending on their variability
    3. Separate the hessians into 2 matrices 
       (1 for height and one for normal components)
    4. Throw out ill-conditioned parameters
       from the hessian all together. (Radial values)