#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use argmin::{prelude::*, solver::brent::Brent};

fn yp_pt_minimization_function(x: f64, y: f64, yt: f64) -> f64{
    0;
}

// Implement the cost function as a ArgminOperator
// 

fn equation_14(x: Array1<f64>, y: Array1<f64>, yp: Array1<f64>) -> Array1<f64> {
    Array1::zeros(20)
}

fn solve_yp(x: Array1<f64>, y: Array1<f64>, yt: Array1<f64>) -> Array1<f64> {
    let element_count = x.len();
    let mut solved_yp = Array1::zeros(element_count);

    for i in 0..element_count {
        let solver_current  = Brent::new(y[i], -y[i], 1E-10);
        let result = Executor::new(None, solver_base, yt[i])
            .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
            .max_iters(30)
            .run()?;
        
        solved_yp[i] = result;
    }

    solved_yp
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
