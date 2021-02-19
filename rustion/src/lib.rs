use ndarray::prelude::*;
use argmin::{prelude::*, solver::brent::Brent};

const USE_ORDINARY_RAY: bool = true;
const EPSILON: f64 = 1E-15;

fn yp_pt_minimization_function(x: f64, y: f64, yt: f64) -> f64{
    0.
}

// Implement the cost function as a ArgminOperator
// 

fn equation_13(x: Array1<f64>, y: Array1<f64>, yp: Array1<f64>) -> Array1<f64> {
    Array1::zeros(20)
}

fn equation_14(x: Array1<f64>, y: Array1<f64>, yp: Array1<f64>) -> Array1<f64> {
    Array1::zeros(20)
}

/// Implementing equation 15 in Coleman 2011
fn calculate_mu_squared(x: &Array1<f64>, y: &Array1<f64>, yp: &Array1<f64>) -> Array1<f64> {
    //! We need x, y, yp to all have the same length
    let n = (&x).len();
    let mut output = ndarray::Array::zeros((n, ));

    let multiplier = if USE_ORDINARY_RAY {
        1.
    } else {
        -1.
    };

    for i in 0..n {
        let one_minus_x = 1. - &x[i];
        let yp_squared = &yp[i] * &yp[i];
        let y_squared_minus_yp_squared = &yp_squared - &y[i] * &y[i];
        output[i] = if &y_squared_minus_yp_squared < &EPSILON && &one_minus_x.abs() < &EPSILON {
            0.
        } else {
            let numerator = &x[i] * 2. * x[i] - 2. * x[i] * x[i];
            let radical = &y_squared_minus_yp_squared * &y_squared_minus_yp_squared;
            let denominator = 2. * one_minus_x - y_squared_minus_yp_squared + &multiplier * radical.sqrt();
        
            1. - numerator/denominator
        };
    }

    output
}

fn equation_16(x: Array1<f64>, y: Array1<f64>, yp: Array1<f64>) -> Array1<f64> {
    let mu_squared = calculate_mu_squared(&x, &y, &yp);
    let numerator = 0.;
    x.clone()
}

fn solve_yp(x: Array1<f64>, y: Array1<f64>, yt: Array1<f64>) -> Array1<f64> {
    let element_count = x.len();
    let mut solved_yp = Array1::zeros(element_count);

    for i in 0..element_count {
        let solver_current  = Brent::new(y[i], -y[i], 1E-10);
        // let result = Executor::new(None, solver_base, yt[i])
        //     .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        //     .max_iters(30)
        //     .run();
        
        // solved_yp[i] = result;
    }

    solved_yp
}

#[cfg(test)]
mod tests {
    #[test]
    fn validated_mu_squared() {
        assert_eq!(2 + 2, 4);
    }
}
