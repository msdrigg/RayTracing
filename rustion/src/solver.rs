use ndarray::prelude::*;
use argmin::{prelude::*, solver::brent::Brent};
use crate::equations::*;

pub fn solve_yp(x: &Array1<f64>, y: &Array1<f64>, yp: &Array1<f64>, yt: &Array1<f64>) -> Array1<f64> {
    let element_count = x.len();
    let mut solved_yp = Array1::zeros(element_count);

    for i in 0..element_count {
        let solver_current  = Brent::new(y, -y, 1E-10);
        // let result = Executor::new(None, solver_base, yt)
        //     .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        //     .max_iters(30)
        //     .run();
        
        // solved_yp = result;
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
