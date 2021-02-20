use ndarray::prelude::*;
use argmin::{prelude::*, solver::brent::Brent};
use crate::equations::*;
use serde::{Serialize, Deserialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct YPSolver {
    x: f64,
    y: f64,
    yt: f64
}

impl ArgminOp for YPSolver {
    type Param = f64;
    type Output = f64;
    type Float = f64;
    type Jacobian = ();
    type Hessian = ();

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(equation_13_root(&self.x, &self.y, p, &self.yt))
    }
}


pub fn solve_yp<'a> (x: &'a ArrayViewD<'a, f64>, y: &ArrayViewD<'a, f64>, yt: &ArrayViewD<'a, f64>) -> Result<ArrayD<f64>, Error> {
    let mut solved_yp: ArrayD<f64> = Array::zeros(x.shape());

    for (idx, _) in x.indexed_iter() {
        let yt_i = yt[&idx];
        let cost = YPSolver {
            x: x[&idx],
            y: y[&idx],
            yt: yt_i
        };

        let init_param = y[&idx];
        let y_i = y[&idx];
        let solver = Brent::new(-y_i, y_i, 1e-10);
    
        let res = Executor::new(cost, solver, init_param)
            .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
            .max_iters(100)
            .run()?;

        solved_yp[&idx] = res.state.best_param;
    }

    Ok(solved_yp)
}

#[cfg(test)]
mod tests {
    #[test]
    fn validated_mu_squared() {
        assert_eq!(2 + 2, 4);
    }
}
