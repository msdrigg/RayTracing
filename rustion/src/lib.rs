mod solver;
mod equations;

use solver::solve_yp;

use pyo3::prelude::*;
use pyo3::exceptions;

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

#[pymodule]
fn rustion(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // wrapper of `axpy`
    #[pyfn(m, "solve_yp")]
    fn solve_yp_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
        yt: PyReadonlyArrayDyn<f64>
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let yt = yt.as_array();
        
        match solve_yp(&x, &y, &yt) {
            Err(e) => {
                let error_message: String = format!("Error calculating yp, {}", e);
                let error: PyErr = exceptions::PyValueError::new_err(error_message);
                Err(error)
            },
            Ok(res) => Ok(res.into_pyarray(py))
        }
    }

    Ok(())
}