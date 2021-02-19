mod solver;
mod equations;

use std::array;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::prelude::{Array1};

#[pyfunction]
#[text_signature = "(ptr, length, /)"]
fn solve_yp_vector(ptr: *mut f64, len: usize) -> *mut f64 {
    let item = Vec::new();
    *item
}

#[pymodule]
fn rustion(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_yp_vector, m)?)?;

    Ok(())
}


#[cfg(test)]
mod tests {
    #[test]
    fn validated_mu_squared() {
        assert_eq!(2 + 2, 4);
    }
}
