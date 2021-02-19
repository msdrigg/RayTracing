use ndarray::prelude::{Array};

fn convert_pointer_to_ndarray(ptr: *mut f64, length: usize) -> Array1<f64> {
    let data = unsafe {
        Array::from_vec(Vec::from_raw_parts(ptr, length, length))
    };
    
    data
}

fn convert_array_to_pointer(array: &Array64)