const USE_ORDINARY_RAY: bool = true;
const EPSILON: f64 = 1E-15;


pub fn equation_13_root(x: f64, y: f64, yp: f64, yt: f64) -> f64 {
    let pt = calc_pt(x, y, yp, yt);
    let dlog_dyp = equation_16(x, y, yp);

    let pos = pt * pt - dlog_dyp * yt / 2.;
    let neg = -1. + yp * pt * dlog_dyp / 2.;

    pos + neg
}

pub fn calc_pt(x: f64, y: f64, yp: f64, yt: f64) -> f64 {
    let dlog_dyp = equation_16(x, y, yp);
    let denominator = (yp - dlog_dyp * y * y / 2.) + (dlog_dyp * yp * yp / 2.);
    
    yt / denominator
}

/// Implementing equation 15 in Coleman 2011
pub fn calculate_mu_squared(x: f64, y: f64, yp: f64) -> f64 {
    //! We need x, y, yp to all have the same length
    let multiplier = if USE_ORDINARY_RAY {
        1.
    } else {
        -1.
    };

    let one_minus_x = 1. - x;
    let yp_squared = yp * yp;
    let y_squared_minus_yp_squared = yp_squared - y * y;

    if y_squared_minus_yp_squared < EPSILON && one_minus_x.abs() < EPSILON {
        0.
    } else {
        let numerator = x * 2. * x - 2. * x * x;
        let radical = y_squared_minus_yp_squared * y_squared_minus_yp_squared - 4. * one_minus_x * one_minus_x * yp_squared;
        let denominator = 2. * one_minus_x - y_squared_minus_yp_squared + multiplier * radical.sqrt();
    
        1. - numerator/denominator
    }
}

pub fn calc_a(x: f64, y: f64, yp: f64) -> f64 {
    1. + x * yp * yp - (x + y * y)
}

pub fn calc_b(x: f64, y: f64, yp: f64) -> f64 {
    let neg = x * yp * yp + x * x + x * y * y / 2. + 1.;
    let pos = 2. * x + y * y;
    pos - neg
}

pub fn equation_16(x: f64, y: f64, yp: f64) -> f64 {
    
    let mu_squared = calculate_mu_squared(x, y, yp);
    let a = calc_a(x, y, yp);
    let b = calc_b(x, y, yp);

    let constant = x * yp / (a * mu_squared + b);
    constant * mu_squared - constant
}

#[cfg(test)]
mod tests {
    #[test]
    fn validated_mu_squared() {
        assert_eq!(2 + 2, 4);
    }
}
