//! Savitzky-Golay filter implementation.
use std::fmt::Debug;

use lstsq::lstsq;
use na::{DMatrix, DVector};
use nalgebra as na;
use polyfit_rs::polyfit_rs::polyfit;

fn factorial(num: usize) -> usize {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

fn savgol_coeffs(
    window_length: usize,
    poly_order: usize,
    deriv: usize,
) -> Result<DVector<f64>, String> {
    if poly_order >= window_length {
        return Err("poly_order must be less than window_length.".to_string());
    }

    let (halflen, rem) = (window_length / 2, window_length % 2);

    let pos = match rem {
        0 => halflen as f64 - 0.5,
        _ => halflen as f64,
    };

    if deriv > poly_order {
        return Ok(DVector::from_element(window_length, 0.0));
    }

    let x = DVector::from_fn(window_length, |i, _| pos - i as f64);
    let order = DVector::from_fn(poly_order + 1, |i, _| i);
    let mat_a = DMatrix::from_fn(poly_order + 1, window_length, |i, j| {
        x[j].powf(order[i] as f64)
    });

    let mut y = DVector::from_element(poly_order + 1, 0.0);
    y[deriv] = factorial(deriv) as f64;

    let epsilon = 1e-14;
    let results = lstsq(&mat_a, &y, epsilon)?;
    let solution = results.solution;

    return Ok(solution);
}

fn poly_derivative(coeffs: &[f64]) -> Vec<f64> {
    coeffs[1..]
        .iter()
        .enumerate()
        .map(|(i, c)| c * (i + 1) as f64)
        .collect()
}

fn polyval(poly: &[f64], values: &[f64]) -> Vec<f64> {
    return values
        .iter()
        .map(|v| {
            poly.iter()
                .enumerate()
                .fold(0.0, |y, (i, c)| y + c * v.powf(i as f64))
        })
        .collect();
}

fn fit_edge(
    x: &DVector<f64>,
    window_start: usize,
    window_stop: usize,
    interp_start: usize,
    interp_stop: usize,
    poly_order: usize,
    deriv: usize,
    y: &mut Vec<f64>,
) -> Result<(), String> {
    let x_edge: Vec<f64> = x.as_slice()[window_start..window_stop].to_vec();
    let y_edge: Vec<f64> = (0..window_stop - window_start).map(|i| i as f64).collect();
    let mut poly_coeffs = polyfit(&y_edge, &x_edge, poly_order)?;

    let mut deriv = deriv;
    while deriv > 0 {
        poly_coeffs = poly_derivative(&poly_coeffs);
        deriv -= 1;
    }

    let i: Vec<f64> = (0..interp_stop - interp_start)
        .map(|i| (interp_start - window_start + i) as f64)
        .collect();
    let values = polyval(&poly_coeffs, &i);
    y.splice(interp_start..interp_stop, values);
    Ok(())
}

fn fit_edges_polyfit(
    x: &DVector<f64>,
    window_length: usize,
    poly_order: usize,
    deriv: usize,
    y: &mut Vec<f64>,
) -> Result<(), String> {
    let halflen = window_length / 2;
    fit_edge(x, 0, window_length, 0, halflen, poly_order, deriv, y)?;
    let n = x.len();
    fit_edge(
        x,
        n - window_length,
        n,
        n - halflen,
        n,
        poly_order,
        deriv,
        y,
    )?;

    Ok(())
}

#[derive(Clone, Debug)]
pub struct SavGolInput<'a, T> {
    pub data: &'a [T],
    pub window_length: usize,
    pub poly_order: usize,
    pub derivative: usize,
}

pub fn savgol_filter<T>(input: &SavGolInput<T>) -> Result<Vec<f64>, String>
where
    T: Clone + TryInto<f64>,
    <T as TryInto<f64>>::Error: Debug,
{
    if input.window_length > input.data.len() {
        return Err(
            "window_length must be less than or equal to the size of the input data".to_string(),
        );
    }

    if input.window_length % 2 == 0 {
        // TODO: figure out how scipy implementation handles the convolution
        // in this case
        return Err("window_length must be odd".to_string());
    }

    let coeffs = savgol_coeffs(input.window_length, input.poly_order, input.derivative)?;

    let x = match input.data.iter().cloned().map(|i| i.try_into()).collect() {
        Err(error) => return Err(format!("{:?}", error)),
        Ok(x) => DVector::from_vec(x)
    };

    let y = x.convolve_full(coeffs);

    // trim extra length gained during convolution to mimic scipy convolve1d
    // with mode="constant"
    let padding = y.len() - x.len();
    let padding = padding / 2;
    let y = y.as_slice();
    let mut y = y[padding..y.len().saturating_sub(padding)].to_vec();

    fit_edges_polyfit(
        &x,
        input.window_length,
        input.poly_order,
        input.derivative,
        &mut y,
    )?;
    return Ok(y);
}
