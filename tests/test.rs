#[macro_use]
extern crate approx;

use itertools::Itertools;
use savgol_rs::{savgol_filter, SavGolInput};
use serde_json;
use subprocess::Exec;

use std::f64;

fn scipy_savgol(input: &SavGolInput<f64>) -> Result<Vec<f64>, String> {
    let data_str: Vec<String> = input
        .data
        .clone()
        .into_iter()
        .map(|i| i.to_string())
        .collect();
    let data_str: Vec<&str> = data_str.iter().map(|i| i.as_str()).collect();

    let window_length = input.window_length.to_string();
    let poly_order = input.poly_order.to_string();
    let derivative = input.derivative.to_string();

    let args = [
        &["run", "savgol", "--input"],
        data_str.as_slice(),
        &[
            "--window",
            window_length.as_str(),
            "--poly-order",
            poly_order.as_str(),
            "--derivative",
            derivative.as_str(),
        ],
    ]
    .concat();

    let scipy_result = Exec::cmd("poetry")
        .args(&args)
        .stdout(subprocess::Redirection::Pipe)
        .capture()
        .expect("Emulation failed!")
        .stdout_str();

    match serde_json::from_str(&scipy_result) {
        Ok(result) => Ok(result),
        Err(_) => Err(scipy_result),
    }
}

// fuzzy equality check
fn assert_savgol_filter_eq(a: &Vec<f64>, b: &Vec<f64>) {
    a.into_iter()
        .zip_eq(b.into_iter())
        .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = f64::EPSILON * 1000.0));
}

const VALUES: &[f64] = &[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.1, 1.3, 1.2, 1.9,
    2.2, 2.3, 2.6, 3.1, 3.5, 3.9, 4.2, 4.5, 4.6, 5.1, 6.0, 6.6, 7.0, 8.6, 9.8, 10.5, 11.1, 12.1,
    13.3, 14.3, 14.7, 15.3, 16.9, 17.6, 18.6, 20.0, 20.9, 21.7, 21.9, 22.6, 23.3, 23.3, 23.5, 24.0,
    24.1, 24.5, 24.7, 24.6, 25.2, 25.2, 25.4, 25.5, 25.6, 26.0, 26.3, 26.7, 26.9, 26.9, 27.0, 27.3,
    27.5, 27.6, 28.2, 28.7, 28.2, 28.6, 28.8, 29.0, 29.2, 29.6, 30.1, 30.0, 30.2, 30.3, 30.7, 30.7,
    30.7, 31.0, 31.4, 31.6, 31.9, 32.0, 32.1, 32.9, 33.2, 33.1, 33.1, 33.4, 33.9, 34.1, 34.1, 34.6,
    34.7, 34.9, 35.0, 35.4, 35.5, 35.8, 36.2, 36.5,
];

#[test]
fn zeroth_derivative() {
    let input = SavGolInput {
        data: VALUES,
        window_length: 11,
        poly_order: 2,
        derivative: 0,
    };

    let result = savgol_filter(&input);
    let expected = scipy_savgol(&input);

    match (result, expected) {
        (Ok(result), Ok(expected)) => assert_savgol_filter_eq(&result, &expected),
        _ => assert!(false),
    };
}

#[test]
fn first_derivative() {
    let input = SavGolInput {
        data: VALUES,
        window_length: 11,
        poly_order: 2,
        derivative: 1,
    };

    let result = savgol_filter(&input);
    let expected = scipy_savgol(&input);

    match (result, expected) {
        (Ok(result), Ok(expected)) => assert_savgol_filter_eq(&result, &expected),
        _ => assert!(false),
    };
}

#[test]
fn second_derivative() {
    let input = SavGolInput {
        data: VALUES,
        window_length: 11,
        poly_order: 2,
        derivative: 2,
    };

    let result = savgol_filter(&input);
    let expected = scipy_savgol(&input);

    match (result, expected) {
        (Ok(result), Ok(expected)) => assert_savgol_filter_eq(&result, &expected),
        _ => assert!(false),
    };
}

#[test]
fn integer_input() {
    let input = SavGolInput {
        data: &[1, 5, 3, 4, 7, 2, 4, 6],
        window_length: 3,
        poly_order: 1,
        derivative: 0,
    };

    let result = savgol_filter(&input);
    assert_eq!(
        result,
        Ok(vec![
            2.0000000000000004,
            3.0000000000000004,
            4.0,
            4.666666666666667,
            4.333333333333334,
            4.333333333333334,
            4.0,
            6.000000000000001
        ])
    );
}
