use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const MIN_LIKELIHOOD: f64 = 1e-300;
const F2_PRIOR_SCALE: [f64; 3] = [4.0, 2.0, 4.0];
const F2_CONSTANT: [f64; 9] = [0.25, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25];
const F2_LINEAR: [f64; 9] = [-0.5, 0.5, 0.0, 0.5, -1.0, 0.5, 0.0, 0.5, -0.5];
const F2_QUADRATIC: [f64; 9] = [0.25, -0.5, 0.25, -0.5, 1.0, -0.5, 0.25, -0.5, 0.25];

type PyVectorPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type PyMatrixPair<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

struct BinaryPairConfig {
    marker_count: usize,
    offspring_count: usize,
    maximum_recombination: f64,
    bisection_iterations: usize,
    beta_prior_shape: f64,
}

fn binary_pair(
    probabilities: &[f64],
    left: usize,
    right: usize,
    config: &BinaryPairConfig,
) -> (f64, f64) {
    let mut same = Vec::with_capacity(config.offspring_count);
    let mut delta = Vec::with_capacity(config.offspring_count);
    let mut flat_sum = 0.0;
    for offspring in 0..config.offspring_count {
        let first = probabilities[offspring * config.marker_count + left];
        let second = probabilities[offspring * config.marker_count + right];
        let same_value = (1.0 - first) * (1.0 - second) + first * second;
        let delta_value = first * (1.0 - second) + (1.0 - first) * second - same_value;
        same.push(same_value);
        delta.push(delta_value);
        flat_sum += delta_value * delta_value;
    }
    let flat = flat_sum <= 1e-20;
    let prior_exponent = config.beta_prior_shape - 1.0;
    let score = |value: f64| -> f64 {
        let mut total = same
            .iter()
            .zip(&delta)
            .map(|(base, change)| change / (base + change * value).max(MIN_LIKELIHOOD))
            .sum::<f64>();
        if prior_exponent > 0.0 {
            let bounded = value.clamp(1e-12, 1.0 - 1e-12);
            total += prior_exponent * (1.0 / bounded - 1.0 / (1.0 - bounded));
        }
        total
    };

    let mut low = if prior_exponent > 0.0 { 1e-12 } else { 0.0 };
    let mut high = config.maximum_recombination;
    let at_zero = score(low) <= 0.0;
    let at_maximum = score(high) >= 0.0;
    if !flat && !at_zero && !at_maximum {
        for _ in 0..config.bisection_iterations {
            let middle = (low + high) / 2.0;
            if score(middle) > 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
    }
    let fitted = if flat || at_maximum {
        config.maximum_recombination
    } else if at_zero {
        0.0
    } else {
        (low + high) / 2.0
    };
    let linked = same
        .iter()
        .zip(&delta)
        .map(|(base, change)| (base + change * fitted).max(MIN_LIKELIHOOD).ln())
        .sum::<f64>();
    let unlinked = same
        .iter()
        .zip(&delta)
        .map(|(base, change)| (base + 0.5 * change).max(MIN_LIKELIHOOD).ln())
        .sum::<f64>();
    (
        fitted,
        ((linked - unlinked) / std::f64::consts::LN_10).max(0.0),
    )
}

#[pyfunction]
#[pyo3(signature = (
    probabilities,
    left,
    right,
    maximum_recombination = 0.499999,
    bisection_iterations = 32,
    beta_prior_shape = 1.0
))]
fn pairwise_recombination_edges<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArray2<'py, f64>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    maximum_recombination: f64,
    bisection_iterations: usize,
    beta_prior_shape: f64,
) -> PyResult<PyVectorPair<'py>> {
    let shape = probabilities.shape();
    let offspring_count = shape[0];
    let marker_count = shape[1];
    let values = probabilities
        .as_slice()
        .map_err(|_| PyValueError::new_err("probabilities must be C-contiguous"))?;
    let left_indices = left
        .as_slice()
        .map_err(|_| PyValueError::new_err("left must be contiguous"))?;
    let right_indices = right
        .as_slice()
        .map_err(|_| PyValueError::new_err("right must be contiguous"))?;
    if left_indices.len() != right_indices.len() {
        return Err(PyValueError::new_err(
            "edge arrays must have matching lengths",
        ));
    }
    let pairs = left_indices
        .iter()
        .zip(right_indices)
        .map(|(&first, &second)| {
            let first = usize::try_from(first)
                .map_err(|_| PyValueError::new_err("edge index cannot be negative"))?;
            let second = usize::try_from(second)
                .map_err(|_| PyValueError::new_err("edge index cannot be negative"))?;
            if first >= marker_count || second >= marker_count || first == second {
                return Err(PyValueError::new_err("edge index is out of bounds"));
            }
            Ok((first, second))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let config = BinaryPairConfig {
        marker_count,
        offspring_count,
        maximum_recombination,
        bisection_iterations,
        beta_prior_shape,
    };
    let results = pairs
        .par_iter()
        .map(|&(first, second)| binary_pair(values, first, second, &config))
        .collect::<Vec<_>>();
    let (recombination, lod): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    Ok((recombination.into_pyarray(py), lod.into_pyarray(py)))
}

fn f2_coefficients(
    probabilities: &[f64],
    marker_count: usize,
    offspring: usize,
    left: usize,
    right: usize,
) -> (f64, f64, f64) {
    let left_start = (offspring * marker_count + left) * 3;
    let right_start = (offspring * marker_count + right) * 3;
    let mut constant = 0.0;
    let mut linear = 0.0;
    let mut quadratic = 0.0;
    for first in 0..3 {
        let first_likelihood = probabilities[left_start + first] * F2_PRIOR_SCALE[first];
        for second in 0..3 {
            let product =
                first_likelihood * probabilities[right_start + second] * F2_PRIOR_SCALE[second];
            let index = first * 3 + second;
            constant += product * F2_CONSTANT[index];
            linear += product * F2_LINEAR[index];
            quadratic += product * F2_QUADRATIC[index];
        }
    }
    (constant, linear, quadratic)
}

fn f2_pair(
    probabilities: &[f64],
    marker_count: usize,
    offspring_count: usize,
    left: usize,
    right: usize,
    maximum_recombination: f64,
    bisection_iterations: usize,
) -> (f64, f64) {
    let mut constant = Vec::with_capacity(offspring_count);
    let mut linear = Vec::with_capacity(offspring_count);
    let mut quadratic = Vec::with_capacity(offspring_count);
    let mut flat_sum = 0.0;
    for offspring in 0..offspring_count {
        let (base, slope, curve) =
            f2_coefficients(probabilities, marker_count, offspring, left, right);
        constant.push(base);
        linear.push(slope);
        quadratic.push(curve);
        flat_sum += slope * slope + curve * curve;
    }
    let flat = flat_sum <= 1e-20;
    if flat {
        return (maximum_recombination, 0.0);
    }
    let mut low = 0.0;
    let mut high = maximum_recombination;
    for _ in 0..bisection_iterations {
        let middle = (low + high) / 2.0;
        let middle_squared = middle * middle;
        let score = (0..offspring_count)
            .map(|index| {
                let denominator =
                    (constant[index] + linear[index] * middle + quadratic[index] * middle_squared)
                        .max(MIN_LIKELIHOOD);
                (linear[index] + 2.0 * quadratic[index] * middle) / denominator
            })
            .sum::<f64>();
        if score > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let fitted = (low + high) / 2.0;
    let fitted_squared = fitted * fitted;
    let linked = (0..offspring_count)
        .map(|index| {
            (constant[index] + linear[index] * fitted + quadratic[index] * fitted_squared)
                .max(MIN_LIKELIHOOD)
                .ln()
        })
        .sum::<f64>();
    let unlinked = (0..offspring_count)
        .map(|index| {
            (constant[index] + 0.5 * linear[index] + 0.25 * quadratic[index])
                .max(MIN_LIKELIHOOD)
                .ln()
        })
        .sum::<f64>();
    (
        fitted,
        ((linked - unlinked) / std::f64::consts::LN_10).max(0.0),
    )
}

#[pyfunction]
#[pyo3(signature = (
    probabilities,
    maximum_recombination = 0.499999,
    bisection_iterations = 32
))]
fn f2_pairwise_recombination<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArray3<'py, f64>,
    maximum_recombination: f64,
    bisection_iterations: usize,
) -> PyResult<PyMatrixPair<'py>> {
    let shape = probabilities.shape();
    let offspring_count = shape[0];
    let marker_count = shape[1];
    if shape[2] != 3 {
        return Err(PyValueError::new_err(
            "F2 probabilities must have three states",
        ));
    }
    let values = probabilities
        .as_slice()
        .map_err(|_| PyValueError::new_err("probabilities must be C-contiguous"))?;
    let pairs = (1..marker_count)
        .flat_map(|left| (0..left).map(move |right| (left, right)))
        .collect::<Vec<_>>();
    let results = pairs
        .par_iter()
        .map(|&(left, right)| {
            f2_pair(
                values,
                marker_count,
                offspring_count,
                left,
                right,
                maximum_recombination,
                bisection_iterations,
            )
        })
        .collect::<Vec<_>>();
    let mut recombination = vec![0.0; marker_count * marker_count];
    let mut lod = vec![0.0; marker_count * marker_count];
    for ((left, right), (rate, score)) in pairs.into_iter().zip(results) {
        recombination[left * marker_count + right] = rate;
        recombination[right * marker_count + left] = rate;
        lod[left * marker_count + right] = score;
        lod[right * marker_count + left] = score;
    }
    let recombination = Array2::from_shape_vec((marker_count, marker_count), recombination)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let lod = Array2::from_shape_vec((marker_count, marker_count), lod)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((recombination.into_pyarray(py), lod.into_pyarray(py)))
}

#[pymodule]
fn _softmap_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pairwise_recombination_edges, module)?)?;
    module.add_function(wrap_pyfunction!(f2_pairwise_recombination, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_config() -> BinaryPairConfig {
        BinaryPairConfig {
            marker_count: 2,
            offspring_count: 4,
            maximum_recombination: 0.499999,
            bisection_iterations: 32,
            beta_prior_shape: 1.0,
        }
    }

    #[test]
    fn identical_binary_markers_fit_zero_recombination() {
        let probabilities = [0.01, 0.01, 0.99, 0.99, 0.01, 0.01, 0.99, 0.99];
        let (rate, lod) = binary_pair(&probabilities, 0, 1, &binary_config());
        assert_eq!(rate, 0.0);
        assert!(lod > 0.0);
    }

    #[test]
    fn uninformative_f2_markers_fit_the_unlinked_boundary() {
        let prior = [0.25, 0.5, 0.25];
        let probabilities = prior.repeat(8);
        let (rate, lod) = f2_pair(&probabilities, 2, 4, 0, 1, 0.499999, 32);
        assert_eq!(rate, 0.499999);
        assert_eq!(lod, 0.0);
    }
}
