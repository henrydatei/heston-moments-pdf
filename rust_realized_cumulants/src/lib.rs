use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use chrono::{NaiveDateTime, Datelike};

/// Diese Funktion entspricht Ihrem Python‑Wrapper rCumulants.
/// Sie erwartet zwei numpy‑Arrays – eines mit den Werten (f64) und eines mit den Zeitstempeln (i64, Unix‑Zeit) – 
/// sowie die Anzahl der Monate im gleitenden Fenster (months_overlap).
#[pyfunction]
fn r_cumulants(py: Python, values: &PyArray1<f64>, timestamps: &PyArray1<i64>, months_overlap: usize) -> PyResult<Py<PyArray2<f64>>> {
    // Konvertiere die Eingaben in Rust‑Slices
    let values = unsafe { values.as_slice()? };
    let timestamps = unsafe { timestamps.as_slice()? };
    // Berechne die kumulantenbezogenen Kennzahlen für die gegebenen Zeitreihen
    let result = r_mom_np_return(values, timestamps, months_overlap)?;
    // Konvertiere das Ergebnis (Vec<[f64;4]>) in ein 2D‑numpy Array
    let result_vec: Vec<Vec<f64>> = result.iter().map(|row| row.to_vec()).collect();
    let array = PyArray2::from_vec2(py, &result_vec)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error creating array: {:?}", e)))?;
    Ok(array.into_py(py))
}

/// Gruppiert die Indizes der Beobachtungen nach Monat (basierend auf den Unix‑Zeitstempeln).
/// Liefert ein Vec von (start_index, end_index) je Gruppe.
fn group_by_month(timestamps: &[i64]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    if timestamps.is_empty() {
        return groups;
    }
    let first_dt = NaiveDateTime::from_timestamp(timestamps[0], 0);
    let mut current_year = first_dt.year();
    let mut current_month = first_dt.month();
    let mut start_index = 0;
    for (i, &ts) in timestamps.iter().enumerate() {
        let dt = NaiveDateTime::from_timestamp(ts, 0);
        if dt.year() != current_year || dt.month() != current_month {
            groups.push((start_index, i - 1));
            start_index = i;
            current_year = dt.year();
            current_month = dt.month();
        }
    }
    groups.push((start_index, timestamps.len() - 1));
    groups
}

/// Repliziert die Python‑Funktion rMomNP_return:
/// Für die übergebenen Werte und Zeitstempel werden monatliche Gruppen gebildet;
/// für jedes gleitende Fenster (über nM Monate) wird dann die Realized Estimators‑Funktion angewandt.
fn r_mom_np_return(values: &[f64], timestamps: &[i64], nM: usize) -> PyResult<Vec<[f64; 4]>> {
    let groups = group_by_month(timestamps);
    if groups.len() < nM {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Nicht genügend Monatsgruppen vorhanden."));
    }
    let mut results = Vec::new();
    for i in 0..=groups.len() - nM {
        let start_index = groups[i].0;
        let end_index = groups[i + nM - 1].1;
        if end_index + 1 > values.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Index out of bounds."));
        }
        let sub_series = &values[start_index..end_index + 1];
        let cumulants = realized_estimators_np_return(sub_series, nM)?;
        results.push(cumulants);
    }
    Ok(results)
}

/// Repliziert die Python‑Funktion RealizedEstimatorsNP_return:
/// Gegeben ist ein Slice mit Log‑Returns (oder Log‑Preisen) und nM (Anzahl der Perioden).
/// Es werden u.a. eine kumulierte Summe, elementweises Exp, Rolling Means etc. berechnet,
/// um anschließend die vier Kennzahlen k1, k2, k3, k4 zu ermitteln.
fn realized_estimators_np_return(x: &[f64], nM: usize) -> PyResult<[f64; 4]> {
    let N = x.len();
    if nM < 2 || N < nM {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Ungültiger nM oder nicht genügend Daten."));
    }
    let tau = N / nM;
    if tau < 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Fenstergröße tau < 1."));
    }
    // rt entspricht den ursprünglichen Log‑Returns
    let rt_full = x;
    let sum_foravg: f64 = rt_full.iter().sum();
    let m1r = sum_foravg / nM as f64;
    
    // Berechne kumulierte Summe (entspricht cumsum)
    let cumsum = cumulative_sum(x);
    // ex = exp(cumsum)
    let ex: Vec<f64> = cumsum.iter().map(|&v| v.exp()).collect();
    // x_inv = 1/ex
    let x_inv: Vec<f64> = ex.iter().map(|&v| 1.0 / v).collect();
    
    // Berechne gleitende Durchschnitte (Rolling Means) mit Fenstergröße tau
    let rm_full = rolling_mean(&x_inv, tau);
    let rm2_full = rolling_mean(&cumsum, tau);
    // Entsprechend des Python‑Codes nehmen wir die Werte ab Index (tau-1) bis zum vorletzten Element.
    if rm_full.len() < 2 || rm2_full.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Nicht genügend Daten für Rolling Mean."));
    }
    let rm = &rm_full[0..rm_full.len() - 1];   // Länge: N - tau
    let rm2 = &rm2_full[0..rm2_full.len() - 1];  // Länge: N - tau
    
    // Slicing: x_sliced und ex_sliced = Werte von Index (tau-1) bis N-1,
    // rt_sliced = rt_full von Index tau bis N.
    if N < tau {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Nicht genügend Daten nach Slicing."));
    }
    let x_sliced = &cumsum[tau - 1..N - 1];
    let ex_sliced = &ex[tau - 1..N - 1];
    let rt_sliced = &x[tau..N];
    
    let L = N - tau;
    if rm.len() != L || rm2.len() != L || x_sliced.len() != L || ex_sliced.len() != L || rt_sliced.len() != L {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Längen stimmen nicht überein."));
    }
    
    // Berechne y1 und y2L elementweise
    let mut y1 = Vec::with_capacity(L);
    let mut y2L = Vec::with_capacity(L);
    for i in 0..L {
        y1.push(ex_sliced[i] * rm[i] - 1.0);
        y2L.push(2.0 * ex_sliced[i] * rm[i] - 2.0 - 2.0 * x_sliced[i] + 2.0 * rm2[i]);
    }
    
    // Berechne die Summen für m2, m3, m4
    let mut sum_m2 = 0.0;
    let mut sum_m3 = 0.0;
    let mut sum_m4 = 0.0;
    let divisor = (nM - 1) as f64;
    for i in 0..L {
        let rt_val = rt_sliced[i];
        let exp_rt = rt_val.exp();
        sum_m2 += 2.0 * (exp_rt - 1.0 - rt_val);
        sum_m3 += 6.0 * ((exp_rt + 1.0) * rt_val - 2.0 * (exp_rt - 1.0))
                  + 6.0 * y1[i] * (rt_val * exp_rt - exp_rt + 1.0);
        sum_m4 += 12.0 * (rt_val * rt_val + 2.0 * (exp_rt + 2.0) * rt_val - 6.0 * (exp_rt - 1.0))
                  + 24.0 * y1[i] * ((exp_rt + 1.0) * rt_val - 2.0 * (exp_rt - 1.0))
                  + 12.0 * y2L[i] * (exp_rt - 1.0 - rt_val);
    }
    let m2r = sum_m2 / divisor;
    let m3r = sum_m3 / divisor;
    let m4r = sum_m4 / divisor;
    
    Ok([m1r, m2r, m3r, m4r])
}

/// Berechnet die kumulierte Summe eines Slices.
fn cumulative_sum(x: &[f64]) -> Vec<f64> {
    let mut cumsum = Vec::with_capacity(x.len());
    let mut sum = 0.0;
    for &v in x {
        sum += v;
        cumsum.push(sum);
    }
    cumsum
}

/// Berechnet den gleitenden Durchschnitt (Rolling Mean) eines Slices über das gegebene Fenster.
fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    if x.len() < window {
        return Vec::new();
    }
    let mut means = Vec::with_capacity(x.len() - window + 1);
    let mut window_sum: f64 = x.iter().take(window).sum();
    means.push(window_sum / window as f64);
    for i in window..x.len() {
        window_sum = window_sum - x[i - window] + x[i];
        means.push(window_sum / window as f64);
    }
    means
}

/// Das Modul, das in Python importiert werden kann.
#[pymodule]
fn realized_cumulants(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r_cumulants, m)?)?;
    Ok(())
}
