//! Handle parameters from toml to pyo3 dictionary.
//!
//! This is mostly conveting toml to pyo3, instead of some geomopt utility.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use toml;

/// Convert `toml::Value` to `PyObject`.
///
/// This function includes a `Python` argument that must be passed in. If you
/// hope to use the default workaround, see [`toml2py_val`] instead.
pub fn toml2py_val_with_bound<'py>(
    py: Python<'py>,
    value: &toml::Value,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        toml::Value::String(s) => Ok(s.into_pyobject(py)?.into_any()),
        toml::Value::Integer(i) => Ok(i.into_pyobject(py)?.into_any()),
        toml::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any()),
        // boolean is not directly convertible to PyBool
        // this is believed to be a bug in pyo3, but will be fixed after #5054
        // https://github.com/PyO3/pyo3/issues/5051
        toml::Value::Boolean(b) => Ok(b.into_pyobject(py)?.to_owned().into_any()),
        toml::Value::Datetime(dt) => Ok(dt.to_string().into_pyobject(py)?.into_any()),
        toml::Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                let py_item = toml2py_val_with_bound(py, item)?;
                py_list.append(py_item)?;
            }
            Ok(py_list.into_any())
        },
        toml::Value::Table(table) => {
            let py_dict = PyDict::new(py);
            for (key, value) in table.iter() {
                let py_value = toml2py_val_with_bound(py, value)?;
                py_dict.set_item(key, py_value)?;
            }
            Ok(py_dict.into_any())
        },
    }
}

/// Convert `toml::Value` to `PyObject`.
pub fn toml2py_val(value: &toml::Value) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| Ok(toml2py_val_with_bound(py, value)?.unbind()))
}

/// Convert TOML value to `Py<PyDict>`.
///
/// Note that this must give PyDict, instead of any python object.
/// The returned result is also unbinded, and you may use it by
/// `dict.into_bound(py)` in a GIL guard.
pub fn toml2py(toml: &toml::Value) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| match toml {
        toml::Value::Table(table) => {
            let py_dict = PyDict::new(py);
            for (key, value) in table.iter() {
                let py_value = toml2py_val(value)?;
                py_dict.set_item(key, py_value)?;
            }
            Ok(py_dict.unbind())
        },
        _ => Err(PyValueError::new_err("TOML value must represent a table")),
    })
}

/// Convert TOML string to `Py<PyDict>`.
///
/// Note that this must give PyDict, instead of any python object.
/// The returned result is also unbinded, and you may use it by
/// `dict.into_bound(py)` in a GIL guard.
pub fn tomlstr2py(toml_str: &str) -> PyResult<Py<PyDict>> {
    let value: toml::Value = toml::de::from_str(toml_str)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse TOML string: {}", e)))?;
    toml2py(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml2py() {
        pyo3::prepare_freethreaded_python();

        let toml_str = r#"
            [package]
            name = "example"
            version = "0.1.0"
            authors = ["Alice", "Bob"]
            license = "MIT"
            description = "An example package"
            keywords = ["example", "rust", "toml"]
            homepage = "https://example.com"
            repository = ""
            [dependencies]
            pyo3 = { version = "0.15", features = ["extension-module"] }
            numpy = "1.21"
            [features]
            default = ["numpy"]
            optional = ["numpy"]
            [build]
            build = "build.rs"
        "#;
        let py_obj = tomlstr2py(toml_str).unwrap();
        Python::with_gil(|py| {
            let dict = py_obj.into_bound(py);
            println!("Converted TOML to PyObject: {:?}", dict);
        });
    }

    #[test]
    fn test_toml2py_2() {
        pyo3::prepare_freethreaded_python();

        let toml_str = r#"
        convergence_energy =   1.0e-8
        convergence_grms =     1.0e-6
        convergence_gmax =     1.0e-6
        convergence_drms =     1.0e-4
        convergence_dmax =     1.0e-4
        "#;
        let py_obj = tomlstr2py(toml_str).unwrap();
        Python::with_gil(|py| {
            let dict = py_obj.into_bound(py);
            println!("Converted TOML to PyObject: {:?}", dict);
        });
    }
}
