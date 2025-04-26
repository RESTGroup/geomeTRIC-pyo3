//! Main optimizer interface for geomeTRIC.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tempfile::NamedTempFile;

pub fn run_optimization(
    custom_engine: PyObject,
    params: &Py<PyDict>,
    input: Option<&str>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Import the geometric Python module
        let run_optimizer = py.import("geometric.optimize")?.getattr("run_optimizer")?;

        // kwargs for run_optimizer: make a deep copy of the params
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let kwargs = deepcopy.call1((params,))?.extract::<Bound<PyDict>>()?;

        // Create a temporary file anyway
        let tmpfile = NamedTempFile::new()?;
        let tmp_path = tmpfile.path().to_str().unwrap();

        // Only use the temporary file if input is None
        match input {
            Some(input) => kwargs.set_item("input", input)?,
            None => kwargs.set_item("input", tmp_path)?,
        }

        // Update custom_engine in kwargs
        kwargs.set_item("customengine", custom_engine)?;
        let result = run_optimizer.call((), Some(&kwargs))?;
        Ok(result.into())
    })
}
