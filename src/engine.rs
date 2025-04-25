use crate::interface::PyGeomDriver;
use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tempfile::NamedTempFile;

/// Mixin class to be mult-inherited together with `geometric.engine.Engine`.
#[pyclass(subclass)]
pub struct EngineMixin {
    driver: Option<PyGeomDriver>,
}

#[pymethods]
impl EngineMixin {
    /// Initialize the EngineMixin class.
    ///
    /// Though this function does not do anything, it is intended to be
    /// inherited by `geometric.engine.Engine`'s initializer. So input
    /// `_molecule` is actually gracefully initialized.
    ///
    /// Please note that `driver` is not initialized here. It should be set
    /// using the `set_driver` method manually.
    #[new]
    pub fn new(_molecule: PyObject) -> PyResult<Self> {
        Ok(EngineMixin { driver: None })
    }

    /// Set the driver for the engine.
    ///
    /// This driver is used to calculate the energy and gradient of the
    /// system. This function must be called before using the engine.
    pub fn set_driver(&mut self, driver: &PyGeomDriver) {
        self.driver = Some(driver.clone());
    }

    /// Inherits `geometric.engine.Engine`'s `calc_new` method.
    pub fn calc_new(&mut self, coords: Vec<f64>, dirname: &str) -> PyResult<PyObject> {
        // Compute the energy and gradient using the driver.
        let mut driver = self.driver.as_mut().unwrap().pointer.lock().unwrap();
        let result = driver.calc_new(&coords, dirname);
        // Convert the result to a Python object.
        // Note: that gradient must be converted to numpy flattened array (natom * 3),
        // list or 2-d array are both incorrect here.
        Python::with_gil(|py| {
            let numpy = py.import("numpy")?;
            let energy = result.energy;
            let gradient = numpy.call_method1("array", (PyList::new(py, result.gradient)?,))?;
            let dict = PyDict::new(py);
            dict.set_item("energy", energy)?;
            dict.set_item("gradient", gradient)?;
            Ok(dict.into())
        })
    }
}

/// Get the PyO3 usable geomeTRIC engine class.
pub fn get_pyo3_engine_cls() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // get the type of base class `geometric.engine.Engine`
        let base_type = py.import("geometric.engine")?.getattr("Engine")?;
        // get the type of `EngineMixin` class
        let engine_mixin_type = EngineMixin::type_object(py);

        // execute and return the following code in Python:
        // ```python
        // PyO3Engine = type('PyO3Engine', (EngineMixin, Engine), {})
        // ```
        let locals = PyDict::new(py);
        locals.set_item("Engine", base_type)?;
        locals.set_item("EngineMixin", engine_mixin_type)?;
        let pyo3_engine_type =
            py.eval(c"type('PyO3Engine', (EngineMixin, Engine), {})", None, Some(&locals))?;
        Ok(pyo3_engine_type.into())
    })
}

/// Initialize a geomeTRIC molecule into Python object.
///
/// # Arguments
///
/// - `elem`: A slice of strings representing the element types.
/// - `xyzs`: A list of vectors representing the coordinates of the atoms. Each
///   vector represents one molecule, where its length is (natom * 3), with
///   dimension of coordinate (3) to be contiguous.
pub fn init_pyo3_molecule(elem: &[&str], xyzs: &[Vec<f64>]) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Import the geometric Python module.
        let molecule_cls = py.import("geometric.molecule")?.getattr("Molecule")?;

        // Create a new instance of the Molecule class
        let molecule_instance = molecule_cls.call0()?;

        // xyzs must be converted into numpy array of shape (natom, 3), where 1-D array
        // or python list are both incorrect.
        let numpy = py.import("numpy")?;
        let xyzs = xyzs
            .iter()
            .map(|xyz| {
                let arr = numpy.call_method1("array", (PyList::new(py, xyz)?,))?;
                let arr = arr.call_method1("reshape", (-1, 3))?;
                Ok(arr)
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Set the attributes
        molecule_instance.setattr("elem", elem)?;
        molecule_instance.setattr("xyzs", xyzs)?;
        Ok(molecule_instance.into())
    })
}

pub fn run_optimization(custom_engine: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Import the geometric Python module
        let run_optimizer = py.import("geometric.optimize")?.getattr("run_optimizer")?;

        // Create a temporary file
        let tmpfile = NamedTempFile::new()?;
        let tmp_path = tmpfile.path().to_str().unwrap();

        // Call run_optimizer with the specified parameters
        let kwargs = PyDict::new(py);
        kwargs.set_item("customengine", custom_engine)?;
        kwargs.set_item("check", 1)?;
        kwargs.set_item("input", tmp_path)?;

        let result = run_optimizer.call((), Some(&kwargs))?;

        Ok(result.into())
    })
}
