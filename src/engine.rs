use pyo3::types::{PyDict, PyList};
use pyo3::{PyTypeInfo, prelude::*};
use tempfile::NamedTempFile;

pub struct GradOutput {
    pub energy: f64,
    pub gradient: Vec<f64>,
}

fn model(coords: Vec<f64>, _dirname: String) -> Result<GradOutput, ()> {
    // This is a dummy implementation of the model function
    let output = GradOutput { energy: 0.0, gradient: vec![0.0; coords.len()] };
    Ok(output)
}

#[pyclass(subclass)]
pub struct EngineMixin;

#[pymethods]
impl EngineMixin {
    #[new]
    pub fn new(_molecule: PyObject) -> Self {
        EngineMixin {}
    }

    fn calc_new(slf: &Bound<Self>, coords: Vec<f64>, dirname: String) -> PyResult<PyObject> {
        // Call the model function
        println!("coords {:?}", coords);
        let result = model(coords, dirname).unwrap();
        // Convert the result to a Python object
        let py = slf.py();
        let energy = result.energy;
        let gradient = result.gradient;
        let numpy = py.import("numpy")?;
        let gradient = numpy.call_method1("array", (PyList::new(py, &gradient)?,))?;
        let dict = PyDict::new(py);
        dict.set_item("energy", energy)?;
        dict.set_item("gradient", gradient)?;
        Ok(dict.into())
    }
}

pub fn get_pyo3_engine_cls() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let base_type = py.import("geometric.engine")?.getattr("Engine")?;
        let engine_mixin_type = EngineMixin::type_object(py);

        let locals = PyDict::new(py);
        locals.set_item("Engine", base_type)?;
        locals.set_item("EngineMixin", engine_mixin_type)?;
        let pyo3_engine_type = py.eval(
            std::ffi::CString::new("type('PyO3Engine', (EngineMixin, Engine), {})")
                .unwrap()
                .as_c_str(),
            None,
            Some(&locals),
        )?;

        Ok(pyo3_engine_type.into())
    })
}

pub fn init_pyo3_molecule(elem: &[&str], xyzs: &[Vec<f64>]) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Import the geometric Python module
        let geometric = py.import("geometric")?;
        let molecule = geometric.getattr("molecule")?.getattr("Molecule")?;

        // Create a new instance of the Molecule class
        let molecule_instance = molecule.call0()?;

        // xyzs must be converted into numpy array
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

fn run_optimization(custom_engine: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Import the geometric Python module
        let geometric = py.import("geometric")?;
        let optimize = geometric.getattr("optimize")?;
        let run_optimizer = optimize.getattr("run_optimizer")?;

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[allow(clippy::useless_vec)]
    fn test_gen_molecule() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();

        let elem = ["O", "H", "H"];
        let xyzs = vec![vec![0.0, 0.3, 0.0, 0.9, 0.8, 0.0, -0.9, 0.5, 0.0]];
        let molecule = init_pyo3_molecule(&elem, &xyzs)?;
        println!("Molecule: {:?}", molecule);

        let pyo3_engine_cls = get_pyo3_engine_cls()?;
        Python::with_gil(|py| -> PyResult<()> {
            // let pyo3_engine_mod = PyModule::new(py, "pyo3_engine_module")?;
            // pyo3_engine_module(&pyo3_engine_mod)?;
            // let pyo3_engine_cls = pyo3_engine_mod.getattr("PyO3Engine")?;
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            println!("Custom Engine: {:?}", custom_engine);

            let res = run_optimization(custom_engine)?;
            println!("Optimization Result: {:?}", res);
            Ok(())
        })?;

        Ok(())
    }
}
