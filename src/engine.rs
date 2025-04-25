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
mod test_blank_driver {
    use super::*;
    use crate::interface::*;
    pub struct BlankDriver {}

    impl GeomDriverAPI for BlankDriver {
        fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
            GradOutput { energy: 0.0, gradient: vec![0.0; coords.len()] }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

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
            let driver = BlankDriver {};
            let driver: PyGeomDriver = driver.into();
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            custom_engine.call_method1(py, "set_driver", (driver,))?;
            println!("Custom Engine: {:?}", custom_engine);

            let res = run_optimization(custom_engine)?;
            println!("Optimization Result: {:?}", res);
            Ok(())
        })?;

        Ok(())
    }
}

/// Test case of custom engine:
/// https://github.com/leeping/geomeTRIC/blob/73912c91708304c2413548f325d0753d70eb3ae5/geometric/tests/test_customengine.py
#[cfg(test)]
mod test_model_driver {

    use super::*;
    use crate::interface::*;
    pub struct Model {
        b: [[f64; 3]; 3],
        w: [[f64; 3]; 3],
        coords: Option<Vec<f64>>,
    }

    impl Model {
        pub fn new() -> Self {
            let b = [[0.0, 1.8, 1.8], [1.8, 0.0, 2.8], [1.8, 2.8, 0.0]];
            let w = [[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]];
            Model { b, w, coords: None }
        }

        pub fn get_coords(&self) -> Option<Vec<f64>> {
            self.coords.clone()
        }

        pub fn calc_eng_grad(&mut self, coords: &[f64]) -> GradOutput {
            self.coords = Some(coords.to_vec());
            let coords: Vec<&[f64]> = coords.chunks(3).collect();
            let natm = coords.len();
            let b = self.b;
            let w = self.w;
            // dr = coords[:,None,:] - coords
            let mut dr = vec![vec![vec![0.0; 3]; natm]; natm];
            for i in 0..natm {
                for j in 0..natm {
                    dr[i][j][0] = coords[i][0] - coords[j][0];
                    dr[i][j][1] = coords[i][1] - coords[j][1];
                    dr[i][j][2] = coords[i][2] - coords[j][2];
                }
            }
            // dist = np.linalg.norm(dr, axis=2)
            let mut dist = vec![vec![0.0; natm]; natm];
            for i in 0..natm {
                for j in 0..natm {
                    dist[i][j] = (dr[i][j][0] * dr[i][j][0]
                        + dr[i][j][1] * dr[i][j][1]
                        + dr[i][j][2] * dr[i][j][2])
                        .sqrt();
                }
            }
            // e = (w * (dist - b)**2).sum()
            let mut energy = 0.0;
            for i in 0..natm {
                for j in 0..natm {
                    energy += w[i][j] * (dist[i][j] - b[i][j]).powi(2);
                }
            }
            // grad = np.einsum('ij,ijx->ix', 2*w*(dist-b)/(dist+1e-60), dr)
            // grad-= np.einsum('ij,ijx->jx', 2*w*(dist-b)/(dist+1e-60), dr)
            let mut tmp = vec![vec![0.0; natm]; natm];
            for i in 0..natm {
                for j in 0..natm {
                    tmp[i][j] = 2.0 * w[i][j] * (dist[i][j] - b[i][j]) / (dist[i][j] + 1e-60);
                }
            }

            let mut grad = vec![vec![0.0; 3]; natm];
            for i in 0..natm {
                for j in 0..natm {
                    grad[i][0] += tmp[i][j] * dr[i][j][0];
                    grad[i][1] += tmp[i][j] * dr[i][j][1];
                    grad[i][2] += tmp[i][j] * dr[i][j][2];
                    grad[j][0] -= tmp[i][j] * dr[i][j][0];
                    grad[j][1] -= tmp[i][j] * dr[i][j][1];
                    grad[j][2] -= tmp[i][j] * dr[i][j][2];
                }
            }

            // grad = grad.flatten()
            let gradient = grad.iter().flat_map(|x| x.iter()).copied().collect();

            GradOutput { energy, gradient }
        }
    }

    impl GeomDriverAPI for Model {
        fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
            self.calc_eng_grad(coords)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

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
            let model = Model::new();
            let driver: PyGeomDriver = model.into();
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            custom_engine.call_method1(py, "set_driver", (driver.clone(),))?;
            println!("Custom Engine: {:?}", custom_engine);

            let res = run_optimization(custom_engine)?;
            println!("Optimization Result: {:?}", res);

            // retrive coordinates
            let coords = res
                .getattr(py, "xyzs")?
                .call_method1(py, "__getitem__", (-1,))?
                .call_method0(py, "flatten")?
                .call_method0(py, "tolist")?
                .extract::<Vec<f64>>(py)?;
            println!("Optimized Coordinates: {:?}", coords);

            Ok(())
        })?;

        Ok(())
    }
}
