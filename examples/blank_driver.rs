use geometric_pyo3::engine::*;
use geometric_pyo3::interface::*;
use pyo3::prelude::*;

pub struct BlankDriver {}

impl GeomDriverAPI for BlankDriver {
    fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
        GradOutput { energy: 0.0, gradient: vec![0.0; coords.len()] }
    }
}

#[allow(clippy::useless_vec)]
fn main_test() -> PyResult<()> {
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

#[test]
fn test() {
    main_test().unwrap();
}

fn main() {
    main_test().unwrap();
}
