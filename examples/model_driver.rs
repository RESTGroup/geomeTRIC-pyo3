//! Test case of custom engine:
//!
//! https://github.com/leeping/geomeTRIC/blob/73912c91708304c2413548f325d0753d70eb3ae5/geometric/tests/test_customengine.py
//!
//! Please note this test case performs transition state, instead of usual
//! geometry optimization.

#![allow(clippy::uninlined_format_args)]

use geometric_pyo3::prelude::*;
use pyo3::prelude::*;

/// A simple model for testing geometric optimization.
///
/// In real applications, this struct is something electronic structure program
/// have already implemented, and not related to geomeTRIC.
///
/// This can be some SCF/MP2/CC gradient code; gives it the coordinates, it
/// gives out energy and gradient.
///
/// This model will calculate energy and gradient from `calc_eng_grad` function.
///
/// The last coordinates and energy will be stored in `current_coords` and
/// `current_energy` fields. In another word, this model will mutate itself
/// after each calculation in geometry optimization.
///
/// For this example, the computation model is implemented in later position in
/// this file.
pub struct Model {
    b: [[f64; 3]; 3],
    w: [[f64; 3]; 3],
    /// This field will change after geometry optimization steps.
    pub current_energy: Option<f64>,
    /// This field will change after geometry optimization steps.
    pub current_coords: Option<Vec<f64>>,
}

/// A driver for passing the model to geomeTRIC.
///
/// It is better to use `&mut Model` instead of `Model` in the driver. In this
/// way, you can still retrieve the original `Model` instance after the
/// optimization.
///
/// You can add other fields you would like, but note that after passing this
/// object to geomeTRIC driver, this object itself cannot be easily retrieved
/// anymore. So this object is better to be a simple wrapper of the reference
/// (either mutable or immutable) for the model.
pub struct ModelDriver<'a> {
    model: &'a mut Model,
}

/// Implementation of `GeomDriverAPI` trait for the `ModelDriver`.
///
/// You need to implement this trait for your driver.
impl GeomDriverAPI for ModelDriver<'_> {
    fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
        self.model.calc_eng_grad(coords)
    }
}

fn main_test() -> PyResult<()> {
    // `model` is the instance of `Model` struct, which is some gradient code of
    // electronic structure program. This instance can be initialized anywhere you
    // like.
    let mut model = Model::new();

    // If it is not your desired , enable `feature=["auto-initialize"]` for pyo3 in
    // Cargo.toml.
    pyo3::prepare_freethreaded_python();

    // Define the molecule instance.
    // The following code gives water molecule:
    // O   0.0  0.3  0.0
    // H   0.9  0.8  0.0
    // H  -0.9  0.5  0.0
    // Please note that `xyzs` is actually a list of molecule coordinates, instead
    // of one coordinate.
    // However, for most cases, you may only wish to perform `run_optimizer` to get
    // energy minimum, and only providing one coordinate is good enough. If your
    // task will be NEB or something else, then multiple coordinates may be
    // useful.
    let elem = ["O", "H", "H"];
    let xyzs = vec![vec![0.0, 0.3, 0.0, 0.9, 0.8, 0.0, -0.9, 0.5, 0.0]];
    let molecule = init_pyo3_molecule(&elem, &xyzs)?;
    println!("Molecule: {:?}", molecule);

    // You can specify parameters for optimizer in toml format by string, and parsed
    // into python recognizable dictionary by `tomlstr2py` function.
    // If you wish to give toml value directly, then use `toml2py` function.
    //
    // **NOTE**: this example is not optimization, but finding the transition state.
    // To perform geometry optimization, please set `transition = false` (the
    // default value for `transition` keyword) in the following parameters.
    let optimizer_params = r#"
        transition           = true    # evaluate transition state instead of local minimum
        convergence_energy   = 1.0e-8  # Eh
        convergence_grms     = 1.0e-6  # Eh/Bohr
        convergence_gmax     = 1.0e-6  # Eh/Bohr
        convergence_drms     = 1.0e-4  # Angstrom
        convergence_dmax     = 1.0e-4  # Angstrom
    "#;
    let params = tomlstr2py(optimizer_params)?;
    // `input` means the file path that geomeTRIC will be logged into. It is
    // `Option<&str>`. If give `None`, then it will logged to a temporary file, and
    // you may not retrieve this temporary file after optimization finished.
    let input = None;

    // `pyo3_engine_cls` is the class `PyO3Engine` at python side.
    // As user, you just only execute `get_pyo3_engine_cls()` to get the class.
    // That's all.
    let pyo3_engine_cls = get_pyo3_engine_cls()?;
    // Then you need to wrap your model into `ModelDriver` struct in rust side.
    // As previously mentioned, it is better to use reference `&model` or mutable
    // reference `&mut model` here. After optimization, you may still retrieve the
    // `model` object.
    let driver = ModelDriver { model: &mut model };
    // The following step is also required. It will convert `ModelDriver` to python
    // side.
    let driver: PyGeomDriver = driver.into();

    Python::with_gil(|py| -> PyResult<()> {
        // The following three lines will perform the optimization.
        // 1. Create a new instance of `PyO3Engine` class.
        // 2. Set the driver to the engine.
        // 3. Run the optimization.
        let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
        custom_engine.call_method1(py, "set_driver", (driver,))?;
        let res = run_optimization(custom_engine, &params, input)?;

        // You can retrieve the optimization result from `res` object.
        // This is the same to python code
        // `list(res.xyzs[-1].flatten())`.
        // The returned coordinates are in Angstrom.
        //
        // `res.xyzs` is actually a list of coordinates, showing the trajectory of
        // optimization.
        //
        // We currently do not implement such kind of post-processing codes in rust
        // side. User may handle those post-processing by themselves.
        let coords = res
            .getattr(py, "xyzs")?
            .call_method1(py, "__getitem__", (-1,))?
            .call_method0(py, "flatten")?
            .call_method0(py, "tolist")?
            .extract::<Vec<f64>>(py)?;
        println!("Optimized Coordinates (Angstrom): {:?}", coords);

        // You can also retrieve the energy from `res` object.
        // This is the same to python code
        // `res.qm_energies[-1]`.
        // The returned energy is in Hartree.
        //
        // `res.qm_energies` also shows the trajectory of optimization.
        // The last energy is the optimized energy.
        let energy = res
            .getattr(py, "qm_energies")?
            .call_method1(py, "__getitem__", (-1,))?
            .extract::<f64>(py)?;
        println!("Optimized Energy (Eh): {:?}", energy);

        // For this specific case, energy should be close to 0.32 Eh for transition
        // state.
        assert!((energy - 0.32).abs() < 1.0e-8);

        Ok(())
    })?;

    // You can also retrieve the optimized coordinates and energy from the original
    // model, since in `Model::calc_eng_grad` function the field `current_coords`
    // will be overwritten when evaluated.
    let coords = model.current_coords.as_ref().unwrap();
    println!("Model Coordinates (Bohr): {:?}", coords);

    // Same thing to energy.
    let energy = model.current_energy.as_ref().unwrap();
    println!("Model Energy (Eh): {:?}", energy);

    Ok(())
}

#[allow(clippy::new_without_default)]
impl Model {
    pub fn new() -> Self {
        let b = [[0.0, 1.8, 1.8], [1.8, 0.0, 2.8], [1.8, 2.8, 0.0]];
        let w = [[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]];
        Model { b, w, current_coords: None, current_energy: None }
    }

    pub fn calc_eng_grad(&mut self, coords: &[f64]) -> GradOutput {
        self.current_coords = Some(coords.to_vec());

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

        self.current_energy = Some(energy);

        GradOutput { energy, gradient }
    }
}

#[test]
fn test() {
    main_test().unwrap();
}

fn main() {
    main_test().unwrap();
}
