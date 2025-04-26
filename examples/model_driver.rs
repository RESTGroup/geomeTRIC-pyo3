//! Test case of custom engine:
//! https://github.com/leeping/geomeTRIC/blob/73912c91708304c2413548f325d0753d70eb3ae5/geometric/tests/test_customengine.py

use geometric_pyo3::engine::*;
use geometric_pyo3::interface::*;
use geometric_pyo3::optimize::*;
use geometric_pyo3::util::*;
use pyo3::prelude::*;

pub struct Model {
    b: [[f64; 3]; 3],
    w: [[f64; 3]; 3],
    coords: Option<Vec<f64>>,
}

#[allow(clippy::new_without_default)]
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

pub struct ModelDriver<'a> {
    model: &'a mut Model,
}

impl GeomDriverAPI for ModelDriver<'_> {
    fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
        self.model.calc_eng_grad(coords)
    }
}

fn main_test() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    let elem = ["O", "H", "H"];
    let xyzs = vec![vec![0.0, 0.3, 0.0, 0.9, 0.8, 0.0, -0.9, 0.5, 0.0]];
    let molecule = init_pyo3_molecule(&elem, &xyzs)?;
    println!("Molecule: {:?}", molecule);

    let optimizer_params = r#"
    transition           = true    # evaluate transition state instead of local minimum
    convergence_energy   = 1.0e-8  # Eh
    convergence_grms     = 1.0e-6  # Eh/Bohr
    convergence_gmax     = 1.0e-6  # Eh/Bohr
    convergence_drms     = 1.0e-4  # Angstrom
    convergence_dmax     = 1.0e-4  # Angstrom
    "#;
    let params = tomlstr2py(optimizer_params)?;
    let input = None;
    let pyo3_engine_cls = get_pyo3_engine_cls()?;

    Python::with_gil(|py| -> PyResult<()> {
        let mut model = Model::new();
        let driver = ModelDriver { model: &mut model };
        let driver: PyGeomDriver = driver.into();
        let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
        custom_engine.call_method1(py, "set_driver", (driver.clone(),))?;
        println!("Custom Engine: {:?}", custom_engine);
        let res = run_optimization(custom_engine, &params, input)?;
        println!("Optimization Result: {:?}", res);

        // retrive coordinates from the result
        let coords = res
            .getattr(py, "xyzs")?
            .call_method1(py, "__getitem__", (-1,))?
            .call_method0(py, "flatten")?
            .call_method0(py, "tolist")?
            .extract::<Vec<f64>>(py)?;
        println!("Optimized Coordinates (Angstrom): {:?}", coords);

        // retrive coordinates from original model
        // this is retrivable since we uses `&mut model` in evaluation
        let coords = model.get_coords().unwrap();
        println!("Model Coordinates (Bohr): {:?}", coords);

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
