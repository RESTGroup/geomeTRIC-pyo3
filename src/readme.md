# PyO3 (Rust) interface to geomeTRIC

This project contains [geomeTRIC](https://github.com/leeping/geomeTRIC) wrapper.

geomeTRIC is molecular structure geometry optimization program and library, written in Python.

Current wrapper corresponds to [v1.1](https://github.com/leeping/geomeTRIC/releases/tag/1.1).

Source code of geomeTRIC is available on [github](https://github.com/leeping/geomeTRIC).

This crate is not official bindgen project. It is originally intended to potentially serve rust electronic structure toolkit [REST](https://gitee.com/RESTGroup/rest).

| Resources | Badges |
|--|--|
| Crate | [![Crate](https://img.shields.io/crates/v/geometric-pyo3.svg)](https://crates.io/crates/geometric-pyo3) |
| API Document | [![API Documentation](https://docs.rs/geometric-pyo3/badge.svg)](https://docs.rs/geometric-pyo3) |
| Wrapper for | [![v1.1](https://img.shields.io/github/v/release/leeping/geomeTRIC)](https://github.com/leeping/geomeTRIC/releases/tag/1.1) |

## Usage of geomeTRIC-pyo3 wrapper

Currently, for a proof-of-existance working example, see [model_driver.rs](examples/model_driver.rs) for more detail. This corresponds to the geomeTRIC example of [custom engine](https://github.com/leeping/geomeTRIC/blob/master/geometric/tests/test_customengine.py) (but performs transition state instead of normal geometry optimization).

Before start, you may need some prelude:
```rust,ignore
use geometric_pyo3::prelude::*;
use pyo3::prelude::*;
```
You may also required to run this code before any PyO3 work:
```rust,ignore
pyo3::prepare_freethreaded_python();
```
otherwise, you need to enable `auto-initialize` cargo feature in PyO3.

### Step 1: Wrap your electronic structure energy/gradient

**Related APIs**:
- [`GeomDriverAPI`](crate::prelude::GeomDriverAPI)

Suppose struct `Model` (in rust side) evaluates the energy and gradient, then you may probably use the following code, to wrap and pass this reference to python side.

```rust,ignore
pub struct ModelDriver<'a> {
    model: &'a mut Model,
}
```

Then you need to implement `GeomDriverAPI` for this wrapper:
```rust,ignore
impl GeomDriverAPI for ModelDriver<'_> {
    fn calc_new(&mut self, coords: &[f64], dirname: &str) -> GradOutput {
        // calculate energy and gradient from coordinates
        // returns GradOutput { energy: ..., gradient: ... }
    }
}
```

### Step 2: Prepare molecule object

**Related APIs**:
- [`init_pyo3_molecule`](crate::prelude::init_pyo3_molecule)

Define the molecule instance. The following code gives water molecule:
```
O   0.0  0.3  0.0
H   0.9  0.8  0.0
H  -0.9  0.5  0.0
```
Please note that `xyzs` is actually a list of molecule coordinates, instead of one coordinate.
However, for most cases, you may only wish to perform `run_optimizer` to get energy minimum, and only providing one coordinate is good enough. If your task will be NEB or something else, then multiple coordinates may be useful.
```rust,ignore
let elem = ["O", "H", "H"];
let xyzs = vec![vec![0.0, 0.3, 0.0, 0.9, 0.8, 0.0, -0.9, 0.5, 0.0]];
let molecule = init_pyo3_molecule(&elem, &xyzs)?;
```

### Step 3: Prepare optimization parameters

**Related APIs**:
- [`tomlstr2py`](crate::prelude::tomlstr2py)

You can specify parameters for optimizer in toml format by string, and parsed into python recognizable dictionary by `tomlstr2py` function. If you wish to give toml value directly, then use `toml2py` function.

**NOTE**: this example is not optimization, but **finding the transition state**. To perform geometry optimization, please set `transition = false` (the default value for `transition` keyword) in the following parameters.

```rust,ignore
let optimizer_params = r#"
    transition           = true    # evaluate transition state instead of local minimum
    convergence_energy   = 1.0e-8  # Eh
    convergence_grms     = 1.0e-6  # Eh/Bohr
    convergence_gmax     = 1.0e-6  # Eh/Bohr
    convergence_drms     = 1.0e-4  # Angstrom
    convergence_dmax     = 1.0e-4  # Angstrom
"#;
let params = tomlstr2py(optimizer_params)?;
```

`input` means the file path that geomeTRIC will be logged into. It is`Option<&str>`. If give `None`, then it will logged to a temporary file, and you may not retrieve this temporary file after optimization finished.
```rust,ignore
let input = None;
```

### Step 4: Prepare engine and driver

**Related APIs**:
- [`get_pyo3_engine_cls`](crate::prelude::get_pyo3_engine_cls)
- [`PyGeomDriver`](crate::prelude::PyGeomDriver)

**Related APIs that is not intended for users**:
- [`EngineMixin`](crate::engine::EngineMixin)

`pyo3_engine_cls`: The class `PyO3Engine` at python side. It is generated dynamically. As user, you just only execute `get_pyo3_engine_cls()` to get the class.

`driver`: Wrap your model into `ModelDriver` struct to rust side, then `PyGeomDriver` to python side.

```rust,ignore
let pyo3_engine_cls = get_pyo3_engine_cls()?;
let driver = ModelDriver { model: &mut model };
let driver: PyGeomDriver = driver.into();
```

### Step 5: Actual optimization (or transition, etc.)

**Related APIs**:
- [`run_optimization`](crate::prelude::run_optimization)

The following three lines will perform the optimization.
1. Create a new instance of `PyO3Engine` class.
2. Set the driver to the engine.
3. Run the optimization.
```rust,ignore
Python::with_gil(|py| -> PyResult<()> {
    let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
    custom_engine.call_method1(py, "set_driver", (driver,))?;
    let res = run_optimization(custom_engine, &params, input)?;

    // then some post-processing code
    Ok(())
});
```

### Step 6.1: Get results from python object

You can retrieve the optimization result from `res` object. For those post-processing works, we currently do not implement such kind of post-processing codes in rust side. User may handle those post-processing by themselves.

```rust,ignore
Python::with_gil(|py| -> PyResult<()> {
    let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
    custom_engine.call_method1(py, "set_driver", (driver,))?;
    let res = run_optimization(custom_engine, &params, input)?;
    
    let coords = res
        .getattr(py, "xyzs")?
        .call_method1(py, "__getitem__", (-1,))?
        .call_method0(py, "flatten")?
        .call_method0(py, "tolist")?
        .extract::<Vec<f64>>(py)?;
    println!("Optimized Coordinates (Angstrom): {:?}", coords);
    
    let energy = res
        .getattr(py, "qm_energies")?
        .call_method1(py, "__getitem__", (-1,))?
        .extract::<f64>(py)?;
    println!("Optimized Energy (Eh): {:?}", energy);

    Ok(())
})?;
```

### Step 6.2: Get results from rust objects

You may remind that variable `model` is still in scope. If you have stored intermediate coordinates and energies in model, then you may also retrieve those value directly from your rust instance.

```rust
let model = ...;

// preparation
let driver = ModelDriver { model: &mut model };
...

// optimization

Python::with_gil(|py| -> PyResult<()> {
    let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
    custom_engine.call_method1(py, "set_driver", (driver,))?;
    let res = run_optimization(custom_engine, &params, input)?;
    Ok(())
});

// The following code is still available!
// Variable `model` has not been moved out, so it is still valid.
let coords = model.get_coords();
let energy = model.get_energy();
```
