//! Interface that electronic structure codes should implement.

use std::{
    mem::transmute,
    sync::{Arc, Mutex},
};

use pyo3::prelude::*;

/// Gradient output from the electronic structure code.
///
/// - `energy`: The energy of the system, scalar.
/// - `gradient`: The gradient of the system, flattened (natom * 3), with
///   dimension of coordinate (3) to be contiguous.
pub struct GradOutput {
    pub energy: f64,
    pub gradient: Vec<f64>,
}

/// Trait API to be implemented in electronic structure code for geomeTRIC PyO3
/// binding.
pub trait GeomDriverAPI: Send {
    /// Calculate the energy and gradient of the system.
    ///
    /// This trait corresponds to the `calc_new` method in the `Engine` class in
    /// geomeTRIC.
    ///
    /// # Arguments
    ///
    /// - `coords` - The coordinates of the system, flattened (natom * 3), with
    ///   dimension of coordinate (3) to be contiguous.
    /// - `dirname` - The directory to run the calculation in. Can be set to
    ///   dummy if directory is not required for gradient computation.
    ///
    /// # Returns
    ///
    /// A `GradOutput` struct containing the energy and gradient of the system.
    fn calc_new(&mut self, coords: &[f64], dirname: &str) -> GradOutput;

    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Python wrapper for the `GeomDriverAPI` trait implementations.
///
/// `GeomDriverAPI` is defined as rust trait, which is not directly usable in
/// Python. This makes the glue between the rust trait and the python class.
#[pyclass]
#[derive(Clone)]
pub struct PyGeomDriver {
    pub pointer: Arc<Mutex<Box<dyn GeomDriverAPI>>>,
}

impl<T> From<T> for PyGeomDriver
where
    T: GeomDriverAPI + 'static,
{
    fn from(driver: T) -> Self {
        PyGeomDriver { pointer: Arc::new(Mutex::new(Box::new(driver))) }
    }
}
