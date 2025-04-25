//! Interface that electronic structure codes should implement.

use std::mem::transmute;
use std::sync::{Arc, Mutex};

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
}

/// Python wrapper for the `GeomDriverAPI` trait implementations.
///
/// `GeomDriverAPI` is defined as rust trait, which is not directly usable in
/// Python. This makes the glue between the rust trait and the python class.
///
/// # Safety
///
/// This struct is marked as `unsafe` because it uses `transmute` to convert
/// local lifetime to static lifetime.
///
/// If your class (to be implemented with trait `GeomDriverAPI`) have lifetime
/// parameters, then this lifetime will be transmuted to static lifetime when it
/// is converted to python object. As long as you don't disturb the lifetime of
/// reference, this transmute should be safe.
#[pyclass]
#[derive(Clone)]
pub struct PyGeomDriver {
    pub pointer: Arc<Mutex<dyn GeomDriverAPI>>,
}

impl<T> From<T> for PyGeomDriver
where
    T: GeomDriverAPI,
{
    fn from(driver: T) -> Self {
        let a: Arc<Mutex<dyn GeomDriverAPI>> = Arc::new(Mutex::new(driver));
        // Safety not checked, and should be provided by the caller.
        // This will convert local lifetime (of `T`) to static lifetime (`'static`) for
        // python calls.
        unsafe { transmute(a) }
    }
}
