use pyo3::prelude::*;
use pyo3::types::PyModule;

mod plan;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    plan::register(module)
}
