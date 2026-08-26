use pyo3::prelude::*;
use pyo3::types::PyModule;

mod plan;

#[pymodule(gil_used = false)]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    plan::register(module)
}
