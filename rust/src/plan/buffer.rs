use super::*;

#[pyclass(module = "msgspec_serde._native", frozen)]
pub(super) struct NativeBuffer {
    pub(super) data: Vec<u8>,
    pub(super) start: usize,
}

impl NativeBuffer {
    fn finished_bytes(&self) -> &[u8] {
        &self.data[self.start..]
    }
}

fn includes_flags(flags: c_int, required: c_int) -> bool {
    flags & required == required
}

#[pymethods]
impl NativeBuffer {
    #[getter]
    fn _allocation_size(&self) -> usize {
        self.data.len()
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("buffer view is null"));
        }
        if includes_flags(flags, ffi::PyBUF_WRITABLE) {
            return Err(PyBufferError::new_err("native FlatBuffer is read-only"));
        }

        let owner = slf.borrow();
        let bytes = owner.finished_bytes();
        let (data_ptr, length) = (bytes.as_ptr(), bytes.len());
        drop(owner);
        // SAFETY: PyO3 provides a valid writable Py_buffer pointer. `slf` is
        // transferred to `view.obj`, keeping the immutable Vec allocation alive
        // until CPython releases the exported buffer.
        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = data_ptr.cast_mut().cast::<c_void>();
            (*view).len = length as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if includes_flags(flags, ffi::PyBUF_FORMAT) {
                c"B".as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if includes_flags(flags, ffi::PyBUF_ND) {
                &raw mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if includes_flags(flags, ffi::PyBUF_STRIDES) {
                &raw mut (*view).itemsize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

pub(super) fn extract_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    let bytes = value
        .py()
        .import("builtins")?
        .getattr("bytes")?
        .call1((value,))?;
    Ok(bytes.cast::<PyBytes>()?.as_bytes().to_vec())
}

pub(super) fn nonnegative_usize(value: isize, name: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be greater than or equal to zero")))
}

pub(super) fn validate_identifier(identifier: Option<&str>) -> PyResult<()> {
    if let Some(value) = identifier
        && (value.len() != 4 || !value.is_ascii())
    {
        return Err(PyValueError::new_err(
            "FlatBuffers file identifiers must contain four ASCII bytes",
        ));
    }
    Ok(())
}

pub(super) fn buffer_byte_length(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(PyUntypedBuffer::get(value)?.len_bytes())
}

pub(super) fn checked_byte_buffer(value: &Bound<'_, PyAny>) -> PyResult<PyBuffer<u8>> {
    let buffer = PyBuffer::<u8>::get(value)?;
    check_byte_buffer(&buffer)?;
    Ok(buffer)
}

pub(super) fn check_byte_buffer(buffer: &PyBuffer<u8>) -> PyResult<()> {
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(PyTypeError::new_err(
            "byte vector fields must be one-dimensional and C-contiguous",
        ));
    }
    Ok(())
}

pub(super) fn immutable_buffer_subslice<'a>(
    buffer: &PyBuffer<u8>,
    owner: &'a [u8],
) -> PyResult<&'a [u8]> {
    let start = (buffer.buf_ptr() as usize)
        .checked_sub(owner.as_ptr() as usize)
        .ok_or_else(|| PyBufferError::new_err("buffer lies outside its immutable owner"))?;
    let end = start
        .checked_add(buffer.len_bytes())
        .filter(|&end| end <= owner.len())
        .ok_or_else(|| PyBufferError::new_err("buffer lies outside its immutable owner"))?;
    Ok(&owner[start..end])
}

#[inline]
pub(super) fn with_input_bytes<T>(
    value: &Bound<'_, PyAny>,
    decode: impl Fn(&[u8]) -> PyResult<T>,
) -> PyResult<T> {
    if let Ok(value) = value.cast::<PyBytes>() {
        return decode(value.as_bytes());
    }
    if let Ok(view) = value.cast::<PyMemoryView>() {
        let owner = view.getattr("obj")?;
        if let Ok(owner) = owner.cast::<PyBytes>() {
            let buffer = checked_byte_buffer(value)?;
            return decode(immutable_buffer_subslice(&buffer, owner.as_bytes())?);
        }
        if let Ok(owner) = owner.cast::<NativeBuffer>() {
            let buffer = checked_byte_buffer(value)?;
            let owner = owner.borrow();
            return decode(immutable_buffer_subslice(&buffer, owner.finished_bytes())?);
        }
    }
    let data = extract_bytes(value)?;
    decode(&data)
}
