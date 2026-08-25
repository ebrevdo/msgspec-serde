use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::mem::size_of;
use std::ptr;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable, cast_slice};
use flatbuffers::{
    FLATBUFFERS_MAX_BUFFER_SIZE, FlatBufferBuilder, Push, PushAlignment, TableFinishedWIPOffset,
    UnionWIPOffset, VOffsetT, WIPOffset,
};
use pyo3::buffer::{Element, PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::{
    PyBufferError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyMemoryView, PyModule, PyString, PyTuple, PyType};
use rmpv::Value;
use serde::Deserialize;

#[derive(Deserialize)]
struct ModuleWire {
    version: u8,
    objects: Vec<ObjectWire>,
}

#[derive(Deserialize)]
struct ObjectWire {
    name: String,
    is_struct: bool,
    byte_size: usize,
    min_alignment: usize,
    fields: Vec<FieldWire>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum FieldKind {
    Scalar,
    String,
    Table,
    Struct,
    VectorByte,
    VectorScalar,
    VectorString,
    VectorTable,
    VectorStruct,
    Nested,
    Dynamic,
    Union,
    UnionVector,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ScalarKind {
    Bool,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float32,
    Float64,
}

#[derive(Clone, Copy)]
enum ScalarValue {
    Bool(bool),
    Int8(i8),
    Uint8(u8),
    Int16(i16),
    Uint16(u16),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Uint64(u64),
    Float32(f32),
    Float64(f64),
}

#[derive(Deserialize)]
struct FieldWire {
    name: String,
    slot: u16,
    offset: usize,
    kind: FieldKind,
    #[serde(default)]
    scalar: Option<ScalarKind>,
    default: Value,
    optional: bool,
    required: bool,
    #[serde(default)]
    target: Option<String>,
    #[serde(skip)]
    target_index: Option<usize>,
    #[serde(default)]
    type_slot: Option<u16>,
    #[serde(default)]
    type_scalar: Option<ScalarKind>,
    #[serde(default)]
    allowed_prefix: Option<String>,
    #[serde(default)]
    arms: Vec<ArmWire>,
}

#[derive(Deserialize)]
struct ArmWire {
    tag: u64,
    target: String,
    #[serde(skip)]
    target_index: usize,
}

type AnyOffset = WIPOffset<UnionWIPOffset>;

#[derive(Clone, Copy)]
enum Discriminator {
    Union(u64),
    Offset(AnyOffset),
}

struct FieldState<'py> {
    value: Bound<'py, PyAny>,
    offset: Option<AnyOffset>,
    discriminator: Option<Discriminator>,
    prepared: Option<PreparedValue<'py>>,
    scalar: Option<ScalarValue>,
}

enum PreparedValue<'py> {
    Nested(Bound<'py, PyAny>),
    Dynamic {
        tag: String,
        data: Bound<'py, PyAny>,
    },
}

#[pyclass(module = "msgspec_flatbuffers._native", frozen)]
struct NativeBuffer {
    data: Vec<u8>,
    start: usize,
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
        if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("native FlatBuffer is read-only"));
        }

        let data = slf.borrow();
        let bytes = &data.data[data.start..];
        let (data_ptr, length) = (bytes.as_ptr(), bytes.len());
        drop(data);
        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = data_ptr.cast_mut().cast::<c_void>();
            (*view).len = length as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
                c"B".as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
                &raw mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
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

fn erase_offset<T>(offset: WIPOffset<T>) -> AnyOffset {
    WIPOffset::new(offset.value())
}

#[inline]
fn sequence_len(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(value) = value.cast::<PyList>() {
        Ok(value.len())
    } else {
        value.len()
    }
}

#[inline]
fn for_each_sequence<'py>(
    value: &Bound<'py, PyAny>,
    mut visit: impl FnMut(Bound<'py, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    if let Ok(value) = value.cast::<PyList>() {
        for item in value.iter() {
            visit(item)?;
        }
    } else {
        for item in value.try_iter()? {
            visit(item?)?;
        }
    }
    Ok(())
}

#[inline]
fn sequence_item<'py>(value: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    value.get_item(index)
}

#[inline]
fn scalar_size(scalar: Option<ScalarKind>) -> PyResult<usize> {
    match scalar {
        Some(ScalarKind::Bool | ScalarKind::Int8 | ScalarKind::Uint8) => Ok(1),
        Some(ScalarKind::Int16 | ScalarKind::Uint16) => Ok(2),
        Some(ScalarKind::Int32 | ScalarKind::Uint32 | ScalarKind::Float32) => Ok(4),
        Some(ScalarKind::Int64 | ScalarKind::Uint64 | ScalarKind::Float64) => Ok(8),
        _ => Err(PyValueError::new_err(
            "native field has no valid scalar kind",
        )),
    }
}

#[inline]
fn aligned_payload_size(payload: usize, alignment: usize) -> usize {
    payload.saturating_add(alignment.saturating_sub(1))
}

fn sampled_sequence_size(
    value: &Bound<'_, PyAny>,
    mut item_size: impl FnMut(Bound<'_, PyAny>) -> PyResult<usize>,
) -> PyResult<usize> {
    let length = sequence_len(value)?;
    if length == 0 {
        return Ok(0);
    }
    let mut sampled = 0usize;
    let sample_count = length.min(6);
    if length <= 6 {
        for index in 0..length {
            sampled = sampled.saturating_add(item_size(sequence_item(value, index)?)?);
        }
    } else {
        let middle = length / 2;
        for index in [0, 1, middle - 1, middle, length - 2, length - 1] {
            sampled = sampled.saturating_add(item_size(sequence_item(value, index)?)?);
        }
    }
    Ok(sampled
        .saturating_mul(length)
        .saturating_add(sample_count - 1)
        / sample_count)
}

fn default_bool(value: &Value) -> PyResult<bool> {
    value
        .as_bool()
        .ok_or_else(|| PyTypeError::new_err("invalid native boolean default"))
}

fn default_i64(value: &Value) -> PyResult<i64> {
    value
        .as_i64()
        .ok_or_else(|| PyTypeError::new_err("invalid native signed default"))
}

fn default_u64(value: &Value) -> PyResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| PyTypeError::new_err("invalid native unsigned default"))
}

fn default_f64(value: &Value) -> PyResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| PyTypeError::new_err("invalid native floating-point default"))
}

impl ScalarValue {
    fn extract(kind: ScalarKind, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match kind {
            ScalarKind::Bool => Self::Bool(value.extract()?),
            ScalarKind::Int8 => Self::Int8(value.extract()?),
            ScalarKind::Uint8 => Self::Uint8(value.extract()?),
            ScalarKind::Int16 => Self::Int16(value.extract()?),
            ScalarKind::Uint16 => Self::Uint16(value.extract()?),
            ScalarKind::Int32 => Self::Int32(value.extract()?),
            ScalarKind::Uint32 => Self::Uint32(value.extract()?),
            ScalarKind::Int64 => Self::Int64(value.extract()?),
            ScalarKind::Uint64 => Self::Uint64(value.extract()?),
            ScalarKind::Float32 => Self::Float32(value.extract()?),
            ScalarKind::Float64 => Self::Float64(value.extract()?),
        })
    }

    fn is_default(self, default: &Value) -> PyResult<bool> {
        Ok(match self {
            Self::Bool(value) => value == default_bool(default)?,
            Self::Int8(value) => value == default_i64(default)? as i8,
            Self::Uint8(value) => value == default_u64(default)? as u8,
            Self::Int16(value) => value == default_i64(default)? as i16,
            Self::Uint16(value) => value == default_u64(default)? as u16,
            Self::Int32(value) => value == default_i64(default)? as i32,
            Self::Uint32(value) => value == default_u64(default)? as u32,
            Self::Int64(value) => value == default_i64(default)?,
            Self::Uint64(value) => value == default_u64(default)?,
            Self::Float32(value) => value == default_f64(default)? as f32,
            Self::Float64(value) => value == default_f64(default)?,
        })
    }

    fn push(
        self,
        builder: &mut FlatBufferBuilder<'_>,
        slot: VOffsetT,
        default: &Value,
        optional: bool,
    ) -> PyResult<()> {
        macro_rules! push {
            ($value:expr, $default:expr) => {
                if optional {
                    builder.push_slot_always(slot, $value);
                } else {
                    builder.push_slot(slot, $value, $default);
                }
            };
        }
        match self {
            Self::Bool(value) => push!(value, default_bool(default)?),
            Self::Int8(value) => push!(value, default_i64(default)? as i8),
            Self::Uint8(value) => push!(value, default_u64(default)? as u8),
            Self::Int16(value) => push!(value, default_i64(default)? as i16),
            Self::Uint16(value) => push!(value, default_u64(default)? as u16),
            Self::Int32(value) => push!(value, default_i64(default)? as i32),
            Self::Uint32(value) => push!(value, default_u64(default)? as u32),
            Self::Int64(value) => push!(value, default_i64(default)?),
            Self::Uint64(value) => push!(value, default_u64(default)?),
            Self::Float32(value) => push!(value, default_f64(default)? as f32),
            Self::Float64(value) => push!(value, default_f64(default)?),
        }
        Ok(())
    }
}

fn push_scalar(
    builder: &mut FlatBufferBuilder<'_>,
    slot: VOffsetT,
    scalar: ScalarKind,
    value: &Bound<'_, PyAny>,
    default: &Value,
    optional: bool,
) -> PyResult<()> {
    ScalarValue::extract(scalar, value)?.push(builder, slot, default, optional)
}

fn push_type_id(
    builder: &mut FlatBufferBuilder<'_>,
    slot: VOffsetT,
    scalar: Option<ScalarKind>,
    value: u64,
) -> PyResult<()> {
    match scalar {
        Some(ScalarKind::Uint8) => builder.push_slot(slot, value as u8, 0),
        Some(ScalarKind::Uint16) => builder.push_slot(slot, value as u16, 0),
        Some(ScalarKind::Uint32) => builder.push_slot(slot, value as u32, 0),
        Some(ScalarKind::Uint64) => builder.push_slot(slot, value, 0),
        _ => {
            return Err(PyValueError::new_err(
                "polymorphic field has an invalid type scalar",
            ));
        }
    }
    Ok(())
}

macro_rules! create_scalar_vector {
    ($builder:expr, $value:expr, $ty:ty, $scalar:expr, $patches:expr) => {{
        if let Ok(raw) = PyUntypedBuffer::get($value) {
            if raw.dimensions() != 1 {
                return Err(PyTypeError::new_err(
                    "numeric vector fields must be one-dimensional",
                ));
            }
            if !buffer_matches_native_scalar(&raw, $scalar) {
                return Err(PyTypeError::new_err(format!(
                    "numeric vector field requires native {:?} data",
                    $scalar,
                )));
            }
            let values = raw.as_typed::<$ty>()?.to_vec($value.py())?;
            build_owned_raw_vector($builder, values, $patches)?
        } else {
            let values = $value.extract::<Vec<$ty>>()?;
            build_owned_raw_vector($builder, values, $patches)?
        }
    }};
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BufferEndian {
    Native,
    Little,
    Big,
}

fn buffer_scalar_format(buffer: &PyUntypedBuffer, scalar: ScalarKind) -> Option<BufferEndian> {
    scalar_format(buffer.format().to_bytes(), buffer.item_size(), scalar)
}

fn scalar_format(format: &[u8], item_size: usize, scalar: ScalarKind) -> Option<BufferEndian> {
    let (endian, type_code) = match format {
        [code] | [b'@', code] | [b'=', code] => (BufferEndian::Native, *code),
        [b'<', code] => (BufferEndian::Little, *code),
        [b'>' | b'!', code] => (BufferEndian::Big, *code),
        _ => return None,
    };
    let expected = match scalar {
        ScalarKind::Bool => b'?',
        ScalarKind::Int8 => b'b',
        ScalarKind::Uint8 => b'B',
        ScalarKind::Int16 => b'h',
        ScalarKind::Uint16 => b'H',
        ScalarKind::Int32 => b'i',
        ScalarKind::Uint32 => b'I',
        ScalarKind::Int64 => b'q',
        ScalarKind::Uint64 => b'Q',
        ScalarKind::Float32 => b'f',
        ScalarKind::Float64 => b'd',
    };
    let matches_type = type_code == expected
        || matches!(
            (scalar, type_code, item_size),
            (ScalarKind::Int32, b'l', 4)
                | (ScalarKind::Uint32, b'L', 4)
                | (ScalarKind::Int64, b'l', 8)
                | (ScalarKind::Uint64, b'L', 8)
        );
    matches_type.then_some(endian)
}

fn buffer_matches_native_scalar(buffer: &PyUntypedBuffer, scalar: ScalarKind) -> bool {
    match buffer_scalar_format(buffer, scalar) {
        Some(BufferEndian::Native) => true,
        Some(BufferEndian::Little) => cfg!(target_endian = "little"),
        Some(BufferEndian::Big) => cfg!(target_endian = "big"),
        None => false,
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BufferBool(u8);

// SAFETY: BufferBool has the same one-byte representation as a PEP 3118 bool.
unsafe impl Element for BufferBool {
    fn is_compatible_format(format: &CStr) -> bool {
        format.to_bytes() == b"?"
    }
}

fn build_scalar_vector(
    builder: &mut FlatBufferBuilder<'_>,
    scalar: ScalarKind,
    value: &Bound<'_, PyAny>,
    vector_length_patches: &mut Vec<(u32, u32)>,
) -> PyResult<AnyOffset> {
    Ok(match scalar {
        ScalarKind::Bool => {
            if let Ok(buffer) = PyUntypedBuffer::get(value) {
                if buffer.item_size() != 1 || buffer.format().to_bytes() != b"?" {
                    return Err(PyTypeError::new_err(
                        "numeric vector field requires native Bool data",
                    ));
                }
                if buffer.dimensions() != 1 {
                    return Err(PyTypeError::new_err(
                        "numeric vector fields must be one-dimensional",
                    ));
                }
                let values = buffer.as_typed::<BufferBool>()?.to_vec(value.py())?;
                let bytes: &[u8] = cast_slice(&values);
                erase_offset(builder.create_vector(bytes))
            } else {
                let values = value.extract::<Vec<bool>>()?;
                erase_offset(builder.create_vector(&values))
            }
        }
        ScalarKind::Int8 => {
            create_scalar_vector!(builder, value, i8, scalar, vector_length_patches)
        }
        ScalarKind::Uint8 => {
            create_scalar_vector!(builder, value, u8, scalar, vector_length_patches)
        }
        ScalarKind::Int16 => {
            create_scalar_vector!(builder, value, i16, scalar, vector_length_patches)
        }
        ScalarKind::Uint16 => {
            create_scalar_vector!(builder, value, u16, scalar, vector_length_patches)
        }
        ScalarKind::Int32 => {
            create_scalar_vector!(builder, value, i32, scalar, vector_length_patches)
        }
        ScalarKind::Uint32 => {
            create_scalar_vector!(builder, value, u32, scalar, vector_length_patches)
        }
        ScalarKind::Int64 => {
            create_scalar_vector!(builder, value, i64, scalar, vector_length_patches)
        }
        ScalarKind::Uint64 => {
            create_scalar_vector!(builder, value, u64, scalar, vector_length_patches)
        }
        ScalarKind::Float32 => {
            create_scalar_vector!(builder, value, f32, scalar, vector_length_patches)
        }
        ScalarKind::Float64 => {
            create_scalar_vector!(builder, value, f64, scalar, vector_length_patches)
        }
    })
}

fn build_owned_raw_vector<'a, T: Pod + Push>(
    builder: &mut FlatBufferBuilder<'a>,
    values: Vec<T>,
    vector_length_patches: &mut Vec<(u32, u32)>,
) -> PyResult<AnyOffset>
where
    T::Output: 'a,
{
    if cfg!(target_endian = "big") {
        return Ok(erase_offset(builder.create_vector(&values)));
    }
    let count = u32::try_from(values.len())
        .map_err(|_| PyValueError::new_err("numeric vector length exceeds the format limit"))?;
    align_builder(builder, size_of::<T>())?;
    let bytes: &[u8] = cast_slice(&values);
    let offset = builder.create_vector(bytes);
    vector_length_patches.push((offset.value(), count));
    Ok(erase_offset(offset))
}

fn extract_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(value) = value.cast::<PyBytes>() {
        return Ok(value.as_bytes().to_vec());
    }
    let value = value
        .py()
        .import("builtins")?
        .getattr("bytes")?
        .call1((value,))?;
    Ok(value.cast::<PyBytes>()?.as_bytes().to_vec())
}

fn buffer_byte_length(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(PyUntypedBuffer::get(value)?.len_bytes())
}

fn build_byte_vector(
    builder: &mut FlatBufferBuilder<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<AnyOffset> {
    if let Ok(value) = value.cast::<PyBytes>() {
        return Ok(erase_offset(builder.create_vector(value.as_bytes())));
    }
    if let Ok(view) = value.cast::<PyMemoryView>() {
        let owner = view.getattr("obj")?;
        if let Ok(owner) = owner.cast::<PyBytes>() {
            let buffer = checked_byte_buffer(value)?;
            let values = immutable_buffer_subslice(&buffer, owner.as_bytes())?;
            return Ok(erase_offset(builder.create_vector(values)));
        }
        if let Ok(owner) = owner.cast::<NativeBuffer>() {
            let buffer = checked_byte_buffer(value)?;
            let owner = owner.borrow();
            let values = immutable_buffer_subslice(&buffer, &owner.data[owner.start..])?;
            return Ok(erase_offset(builder.create_vector(values)));
        }
    }
    if let Ok(buffer) = PyBuffer::<u8>::get(value) {
        check_byte_buffer(&buffer)?;
        let values = buffer.to_vec(value.py())?;
        return Ok(erase_offset(builder.create_vector(&values)));
    }
    let values = extract_bytes(value)?;
    Ok(erase_offset(builder.create_vector(&values)))
}

fn checked_byte_buffer(value: &Bound<'_, PyAny>) -> PyResult<PyBuffer<u8>> {
    let buffer = PyBuffer::<u8>::get(value)?;
    check_byte_buffer(&buffer)?;
    Ok(buffer)
}

fn check_byte_buffer(buffer: &PyBuffer<u8>) -> PyResult<()> {
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(PyTypeError::new_err(
            "byte vector fields must be one-dimensional and C-contiguous",
        ));
    }
    Ok(())
}

fn immutable_buffer_subslice<'a>(buffer: &PyBuffer<u8>, owner: &'a [u8]) -> PyResult<&'a [u8]> {
    let start = (buffer.buf_ptr() as usize)
        .checked_sub(owner.as_ptr() as usize)
        .ok_or_else(|| PyBufferError::new_err("buffer lies outside its immutable owner"))?;
    let end = start
        .checked_add(buffer.len_bytes())
        .filter(|&end| end <= owner.len())
        .ok_or_else(|| PyBufferError::new_err("buffer lies outside its immutable owner"))?;
    Ok(&owner[start..end])
}

struct Align<const VALUE: usize>;

impl<const VALUE: usize> Push for Align<VALUE> {
    type Output = Self;

    unsafe fn push(&self, _dst: &mut [u8], _written_len: usize) {}

    fn size() -> usize {
        0
    }

    fn alignment() -> PushAlignment {
        PushAlignment::new(VALUE)
    }
}

struct StructRef;

impl Push for StructRef {
    type Output = Self;

    unsafe fn push(&self, _dst: &mut [u8], _written_len: usize) {}

    fn size() -> usize {
        0
    }
}

fn align_builder(builder: &mut FlatBufferBuilder<'_>, alignment: usize) -> PyResult<()> {
    match alignment {
        1 => {
            builder.push(Align::<1>);
        }
        2 => {
            builder.push(Align::<2>);
        }
        4 => {
            builder.push(Align::<4>);
        }
        8 => {
            builder.push(Align::<8>);
        }
        value => {
            return Err(PyValueError::new_err(format!(
                "unsupported native struct alignment {value}"
            )));
        }
    }
    Ok(())
}

fn write_struct_scalar(
    buffer: &mut [u8],
    offset: usize,
    scalar: ScalarKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    macro_rules! write {
        ($ty:ty) => {{
            let bytes = value.extract::<$ty>()?.to_le_bytes();
            buffer[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }};
    }
    match scalar {
        ScalarKind::Bool => buffer[offset] = u8::from(value.extract::<bool>()?),
        ScalarKind::Int8 => buffer[offset] = value.extract::<i8>()? as u8,
        ScalarKind::Uint8 => buffer[offset] = value.extract::<u8>()?,
        ScalarKind::Int16 => write!(i16),
        ScalarKind::Uint16 => write!(u16),
        ScalarKind::Int32 => write!(i32),
        ScalarKind::Uint32 => write!(u32),
        ScalarKind::Int64 => write!(i64),
        ScalarKind::Uint64 => write!(u64),
        ScalarKind::Float32 => write!(f32),
        ScalarKind::Float64 => write!(f64),
    }
    Ok(())
}

#[pyclass(module = "msgspec_flatbuffers._native", frozen)]
pub struct NativePlan {
    objects: Vec<ObjectWire>,
    by_name: HashMap<String, usize>,
    model_types: OnceLock<HashMap<usize, usize>>,
    dynamic_encoder: Option<Py<PyAny>>,
}

impl NativePlan {
    fn object(&self, name: &str) -> PyResult<&ObjectWire> {
        self.by_name
            .get(name)
            .map(|index| &self.objects[*index])
            .ok_or_else(|| PyValueError::new_err(format!("unknown native table {name:?}")))
    }

    fn target_object(&self, field: &FieldWire) -> PyResult<&ObjectWire> {
        field
            .target_index
            .map(|index| &self.objects[index])
            .ok_or_else(|| PyValueError::new_err("native field has no target"))
    }

    fn encode_dynamic_value<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        allowed_prefix: &str,
    ) -> PyResult<(String, Bound<'py, PyAny>)> {
        let encoded = self
            .dynamic_encoder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("native dynamic encoder is not loaded"))?
            .bind(value.py())
            .call1((value, allowed_prefix))?;
        let encoded = encoded.cast::<PyTuple>()?;
        let tag = encoded.get_item(0)?.extract::<String>()?;
        let data = encoded.get_item(1)?;
        Ok((tag, data))
    }

    fn union_arm<'a>(
        &self,
        field: &'a FieldWire,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<&'a ArmWire> {
        let model_types = self
            .model_types
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
        let value_type = value.get_type();
        let target = model_types.get(&(value_type.as_ptr() as usize));
        let arm =
            target.and_then(|target| field.arms.iter().find(|arm| arm.target_index == *target));
        let type_name = value_type.name()?.to_str()?.to_owned();
        arm.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "union field {:?} does not accept model type {}",
                field.name, type_name
            ))
        })
    }

    fn encode_struct_into(
        &self,
        object: &ObjectWire,
        model: &Bound<'_, PyAny>,
        buffer: &mut [u8],
    ) -> PyResult<()> {
        if !object.is_struct {
            return Err(PyTypeError::new_err(format!(
                "{} is a table, not a struct",
                object.name
            )));
        }
        buffer.fill(0);
        for field in &object.fields {
            if field.kind != FieldKind::Scalar {
                return Err(PyNotImplementedError::new_err(
                    "nested native structs are not implemented",
                ));
            }
            write_struct_scalar(
                buffer,
                field.offset,
                field.scalar.ok_or_else(|| {
                    PyValueError::new_err("native struct field has no scalar kind")
                })?,
                &model.getattr(field.name.as_str())?,
            )?;
        }
        Ok(())
    }

    fn encode_struct(&self, object: &ObjectWire, model: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let mut buffer = vec![0; object.byte_size];
        self.encode_struct_into(object, model, &mut buffer)?;
        Ok(buffer)
    }

    fn push_struct(
        &self,
        builder: &mut FlatBufferBuilder<'_>,
        object: &ObjectWire,
        model: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bytes = self.encode_struct(object, model)?;
        align_builder(builder, object.min_alignment)?;
        for value in bytes.into_iter().rev() {
            builder.push(value);
        }
        Ok(())
    }

    fn load_fields<'py>(
        &self,
        object: &ObjectWire,
        model: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<FieldState<'py>>> {
        object
            .fields
            .iter()
            .map(|field| {
                Ok(FieldState {
                    value: model.getattr(field.name.as_str())?,
                    offset: None,
                    discriminator: None,
                    prepared: None,
                    scalar: None,
                })
            })
            .collect()
    }

    fn estimate_table(&self, object: &ObjectWire, model: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut fields = self.load_fields(object, model)?;
        self.estimate_table_fields(object, &mut fields)
    }

    fn estimate_table_fields(
        &self,
        object: &ObjectWire,
        fields: &mut [FieldState<'_>],
    ) -> PyResult<usize> {
        if object.is_struct {
            return Err(PyTypeError::new_err(format!(
                "{} is a struct, not a table",
                object.name
            )));
        }

        let mut size = 8usize;
        let mut highest_slot: Option<u16> = None;
        for (field, state) in object.fields.iter().zip(fields) {
            let value = &state.value;
            if value.is_none() {
                continue;
            }
            if field.kind == FieldKind::Scalar {
                let scalar = ScalarValue::extract(
                    field.scalar.ok_or_else(|| {
                        PyValueError::new_err("native scalar field has no scalar kind")
                    })?,
                    value,
                )?;
                state.scalar = Some(scalar);
                if !field.optional && scalar.is_default(&field.default)? {
                    continue;
                }
            }

            highest_slot = Some(highest_slot.map_or(field.slot, |slot| slot.max(field.slot)));
            let referenced = match field.kind {
                FieldKind::Scalar => {
                    let width = scalar_size(field.scalar)?;
                    size = size.saturating_add(aligned_payload_size(width, width));
                    0
                }
                FieldKind::Struct => {
                    let target = self.target_object(field)?;
                    size = size.saturating_add(aligned_payload_size(
                        target.byte_size,
                        target.min_alignment,
                    ));
                    0
                }
                FieldKind::String => {
                    size = size.saturating_add(7);
                    let length = value.cast::<PyString>()?.to_str()?.len();
                    aligned_payload_size(length.saturating_add(5), 4)
                }
                FieldKind::Table => {
                    size = size.saturating_add(7);
                    let target = self.target_object(field)?;
                    self.estimate_table(target, value)?
                }
                FieldKind::VectorByte => {
                    size = size.saturating_add(7);
                    let length = PyUntypedBuffer::get(value)
                        .map(|buffer| buffer.len_bytes())
                        .unwrap_or_else(|_| value.len().unwrap_or(0));
                    aligned_payload_size(length.saturating_add(4), 4)
                }
                FieldKind::VectorScalar => {
                    size = size.saturating_add(7);
                    let width = scalar_size(field.scalar)?;
                    let length = sequence_len(value)?;
                    aligned_payload_size(
                        length.saturating_mul(width).saturating_add(4),
                        width.max(4),
                    )
                }
                FieldKind::VectorString => {
                    size = size.saturating_add(7);
                    let length = sequence_len(value)?;
                    let strings = sampled_sequence_size(value, |item| {
                        let length = item.cast::<PyString>()?.to_str()?.len();
                        Ok(aligned_payload_size(length.saturating_add(5), 4))
                    })?;
                    aligned_payload_size(length.saturating_mul(4).saturating_add(4), 4)
                        .saturating_add(strings)
                }
                FieldKind::VectorTable => {
                    size = size.saturating_add(7);
                    let length = sequence_len(value)?;
                    let target = self.target_object(field)?;
                    let tables =
                        sampled_sequence_size(value, |item| self.estimate_table(target, &item))?;
                    aligned_payload_size(length.saturating_mul(4).saturating_add(4), 4)
                        .saturating_add(tables)
                }
                FieldKind::VectorStruct => {
                    size = size.saturating_add(7);
                    let length = sequence_len(value)?;
                    let target = self.target_object(field)?;
                    aligned_payload_size(
                        length.saturating_mul(target.byte_size).saturating_add(4),
                        target.min_alignment.max(4),
                    )
                }
                FieldKind::Nested => {
                    size = size.saturating_add(7);
                    let data = value.call_method0("to_flatbuffer")?;
                    let data_size = buffer_byte_length(&data)?;
                    state.prepared = Some(PreparedValue::Nested(data));
                    aligned_payload_size(data_size.saturating_add(4), 4)
                }
                FieldKind::Dynamic => {
                    size = size.saturating_add(7);
                    if let Some(type_slot) = field.type_slot {
                        highest_slot =
                            Some(highest_slot.map_or(type_slot, |slot| slot.max(type_slot)));
                        size = size.saturating_add(7);
                    }
                    let allowed_prefix = field.allowed_prefix.as_deref().ok_or_else(|| {
                        PyValueError::new_err("dynamic field has no allowed prefix")
                    })?;
                    let (tag, data) = self.encode_dynamic_value(value, allowed_prefix)?;
                    let data_size = buffer_byte_length(&data)?;
                    let tag_size = tag.len();
                    state.prepared = Some(PreparedValue::Dynamic { tag, data });
                    aligned_payload_size(tag_size.saturating_add(5), 4)
                        .saturating_add(aligned_payload_size(data_size.saturating_add(4), 4))
                }
                FieldKind::Union => {
                    size = size.saturating_add(7);
                    if let Some(type_slot) = field.type_slot {
                        highest_slot =
                            Some(highest_slot.map_or(type_slot, |slot| slot.max(type_slot)));
                    }
                    let width = scalar_size(field.type_scalar)?;
                    size = size.saturating_add(aligned_payload_size(width, width));
                    let arm = self.union_arm(field, value)?;
                    self.estimate_table(&self.objects[arm.target_index], value)?
                }
                FieldKind::UnionVector => {
                    size = size.saturating_add(7);
                    if let Some(type_slot) = field.type_slot {
                        highest_slot =
                            Some(highest_slot.map_or(type_slot, |slot| slot.max(type_slot)));
                        size = size.saturating_add(7);
                    }
                    let length = sequence_len(value)?;
                    let width = scalar_size(field.type_scalar)?;
                    let tables = sampled_sequence_size(value, |item| {
                        let arm = self.union_arm(field, &item)?;
                        self.estimate_table(&self.objects[arm.target_index], &item)
                    })?;
                    aligned_payload_size(length.saturating_mul(4).saturating_add(4), 4)
                        .saturating_add(aligned_payload_size(
                            length.saturating_mul(width).saturating_add(4),
                            width.max(4),
                        ))
                        .saturating_add(tables)
                }
            };
            size = size.saturating_add(referenced);
        }

        if let Some(slot) = highest_slot {
            size = size.saturating_add(2usize.saturating_mul(slot as usize + 1));
        }
        Ok(size)
    }

    fn build_table<'fbb, 'py>(
        &self,
        builder: &mut FlatBufferBuilder<'fbb>,
        object: &ObjectWire,
        model: &Bound<'py, PyAny>,
        prepared: Option<Vec<FieldState<'py>>>,
        vector_length_patches: &mut Vec<(u32, u32)>,
    ) -> PyResult<WIPOffset<TableFinishedWIPOffset>> {
        if object.is_struct {
            return Err(PyTypeError::new_err(format!(
                "{} is a struct, not a table",
                object.name
            )));
        }
        let mut states = match prepared {
            Some(fields) => fields,
            None => self.load_fields(object, model)?,
        };
        for (index, field) in object.fields.iter().enumerate() {
            if !matches!(
                field.kind,
                FieldKind::String
                    | FieldKind::Table
                    | FieldKind::VectorByte
                    | FieldKind::VectorScalar
                    | FieldKind::VectorString
                    | FieldKind::VectorTable
                    | FieldKind::VectorStruct
                    | FieldKind::Nested
                    | FieldKind::Dynamic
                    | FieldKind::Union
                    | FieldKind::UnionVector
            ) {
                continue;
            }
            let prepared = states[index].prepared.take();
            let value = &states[index].value;
            if value.is_none() {
                if field.required {
                    return Err(PyTypeError::new_err(format!(
                        "required field {:?} is absent",
                        field.name
                    )));
                }
                continue;
            }
            let mut discriminator = None;
            let offset = match field.kind {
                FieldKind::String => {
                    erase_offset(builder.create_string(value.cast::<PyString>()?.to_str()?))
                }
                FieldKind::Table => erase_offset(self.build_table(
                    builder,
                    self.target_object(field)?,
                    value,
                    None,
                    vector_length_patches,
                )?),
                FieldKind::VectorByte => build_byte_vector(builder, value)?,
                FieldKind::VectorScalar => build_scalar_vector(
                    builder,
                    field.scalar.ok_or_else(|| {
                        PyValueError::new_err("native scalar vector has no scalar kind")
                    })?,
                    value,
                    vector_length_patches,
                )?,
                FieldKind::VectorString => {
                    let mut values = Vec::with_capacity(sequence_len(value)?);
                    for_each_sequence(value, |item| {
                        values.push(builder.create_string(item.cast::<PyString>()?.to_str()?));
                        Ok(())
                    })?;
                    erase_offset(builder.create_vector(&values))
                }
                FieldKind::VectorTable => {
                    let target = self.target_object(field)?;
                    let mut values = Vec::with_capacity(sequence_len(value)?);
                    for_each_sequence(value, |item| {
                        values.push(self.build_table(
                            builder,
                            target,
                            &item,
                            None,
                            vector_length_patches,
                        )?);
                        Ok(())
                    })?;
                    erase_offset(builder.create_vector(&values))
                }
                FieldKind::VectorStruct => {
                    let target = self.target_object(field)?;
                    let length = sequence_len(value)?;
                    let byte_length = length.checked_mul(target.byte_size).ok_or_else(|| {
                        PyValueError::new_err("struct vector size exceeds the format limit")
                    })?;
                    let mut bytes = Vec::with_capacity(byte_length);
                    for_each_sequence(value, |item| {
                        let start = bytes.len();
                        bytes.resize(start + target.byte_size, 0);
                        self.encode_struct_into(target, &item, &mut bytes[start..])
                    })?;
                    if target.min_alignment > 4 {
                        align_builder(builder, target.min_alignment)?;
                    }
                    let offset = builder.create_vector(&bytes);
                    let count = u32::try_from(length).map_err(|_| {
                        PyValueError::new_err("struct vector length exceeds the format limit")
                    })?;
                    vector_length_patches.push((offset.value(), count));
                    erase_offset(offset)
                }
                FieldKind::Nested => {
                    let data = match prepared {
                        Some(PreparedValue::Nested(data)) => data,
                        _ => value.call_method0("to_flatbuffer")?,
                    };
                    build_byte_vector(builder, &data)?
                }
                FieldKind::Dynamic => {
                    let (tag, data) = match prepared {
                        Some(PreparedValue::Dynamic { tag, data }) => (tag, data),
                        _ => {
                            let allowed_prefix =
                                field.allowed_prefix.as_deref().ok_or_else(|| {
                                    PyValueError::new_err("dynamic field has no allowed prefix")
                                })?;
                            self.encode_dynamic_value(value, allowed_prefix)?
                        }
                    };
                    discriminator = Some(Discriminator::Offset(erase_offset(
                        builder.create_string(&tag),
                    )));
                    build_byte_vector(builder, &data)?
                }
                FieldKind::Union => {
                    let arm = self.union_arm(field, value)?;
                    discriminator = Some(Discriminator::Union(arm.tag));
                    erase_offset(self.build_table(
                        builder,
                        &self.objects[arm.target_index],
                        value,
                        None,
                        vector_length_patches,
                    )?)
                }
                FieldKind::UnionVector => {
                    let length = sequence_len(value)?;
                    let mut values = Vec::with_capacity(length);
                    let mut tags = Vec::with_capacity(length);
                    for_each_sequence(value, |item| {
                        let arm = self.union_arm(field, &item)?;
                        tags.push(arm.tag as u8);
                        values.push(self.build_table(
                            builder,
                            &self.objects[arm.target_index],
                            &item,
                            None,
                            vector_length_patches,
                        )?);
                        Ok(())
                    })?;
                    discriminator = Some(Discriminator::Offset(erase_offset(
                        builder.create_vector(&tags),
                    )));
                    erase_offset(builder.create_vector(&values))
                }
                FieldKind::Scalar | FieldKind::Struct => unreachable!(),
            };
            states[index].offset = Some(offset);
            states[index].discriminator = discriminator;
        }

        let table = builder.start_table();
        for (field, state) in object.fields.iter().zip(&states) {
            let slot = 4 + field.slot * 2;
            if let Some(offset) = state.offset {
                builder.push_slot_always(slot, offset);
                match field.kind {
                    FieldKind::Dynamic => {
                        let Some(Discriminator::Offset(type_offset)) = state.discriminator else {
                            return Err(PyTypeError::new_err("dynamic FlatBuffer has no type tag"));
                        };
                        let type_slot = 4 + field.type_slot.ok_or_else(|| {
                            PyValueError::new_err("dynamic field has no type slot")
                        })? * 2;
                        builder.push_slot_always(type_slot, type_offset);
                    }
                    FieldKind::Union => {
                        let Some(Discriminator::Union(tag)) = state.discriminator else {
                            return Err(PyTypeError::new_err("union has no discriminator"));
                        };
                        let type_slot = 4 + field
                            .type_slot
                            .ok_or_else(|| PyValueError::new_err("union has no type slot"))?
                            * 2;
                        push_type_id(builder, type_slot, field.type_scalar, tag)?;
                    }
                    FieldKind::UnionVector => {
                        let Some(Discriminator::Offset(type_offset)) = state.discriminator else {
                            return Err(PyTypeError::new_err(
                                "union vector has no discriminator vector",
                            ));
                        };
                        let type_slot = 4 + field.type_slot.ok_or_else(|| {
                            PyValueError::new_err("union vector has no type slot")
                        })? * 2;
                        builder.push_slot_always(type_slot, type_offset);
                    }
                    _ => {}
                }
                continue;
            }
            if field.kind != FieldKind::Scalar {
                let value = &state.value;
                if value.is_none() {
                    if field.required {
                        return Err(PyTypeError::new_err(format!(
                            "required field {:?} is absent",
                            field.name
                        )));
                    }
                    continue;
                }
            }
            match field.kind {
                FieldKind::Scalar => {
                    let value = &state.value;
                    if field.optional && value.is_none() {
                        continue;
                    }
                    if let Some(value) = state.scalar {
                        value.push(builder, slot, &field.default, field.optional)?;
                    } else {
                        push_scalar(
                            builder,
                            slot,
                            field.scalar.ok_or_else(|| {
                                PyValueError::new_err("native scalar field has no scalar kind")
                            })?,
                            value,
                            &field.default,
                            field.optional,
                        )?;
                    }
                }
                FieldKind::Struct => {
                    let value = &state.value;
                    let target = self.target_object(field)?;
                    self.push_struct(builder, target, value)?;
                    builder.push_slot_always(slot, StructRef);
                }
                FieldKind::String
                | FieldKind::Table
                | FieldKind::VectorByte
                | FieldKind::VectorScalar
                | FieldKind::VectorString
                | FieldKind::VectorTable
                | FieldKind::VectorStruct
                | FieldKind::Nested
                | FieldKind::Dynamic
                | FieldKind::Union
                | FieldKind::UnionVector => {}
            }
        }
        Ok(builder.end_table(table))
    }
}

#[pymethods]
impl NativePlan {
    #[new]
    fn new(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut wire: ModuleWire = rmp_serde::from_slice(data.as_bytes()).map_err(|error| {
            PyValueError::new_err(format!("invalid native packing plan: {error}"))
        })?;
        if wire.version != 1 {
            return Err(PyValueError::new_err(format!(
                "unsupported native packing plan version {}",
                wire.version
            )));
        }
        let by_name: HashMap<_, _> = wire
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| (object.name.clone(), index))
            .collect();
        for object in &mut wire.objects {
            for field in &mut object.fields {
                if let Some(target) = field.target.take() {
                    field.target_index = Some(*by_name.get(&target).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "native field target {target:?} does not exist"
                        ))
                    })?);
                }
                for arm in &mut field.arms {
                    arm.target_index = *by_name.get(&arm.target).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "native union target {:?} does not exist",
                            arm.target
                        ))
                    })?;
                    arm.target.clear();
                }
            }
        }
        let has_dynamic = wire
            .objects
            .iter()
            .flat_map(|object| &object.fields)
            .any(|field| field.kind == FieldKind::Dynamic);
        let dynamic_encoder = if has_dynamic {
            Some(
                data.py()
                    .import("msgspec_flatbuffers._dynamic")?
                    .getattr("encode_dynamic")?
                    .unbind(),
            )
        } else {
            None
        };
        Ok(Self {
            objects: wire.objects,
            by_name,
            model_types: OnceLock::new(),
            dynamic_encoder,
        })
    }

    fn bind_types(&self, types: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut model_types = HashMap::with_capacity(types.len());
        for (name, model_type) in types.iter() {
            let name = name.extract::<String>()?;
            let object_index = *self.by_name.get(&name).ok_or_else(|| {
                PyValueError::new_err(format!("cannot bind unknown native model type {name:?}"))
            })?;
            let model_type = model_type.cast::<PyType>()?;
            model_types.insert(model_type.as_ptr() as usize, object_index);
        }
        self.model_types
            .set(model_types)
            .map_err(|_| PyRuntimeError::new_err("native model types are already bound"))
    }

    #[pyo3(signature = (root, model, *, identifier=None, size_prefixed=false, initial_size=0))]
    fn pack<'py>(
        &self,
        py: Python<'py>,
        root: &str,
        model: &Bound<'py, PyAny>,
        identifier: Option<&str>,
        size_prefixed: bool,
        initial_size: isize,
    ) -> PyResult<Bound<'py, PyMemoryView>> {
        if initial_size < 0 {
            return Err(PyValueError::new_err(
                "initial_size must be greater than or equal to zero",
            ));
        }
        if let Some(value) = identifier
            && (value.len() != 4 || !value.is_ascii())
        {
            return Err(PyValueError::new_err(
                "FlatBuffers file identifiers must contain four ASCII bytes",
            ));
        }
        let root_object = self.object(root)?;
        let mut root_fields = self.load_fields(root_object, model)?;
        let finish_size = 4usize
            .saturating_add(identifier.map_or(0, |_| 4))
            .saturating_add(if size_prefixed { 4 } else { 0 });
        let estimate = self
            .estimate_table_fields(root_object, &mut root_fields)?
            .saturating_add(finish_size);
        let estimate = estimate.saturating_mul(101).saturating_add(99) / 100;
        let capacity = (initial_size as usize).max(estimate);
        if capacity > FLATBUFFERS_MAX_BUFFER_SIZE {
            return Err(PyValueError::new_err(
                "estimated FlatBuffer size exceeds the 2 GiB format limit",
            ));
        }
        let mut builder = FlatBufferBuilder::with_capacity(capacity);
        let mut vector_length_patches = Vec::new();
        let root = self.build_table(
            &mut builder,
            root_object,
            model,
            Some(root_fields),
            &mut vector_length_patches,
        )?;
        if size_prefixed {
            builder.finish_size_prefixed(root, identifier);
        } else {
            builder.finish(root, identifier);
        }
        if !vector_length_patches.is_empty() {
            let (buffer, _) = builder.mut_finished_buffer();
            for (reverse_offset, length) in vector_length_patches {
                let position = buffer.len() - reverse_offset as usize;
                buffer[position..position + 4].copy_from_slice(&length.to_le_bytes());
            }
        }
        let (data, start) = builder.collapse();
        let owner = Bound::new(py, NativeBuffer { data, start })?;
        PyMemoryView::from(owner.as_any())
    }

    fn __repr__(&self) -> String {
        format!("NativePlan(objects={})", self.objects.len())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBuffer>()?;
    module.add_class::<NativePlan>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BufferEndian, ScalarKind, scalar_format};

    #[test]
    fn scalar_buffer_formats_use_declared_width_and_endianness() {
        assert!(matches!(
            scalar_format(b"l", 8, ScalarKind::Int64),
            Some(BufferEndian::Native)
        ));
        assert!(matches!(
            scalar_format(b"l", 4, ScalarKind::Int32),
            Some(BufferEndian::Native)
        ));
        assert!(scalar_format(b"l", 8, ScalarKind::Int32).is_none());
        assert!(matches!(
            scalar_format(b"<q", 8, ScalarKind::Int64),
            Some(BufferEndian::Little)
        ));
        assert!(matches!(
            scalar_format(b">q", 8, ScalarKind::Int64),
            Some(BufferEndian::Big)
        ));
        assert!(scalar_format(b"<d", 8, ScalarKind::Int64).is_none());
    }
}
