use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use bytemuck::{Pod, Zeroable, cast_slice};
use flatbuffers::{
    FLATBUFFERS_MAX_BUFFER_SIZE, FlatBufferBuilder, Push, PushAlignment, TableFinishedWIPOffset,
    UnionWIPOffset, VOffsetT, WIPOffset,
};
use pyo3::IntoPyObjectExt;
use pyo3::buffer::{Element, PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::{
    PyBufferError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyUnicodeDecodeError,
    PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
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
    #[serde(skip)]
    index: usize,
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
    ArrayScalar,
    ArrayStruct,
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
    enum_type: Option<String>,
    #[serde(default)]
    dynamic_type: Option<String>,
    #[serde(default)]
    arms: Vec<ArmWire>,
    #[serde(default)]
    fixed_length: usize,
    #[serde(default)]
    element_size: usize,
}

#[derive(Deserialize)]
struct ArmWire {
    tag: u64,
    target: String,
    #[serde(skip)]
    target_index: usize,
}

type AnyOffset = WIPOffset<UnionWIPOffset>;

struct NullableOffset(Option<u32>);

impl Push for NullableOffset {
    type Output = u32;

    unsafe fn push(&self, dst: &mut [u8], written_len: usize) {
        let value = self
            .0
            .map_or(0, |offset| (4 + written_len - offset as usize) as u32);
        unsafe { value.push(dst, written_len) };
    }
}

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

struct BoundTypes {
    by_pointer: HashMap<usize, usize>,
    by_name: HashMap<String, Py<PyType>>,
}

struct BoundModelSubclass {
    _type_owner: Py<PyType>,
    object_index: usize,
}

type ChildTypeKey = (usize, u16, usize);

#[pyclass(module = "msgspec_flatbuffers._native", frozen)]
pub struct NativeModelTypes {
    plan_identity: Arc<()>,
    root_index: usize,
    root_type: Py<PyType>,
    child_types: HashMap<ChildTypeKey, Py<PyType>>,
}

#[derive(Clone, Copy)]
struct TableInfo {
    table_offset: usize,
    vtable_offset: usize,
    vtable_size: usize,
    object_size: usize,
}

#[derive(Clone, Copy)]
struct RootDecode<'a> {
    offset: usize,
    size_prefixed: bool,
    identifier: Option<&'a str>,
    check_identifier: bool,
}

#[derive(Clone, Copy)]
struct DecodeContext<'a, 'py> {
    model_types: Option<&'a NativeModelTypes>,
    dynamic_overrides: Option<&'a Bound<'py, PyAny>>,
}

const MAX_DECODED_OBJECTS: usize = 1_000_000;

#[derive(Clone, Copy)]
struct FieldTask {
    object_index: usize,
    model_type_index: usize,
    field_index: usize,
}

#[derive(Clone, Copy)]
enum ObjectLocation {
    Table(TableInfo),
    Struct { offset: usize },
}

#[derive(Clone, Copy)]
struct ObjectFrame {
    next_field: FieldTask,
    location: ObjectLocation,
    value_start: usize,
}

#[derive(Clone, Copy)]
enum VectorElement {
    Table,
    Struct { stride: usize },
}

#[derive(Clone, Copy)]
struct ObjectVectorFrame {
    object_index: usize,
    model_type_index: usize,
    element: VectorElement,
    start: usize,
    length: usize,
    index: usize,
    result_start: usize,
}

#[derive(Clone, Copy)]
struct UnionVectorFrame {
    field: FieldTask,
    type_scalar: ScalarKind,
    type_start: usize,
    value_start: usize,
    length: usize,
    width: usize,
    index: usize,
    result_start: usize,
}

#[derive(Clone, Copy)]
enum DecodeFrame {
    Object(ObjectFrame),
    ObjectVector(ObjectVectorFrame),
    UnionVector(UnionVectorFrame),
}

struct Materializer<'plan, 'data, 'context, 'py> {
    plan: &'plan NativePlan,
    py: Python<'py>,
    data: &'data [u8],
    context: DecodeContext<'context, 'py>,
    frames: Vec<DecodeFrame>,
    values: Vec<Py<PyAny>>,
    model_types: Vec<Py<PyType>>,
    model_type_indices: HashMap<usize, usize>,
    decoded_objects: usize,
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

#[inline]
fn include_slot(highest_slot: &mut Option<u16>, slot: u16) {
    *highest_slot = Some(highest_slot.map_or(slot, |highest| highest.max(slot)));
}

#[inline]
fn vtable_offset(slot: u16) -> VOffsetT {
    4 + slot * 2
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
    fn read(kind: ScalarKind, bytes: &[u8]) -> Self {
        macro_rules! read {
            ($ty:ty) => {
                <$ty>::from_le_bytes(bytes.try_into().expect("checked scalar width"))
            };
        }
        match kind {
            ScalarKind::Bool => Self::Bool(bytes[0] != 0),
            ScalarKind::Int8 => Self::Int8(bytes[0] as i8),
            ScalarKind::Uint8 => Self::Uint8(bytes[0]),
            ScalarKind::Int16 => Self::Int16(read!(i16)),
            ScalarKind::Uint16 => Self::Uint16(read!(u16)),
            ScalarKind::Int32 => Self::Int32(read!(i32)),
            ScalarKind::Uint32 => Self::Uint32(read!(u32)),
            ScalarKind::Int64 => Self::Int64(read!(i64)),
            ScalarKind::Uint64 => Self::Uint64(read!(u64)),
            ScalarKind::Float32 => Self::Float32(read!(f32)),
            ScalarKind::Float64 => Self::Float64(read!(f64)),
        }
    }

    fn from_default(kind: ScalarKind, default: &Value) -> PyResult<Self> {
        Ok(match kind {
            ScalarKind::Bool => Self::Bool(default_bool(default)?),
            ScalarKind::Int8 => Self::Int8(default_i64(default)? as i8),
            ScalarKind::Uint8 => Self::Uint8(default_u64(default)? as u8),
            ScalarKind::Int16 => Self::Int16(default_i64(default)? as i16),
            ScalarKind::Uint16 => Self::Uint16(default_u64(default)? as u16),
            ScalarKind::Int32 => Self::Int32(default_i64(default)? as i32),
            ScalarKind::Uint32 => Self::Uint32(default_u64(default)? as u32),
            ScalarKind::Int64 => Self::Int64(default_i64(default)?),
            ScalarKind::Uint64 => Self::Uint64(default_u64(default)?),
            ScalarKind::Float32 => Self::Float32(default_f64(default)? as f32),
            ScalarKind::Float64 => Self::Float64(default_f64(default)?),
        })
    }

    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Bool(value) => value.into_py_any(py),
            Self::Int8(value) => value.into_py_any(py),
            Self::Uint8(value) => value.into_py_any(py),
            Self::Int16(value) => value.into_py_any(py),
            Self::Uint16(value) => value.into_py_any(py),
            Self::Int32(value) => value.into_py_any(py),
            Self::Uint32(value) => value.into_py_any(py),
            Self::Int64(value) => value.into_py_any(py),
            Self::Uint64(value) => value.into_py_any(py),
            Self::Float32(value) => value.into_py_any(py),
            Self::Float64(value) => value.into_py_any(py),
        }
    }

    fn as_u64(self) -> PyResult<u64> {
        match self {
            Self::Uint8(value) => Ok(value.into()),
            Self::Uint16(value) => Ok(value.into()),
            Self::Uint32(value) => Ok(value.into()),
            Self::Uint64(value) => Ok(value),
            _ => Err(PyValueError::new_err(
                "union discriminator must use an unsigned integer type",
            )),
        }
    }

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

fn nonnegative_usize(value: isize, name: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be greater than or equal to zero")))
}

fn validate_identifier(identifier: Option<&str>) -> PyResult<()> {
    if let Some(value) = identifier
        && (value.len() != 4 || !value.is_ascii())
    {
        return Err(PyValueError::new_err(
            "FlatBuffers file identifiers must contain four ASCII bytes",
        ));
    }
    Ok(())
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

#[inline]
fn with_input_bytes<T>(
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
            return decode(immutable_buffer_subslice(
                &buffer,
                &owner.data[owner.start..],
            )?);
        }
    }
    let data = extract_bytes(value)?;
    decode(&data)
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

fn require_fixed_array_length(expected: usize, actual: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "fixed array requires {expected} items, got {actual}"
        )));
    }
    Ok(())
}

fn write_struct_scalar_array(
    buffer: &mut [u8],
    offset: usize,
    length: usize,
    scalar: ScalarKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    macro_rules! write_array {
        ($ty:ty) => {{
            let values = if let Ok(raw) = PyUntypedBuffer::get(value) {
                if raw.dimensions() != 1 || !buffer_matches_native_scalar(&raw, scalar) {
                    return Err(PyTypeError::new_err(format!(
                        "numeric array field requires one-dimensional native {:?} data",
                        scalar,
                    )));
                }
                raw.as_typed::<$ty>()?.to_vec(value.py())?
            } else {
                value.extract::<Vec<$ty>>()?
            };
            require_fixed_array_length(length, values.len())?;
            let byte_length = length * size_of::<$ty>();
            let output = buffer
                .get_mut(offset..offset + byte_length)
                .ok_or_else(|| PyValueError::new_err("fixed array lies outside its struct"))?;
            if cfg!(target_endian = "little") {
                output.copy_from_slice(cast_slice(&values));
            } else {
                for (chunk, item) in output.chunks_exact_mut(size_of::<$ty>()).zip(values) {
                    chunk.copy_from_slice(&item.to_le_bytes());
                }
            }
        }};
    }

    match scalar {
        ScalarKind::Bool => {
            if let Ok(raw) = PyUntypedBuffer::get(value) {
                if raw.dimensions() != 1 || raw.format().to_bytes() != b"?" {
                    return Err(PyTypeError::new_err(
                        "numeric array field requires one-dimensional native Bool data",
                    ));
                }
                let values = raw.as_typed::<BufferBool>()?.to_vec(value.py())?;
                require_fixed_array_length(length, values.len())?;
                buffer[offset..offset + length].copy_from_slice(cast_slice(&values));
            } else {
                let values = value.extract::<Vec<bool>>()?;
                require_fixed_array_length(length, values.len())?;
                for (output, item) in buffer[offset..offset + length].iter_mut().zip(values) {
                    *output = u8::from(item);
                }
            }
        }
        ScalarKind::Int8 => write_array!(i8),
        ScalarKind::Uint8 => write_array!(u8),
        ScalarKind::Int16 => write_array!(i16),
        ScalarKind::Uint16 => write_array!(u16),
        ScalarKind::Int32 => write_array!(i32),
        ScalarKind::Uint32 => write_array!(u32),
        ScalarKind::Int64 => write_array!(i64),
        ScalarKind::Uint64 => write_array!(u64),
        ScalarKind::Float32 => write_array!(f32),
        ScalarKind::Float64 => write_array!(f64),
    }
    Ok(())
}

#[pyclass(module = "msgspec_flatbuffers._native", frozen)]
pub struct NativePlan {
    identity: Arc<()>,
    objects: Vec<ObjectWire>,
    by_name: HashMap<String, usize>,
    bound_types: OnceLock<BoundTypes>,
    model_subclass_cache: Mutex<HashMap<usize, BoundModelSubclass>>,
    dynamic_encoder: Option<Py<PyAny>>,
    dynamic_registry: Py<PyAny>,
    numpy_empty: Py<PyAny>,
    buffer_bounds_error: Py<PyType>,
    invalid_buffer_error: Py<PyType>,
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

    fn bound_type<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyType>> {
        self.bound_types
            .get()
            .and_then(|types| types.by_name.get(name))
            .map(|value| value.bind(py).clone())
            .ok_or_else(|| PyRuntimeError::new_err(format!("native type {name:?} is not bound")))
    }

    fn prepare_decode<'a, 'py>(
        &self,
        py: Python<'py>,
        object: &ObjectWire,
        model_types: Option<&'a NativeModelTypes>,
        dynamic_overrides: Option<&'a Bound<'py, PyAny>>,
        mismatch_message: &'static str,
    ) -> PyResult<(Bound<'py, PyType>, DecodeContext<'a, 'py>)> {
        if let Some(model_types) = model_types
            && (!Arc::ptr_eq(&self.identity, &model_types.plan_identity)
                || model_types.root_index != object.index)
        {
            return Err(PyTypeError::new_err(mismatch_message));
        }
        let model_type = match model_types {
            Some(types) => types.root_type.bind(py).clone(),
            None => self.bound_type(py, &object.name)?,
        };
        Ok((
            model_type,
            DecodeContext {
                model_types,
                dynamic_overrides,
            },
        ))
    }

    fn child_decode_type<'py>(
        &self,
        py: Python<'py>,
        parent_type: &Bound<'py, PyType>,
        field: &FieldWire,
        target: &ObjectWire,
        model_types: Option<&NativeModelTypes>,
    ) -> PyResult<Bound<'py, PyType>> {
        if let Some(model_types) = model_types
            && let Some(model_type) = model_types.child_types.get(&(
                parent_type.as_ptr() as usize,
                field.slot,
                target.index,
            ))
        {
            return Ok(model_type.bind(py).clone());
        }
        self.bound_type(py, &target.name)
    }

    fn lock_model_subclass_cache(
        &self,
        py: Python<'_>,
    ) -> PyResult<MutexGuard<'_, HashMap<usize, BoundModelSubclass>>> {
        self.model_subclass_cache
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("native model subclass cache is poisoned"))
    }

    fn model_object_index(&self, model: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = model.py();
        let bound_types = self
            .bound_types
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
        let model_type = model.get_type();
        let pointer = model_type.as_ptr() as usize;
        if let Some(index) = bound_types.by_pointer.get(&pointer) {
            return Ok(*index);
        }
        if let Some(entry) = self.lock_model_subclass_cache(py)?.get(&pointer) {
            return Ok(entry.object_index);
        }

        let mut matches = Vec::new();
        for base in model_type
            .getattr("__mro__")?
            .cast::<PyTuple>()?
            .iter()
            .skip(1)
        {
            if let Some(index) = bound_types.by_pointer.get(&(base.as_ptr() as usize))
                && !matches.contains(index)
            {
                matches.push(*index);
            }
        }
        let [object_index] = matches.as_slice() else {
            return Err(PyTypeError::new_err(format!(
                "{} does not have exactly one generated FlatBuffer model base",
                model_type.name()?,
            )));
        };
        let generated_type = self.bound_type(py, &self.objects[*object_index].name)?;
        let model_fields = model_type
            .getattr("__struct_fields__")?
            .cast_into::<PyTuple>()?;
        let generated_fields = generated_type
            .getattr("__struct_fields__")?
            .cast_into::<PyTuple>()?;
        if model_fields.len() != generated_fields.len()
            || generated_fields
                .iter()
                .any(|field| !model_fields.contains(&field).unwrap_or(false))
        {
            return Err(PyTypeError::new_err(format!(
                "{} changes the serialized msgspec fields of {}",
                model_type.name()?,
                generated_type.name()?,
            )));
        }
        self.lock_model_subclass_cache(py)?.insert(
            pointer,
            BoundModelSubclass {
                _type_owner: model_type.unbind(),
                object_index: *object_index,
            },
        );
        Ok(*object_index)
    }

    fn require_model_type(&self, object: &ObjectWire, model: &Bound<'_, PyAny>) -> PyResult<()> {
        let actual = self.model_object_index(model)?;
        if actual == object.index {
            return Ok(());
        }
        let model_name = model.get_type().name()?.to_str()?.to_owned();
        let expected_name = self
            .bound_type(model.py(), &object.name)?
            .name()?
            .to_str()?
            .to_owned();
        Err(PyTypeError::new_err(format!(
            "{} must subclass {}",
            model_name, expected_name,
        )))
    }

    fn bounds_error(&self, py: Python<'_>, message: impl Into<String>) -> PyErr {
        PyErr::from_type(self.buffer_bounds_error.bind(py).clone(), (message.into(),))
    }

    fn invalid_error(&self, py: Python<'_>, message: impl Into<String>) -> PyErr {
        PyErr::from_type(
            self.invalid_buffer_error.bind(py).clone(),
            (message.into(),),
        )
    }

    #[inline]
    fn require_span(
        &self,
        py: Python<'_>,
        data: &[u8],
        offset: usize,
        size: usize,
        description: &str,
    ) -> PyResult<()> {
        if size > data.len() || offset > data.len() - size {
            return Err(self.bounds_error(
                py,
                format!(
                    "{description} at offset {offset} with size {size} exceeds a {}-byte buffer",
                    data.len()
                ),
            ));
        }
        Ok(())
    }

    #[inline]
    fn read_u16(&self, py: Python<'_>, data: &[u8], offset: usize) -> PyResult<u16> {
        self.require_span(py, data, offset, 2, "uint16")?;
        Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
    }

    #[inline]
    fn read_u32(&self, py: Python<'_>, data: &[u8], offset: usize) -> PyResult<u32> {
        self.require_span(py, data, offset, 4, "uint32")?;
        Ok(u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .expect("checked uint32 span"),
        ))
    }

    #[inline]
    fn read_i32(&self, py: Python<'_>, data: &[u8], offset: usize) -> PyResult<i32> {
        self.require_span(py, data, offset, 4, "int32")?;
        Ok(i32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .expect("checked int32 span"),
        ))
    }

    fn table_info(&self, py: Python<'_>, data: &[u8], table_offset: usize) -> PyResult<TableInfo> {
        let distance = self.read_i32(py, data, table_offset)? as i64;
        let vtable_offset = i64::try_from(table_offset)
            .ok()
            .and_then(|offset| offset.checked_sub(distance))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| {
                self.bounds_error(
                    py,
                    format!("vtable header lies outside a {}-byte buffer", data.len()),
                )
            })?;
        self.require_span(py, data, vtable_offset, 4, "vtable header")?;
        let vtable_size = usize::from(self.read_u16(py, data, vtable_offset)?);
        let object_size = usize::from(self.read_u16(py, data, vtable_offset + 2)?);
        if vtable_size < 4 || !vtable_size.is_multiple_of(2) || object_size < 4 {
            return Err(self.invalid_error(py, "invalid FlatBuffers table metadata"));
        }
        self.require_span(py, data, vtable_offset, vtable_size, "vtable")?;
        self.require_span(py, data, table_offset, object_size, "table")?;
        Ok(TableInfo {
            table_offset,
            vtable_offset,
            vtable_size,
            object_size,
        })
    }

    fn field_position(
        &self,
        py: Python<'_>,
        data: &[u8],
        table: TableInfo,
        vtable_field: usize,
        size: usize,
    ) -> PyResult<Option<usize>> {
        if vtable_field < 4 || !vtable_field.is_multiple_of(2) {
            return Err(PyValueError::new_err(
                "vtable field offsets must be even and at least 4",
            ));
        }
        if vtable_field >= table.vtable_size {
            return Ok(None);
        }
        let relative = usize::from(self.read_u16(py, data, table.vtable_offset + vtable_field)?);
        if relative == 0 {
            return Ok(None);
        }
        if size > table.object_size || relative < 4 || relative > table.object_size - size {
            return Err(self.invalid_error(py, "field lies outside its FlatBuffers table"));
        }
        Ok(Some(table.table_offset + relative))
    }

    fn offset_target(
        &self,
        py: Python<'_>,
        data: &[u8],
        position: usize,
        description: &str,
    ) -> PyResult<usize> {
        let relative = usize::try_from(self.read_u32(py, data, position)?).unwrap();
        if relative == 0 {
            return Err(self.invalid_error(py, format!("{description} contains a null offset")));
        }
        let target = position.checked_add(relative).ok_or_else(|| {
            self.bounds_error(py, format!("{description} target offset overflows"))
        })?;
        self.require_span(py, data, target, 1, description)?;
        Ok(target)
    }

    fn decode_string_at(
        &self,
        py: Python<'_>,
        data: &[u8],
        position: usize,
    ) -> PyResult<Py<PyAny>> {
        let target = self.offset_target(py, data, position, "string offset")?;
        let length = usize::try_from(self.read_u32(py, data, target)?).unwrap();
        let start = target + 4;
        let size = length
            .checked_add(1)
            .ok_or_else(|| self.bounds_error(py, "string length overflows"))?;
        self.require_span(py, data, start, size, "string data")?;
        if data[start + length] != 0 {
            return Err(self.invalid_error(py, "FlatBuffers string is not null-terminated"));
        }
        let bytes = &data[start..start + length];
        let value = std::str::from_utf8(bytes).map_err(|error| {
            let error_start = error.valid_up_to();
            let error_end = error_start + error.error_len().unwrap_or(1);
            PyUnicodeDecodeError::new_err((
                "utf-8",
                PyBytes::new(py, bytes).unbind(),
                error_start,
                error_end,
                error.to_string(),
            ))
        })?;
        Ok(PyString::new(py, value).into_any().unbind())
    }

    fn vector_info(
        &self,
        py: Python<'_>,
        data: &[u8],
        table: TableInfo,
        vtable_field: usize,
        item_size: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let Some(position) = self.field_position(py, data, table, vtable_field, 4)? else {
            return Ok(None);
        };
        let target = self.offset_target(py, data, position, "vector offset")?;
        let length = usize::try_from(self.read_u32(py, data, target)?).unwrap();
        let start = target + 4;
        let byte_length = length
            .checked_mul(item_size)
            .ok_or_else(|| self.bounds_error(py, "vector byte length overflows"))?;
        self.require_span(py, data, start, byte_length, "vector data")?;
        Ok(Some((start, length)))
    }

    fn read_scalar_value(
        &self,
        py: Python<'_>,
        data: &[u8],
        position: usize,
        kind: ScalarKind,
    ) -> PyResult<ScalarValue> {
        let width = scalar_size(Some(kind))?;
        self.require_span(py, data, position, width, "scalar data")?;
        Ok(ScalarValue::read(kind, &data[position..position + width]))
    }

    fn apply_enum(
        &self,
        py: Python<'_>,
        field: &FieldWire,
        value: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let Some(name) = field.enum_type.as_deref() else {
            return Ok(value);
        };
        Ok(self.bound_type(py, name)?.call1((value,))?.unbind())
    }

    fn decode_numeric_vector(
        &self,
        py: Python<'_>,
        data: &[u8],
        start: usize,
        length: usize,
        scalar: ScalarKind,
    ) -> PyResult<Py<PyAny>> {
        let width = scalar_size(Some(scalar))?;
        let byte_length = length
            .checked_mul(width)
            .ok_or_else(|| self.bounds_error(py, "numeric vector byte length overflows"))?;
        self.require_span(py, data, start, byte_length, "numeric vector data")?;
        let native_dtype = match scalar {
            ScalarKind::Bool => "?",
            ScalarKind::Int8 => "i1",
            ScalarKind::Uint8 => "u1",
            ScalarKind::Int16 => "i2",
            ScalarKind::Uint16 => "u2",
            ScalarKind::Int32 => "i4",
            ScalarKind::Uint32 => "u4",
            ScalarKind::Int64 => "i8",
            ScalarKind::Uint64 => "u8",
            ScalarKind::Float32 => "f4",
            ScalarKind::Float64 => "f8",
        };
        let array = self.numpy_empty.bind(py).call1((length, native_dtype))?;
        let output = PyUntypedBuffer::get(&array)?;
        if output.readonly() || !output.is_c_contiguous() || output.len_bytes() != byte_length {
            return Err(PyRuntimeError::new_err(
                "NumPy returned incompatible numeric vector storage",
            ));
        }
        if byte_length != 0 {
            unsafe {
                ptr::copy_nonoverlapping(
                    data[start..start + byte_length].as_ptr(),
                    output.buf_ptr().cast(),
                    byte_length,
                );
            }
        }
        drop(output);
        if cfg!(target_endian = "big") && width > 1 {
            array.call_method1("byteswap", (true,))?;
        }
        Ok(array.unbind())
    }

    fn missing_field(&self, py: Python<'_>, field: &FieldWire) -> PyResult<Py<PyAny>> {
        if field.required {
            return Err(
                self.invalid_error(py, format!("required field {:?} is absent", field.name))
            );
        }
        Ok(py.None())
    }

    fn decode_struct_at<'py>(
        &self,
        py: Python<'py>,
        object: &ObjectWire,
        model_type: &Bound<'py, PyType>,
        context: DecodeContext<'_, 'py>,
        data: &[u8],
        offset: usize,
    ) -> PyResult<Py<PyAny>> {
        Materializer::new(self, py, data, context).decode_struct(
            object.index,
            model_type.clone().unbind(),
            offset,
        )
    }

    fn decode_table_at<'py>(
        &self,
        py: Python<'py>,
        object: &ObjectWire,
        model_type: &Bound<'py, PyType>,
        context: DecodeContext<'_, 'py>,
        data: &[u8],
        table_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        Materializer::new(self, py, data, context).decode_table(
            object.index,
            model_type.clone().unbind(),
            table_offset,
        )
    }

    fn decode_dynamic<'py>(
        &self,
        py: Python<'py>,
        field: &FieldWire,
        dynamic_overrides: Option<&Bound<'py, PyAny>>,
        data: &[u8],
        table: TableInfo,
    ) -> PyResult<Py<PyAny>> {
        let type_slot = field
            .type_slot
            .ok_or_else(|| PyValueError::new_err("dynamic field has no type slot"))?;
        let type_position =
            self.field_position(py, data, table, usize::from(vtable_offset(type_slot)), 4)?;
        let payload = self.vector_info(py, data, table, field.offset, 1)?;
        let (tag, start, length) = match (type_position, payload) {
            (None, None) => return self.missing_field(py, field),
            (None, Some(_)) => {
                return Err(self.invalid_error(
                    py,
                    format!("dynamic field {:?} has data without a type tag", field.name),
                ));
            }
            (Some(_), None) => {
                return Err(self.invalid_error(
                    py,
                    format!("dynamic field {:?} type tag has no data", field.name),
                ));
            }
            (Some(position), Some((start, length))) => {
                let tag = self.decode_string_at(py, data, position)?;
                (tag, start, length)
            }
        };
        let tag = tag.bind(py).cast::<PyString>()?;
        let tag_value = tag.to_str()?;
        let prefix = field
            .allowed_prefix
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("dynamic field has no allowed prefix"))?;
        if !tag_value.starts_with(prefix) || tag_value.len() == prefix.len() {
            return Err(PyValueError::new_err(format!(
                "dynamic FlatBuffer tag {tag_value:?} is outside {:?}",
                format!("{prefix}*")
            )));
        }
        let wrapper_name = field
            .dynamic_type
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("dynamic field has no wrapper type"))?;
        let wrapper = self.bound_type(py, wrapper_name)?;
        let bytes = PyBytes::new(py, &data[start..start + length]);
        let entry = self
            .dynamic_registry
            .bind(py)
            .call_method1("lookup_tag", (tag,))?;
        if entry.is_none() {
            return Ok(wrapper.call_method1("opaque", (tag, bytes))?.unbind());
        }
        let generated_type = entry.getattr("model_type")?;
        let model = match dynamic_overrides {
            Some(overrides) => {
                let model_type =
                    overrides.call_method1("get", (generated_type.clone(), generated_type))?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("dynamic_overrides", overrides)?;
                model_type.call_method("from_flatbuffer", (bytes,), Some(&kwargs))?
            }
            None => generated_type.call_method1("from_flatbuffer", (bytes,))?,
        };
        Ok(wrapper.call1((model,))?.unbind())
    }

    fn decode_root<'py>(
        &self,
        py: Python<'py>,
        object: &ObjectWire,
        model_type: &Bound<'py, PyType>,
        context: DecodeContext<'_, 'py>,
        input: &[u8],
        options: RootDecode<'_>,
    ) -> PyResult<Py<PyAny>> {
        let (data, root_offset) = if options.size_prefixed {
            let size = usize::try_from(self.read_u32(py, input, options.offset)?).unwrap();
            let start = options
                .offset
                .checked_add(4)
                .ok_or_else(|| self.bounds_error(py, "size-prefixed payload offset overflows"))?;
            self.require_span(py, input, start, size, "size-prefixed buffer")?;
            (&input[start..start + size], 0)
        } else {
            (input, options.offset)
        };
        self.require_span(py, data, root_offset, 4, "root offset")?;
        if options.check_identifier
            && let Some(expected) = options.identifier
        {
            let position = root_offset + 4;
            self.require_span(py, data, position, 4, "file identifier")?;
            if &data[position..position + 4] != expected.as_bytes() {
                return Err(self.invalid_error(
                    py,
                    format!(
                        "expected file identifier {:?}, got {:?}",
                        expected.as_bytes(),
                        &data[position..position + 4]
                    ),
                ));
            }
        }
        let relative = usize::try_from(self.read_u32(py, data, root_offset)?).unwrap();
        if relative == 0 {
            return Err(self.invalid_error(py, "root table offset is null"));
        }
        let table_offset = root_offset
            .checked_add(relative)
            .ok_or_else(|| self.bounds_error(py, "root table offset overflows"))?;
        self.decode_table_at(py, object, model_type, context, data, table_offset)
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
        let value_type = value.get_type();
        let type_name = value_type.name()?.to_str()?.to_owned();
        let target = self.model_object_index(value).map_err(|_| {
            PyTypeError::new_err(format!(
                "union field {:?} does not accept model type {}",
                field.name, type_name,
            ))
        })?;
        let arm = field.arms.iter().find(|arm| arm.target_index == target);
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
        self.require_model_type(object, model)?;
        buffer.fill(0);
        for field in &object.fields {
            let value = model.getattr(field.name.as_str())?;
            match field.kind {
                FieldKind::Scalar => write_struct_scalar(
                    buffer,
                    field.offset,
                    field.scalar.ok_or_else(|| {
                        PyValueError::new_err("native struct field has no scalar kind")
                    })?,
                    &value,
                )?,
                FieldKind::Struct => {
                    let target = self.target_object(field)?;
                    let end = field.offset + target.byte_size;
                    let output = buffer.get_mut(field.offset..end).ok_or_else(|| {
                        PyValueError::new_err("nested struct lies outside its parent")
                    })?;
                    self.encode_struct_into(target, &value, output)?;
                }
                FieldKind::ArrayScalar => write_struct_scalar_array(
                    buffer,
                    field.offset,
                    field.fixed_length,
                    field.scalar.ok_or_else(|| {
                        PyValueError::new_err("native scalar array has no scalar kind")
                    })?,
                    &value,
                )?,
                FieldKind::ArrayStruct => {
                    let target = self.target_object(field)?;
                    let length = sequence_len(&value)?;
                    require_fixed_array_length(field.fixed_length, length)?;
                    for index in 0..length {
                        let start = field.offset + index * field.element_size;
                        let end = start + target.byte_size;
                        let output = buffer.get_mut(start..end).ok_or_else(|| {
                            PyValueError::new_err("struct array lies outside its parent")
                        })?;
                        self.encode_struct_into(target, &sequence_item(&value, index)?, output)?;
                    }
                }
                _ => {
                    return Err(PyNotImplementedError::new_err(
                        "unsupported native struct field",
                    ));
                }
            }
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
        self.require_model_type(object, model)?;
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

            include_slot(&mut highest_slot, field.slot);
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
                        include_slot(&mut highest_slot, type_slot);
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
                        include_slot(&mut highest_slot, type_slot);
                    }
                    let width = scalar_size(field.type_scalar)?;
                    size = size.saturating_add(aligned_payload_size(width, width));
                    let arm = self.union_arm(field, value)?;
                    self.estimate_table(&self.objects[arm.target_index], value)?
                }
                FieldKind::UnionVector => {
                    size = size.saturating_add(7);
                    if let Some(type_slot) = field.type_slot {
                        include_slot(&mut highest_slot, type_slot);
                        size = size.saturating_add(7);
                    }
                    let length = sequence_len(value)?;
                    let width = scalar_size(field.type_scalar)?;
                    let tables = sampled_sequence_size(value, |item| {
                        if item.is_none() {
                            return Ok(0);
                        }
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
                FieldKind::ArrayScalar | FieldKind::ArrayStruct => {
                    return Err(PyValueError::new_err(
                        "fixed arrays may only be fields of structs",
                    ));
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
                        if item.is_none() {
                            tags.push(0);
                            values.push(NullableOffset(None));
                            return Ok(());
                        }
                        let arm = self.union_arm(field, &item)?;
                        tags.push(arm.tag as u8);
                        let offset = self.build_table(
                            builder,
                            &self.objects[arm.target_index],
                            &item,
                            None,
                            vector_length_patches,
                        )?;
                        values.push(NullableOffset(Some(offset.value())));
                        Ok(())
                    })?;
                    discriminator = Some(Discriminator::Offset(erase_offset(
                        builder.create_vector(&tags),
                    )));
                    erase_offset(builder.create_vector(&values))
                }
                FieldKind::Scalar
                | FieldKind::Struct
                | FieldKind::ArrayScalar
                | FieldKind::ArrayStruct => unreachable!(),
            };
            states[index].offset = Some(offset);
            states[index].discriminator = discriminator;
        }

        let table = builder.start_table();
        for (field, state) in object.fields.iter().zip(&states) {
            let slot = vtable_offset(field.slot);
            if let Some(offset) = state.offset {
                builder.push_slot_always(slot, offset);
                match field.kind {
                    FieldKind::Dynamic => {
                        let Some(Discriminator::Offset(type_offset)) = state.discriminator else {
                            return Err(PyTypeError::new_err("dynamic FlatBuffer has no type tag"));
                        };
                        let type_slot = vtable_offset(field.type_slot.ok_or_else(|| {
                            PyValueError::new_err("dynamic field has no type slot")
                        })?);
                        builder.push_slot_always(type_slot, type_offset);
                    }
                    FieldKind::Union => {
                        let Some(Discriminator::Union(tag)) = state.discriminator else {
                            return Err(PyTypeError::new_err("union has no discriminator"));
                        };
                        let type_slot = vtable_offset(
                            field
                                .type_slot
                                .ok_or_else(|| PyValueError::new_err("union has no type slot"))?,
                        );
                        push_type_id(builder, type_slot, field.type_scalar, tag)?;
                    }
                    FieldKind::UnionVector => {
                        let Some(Discriminator::Offset(type_offset)) = state.discriminator else {
                            return Err(PyTypeError::new_err(
                                "union vector has no discriminator vector",
                            ));
                        };
                        let type_slot = vtable_offset(field.type_slot.ok_or_else(|| {
                            PyValueError::new_err("union vector has no type slot")
                        })?);
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
                FieldKind::ArrayScalar | FieldKind::ArrayStruct => {
                    return Err(PyValueError::new_err(
                        "fixed arrays may only be fields of structs",
                    ));
                }
            }
        }
        Ok(builder.end_table(table))
    }
}

impl<'plan, 'data, 'context, 'py> Materializer<'plan, 'data, 'context, 'py> {
    fn new(
        plan: &'plan NativePlan,
        py: Python<'py>,
        data: &'data [u8],
        context: DecodeContext<'context, 'py>,
    ) -> Self {
        Self {
            plan,
            py,
            data,
            context,
            frames: Vec::new(),
            values: Vec::new(),
            model_types: Vec::new(),
            model_type_indices: HashMap::new(),
            decoded_objects: 0,
        }
    }

    fn decode_table(
        mut self,
        object_index: usize,
        model_type: Py<PyType>,
        offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let model_type_index = self.intern_model_type(model_type);
        self.push_table(object_index, model_type_index, offset)?;
        self.run()
    }

    fn decode_struct(
        mut self,
        object_index: usize,
        model_type: Py<PyType>,
        offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let model_type_index = self.intern_model_type(model_type);
        self.push_struct(object_index, model_type_index, offset)?;
        self.run()
    }

    fn intern_model_type(&mut self, model_type: Py<PyType>) -> usize {
        let pointer = model_type.bind(self.py).as_ptr() as usize;
        if let Some(&index) = self.model_type_indices.get(&pointer) {
            return index;
        }
        let index = self.model_types.len();
        self.model_types.push(model_type);
        self.model_type_indices.insert(pointer, index);
        index
    }

    fn child_model_type(&mut self, task: FieldTask, target_index: usize) -> PyResult<usize> {
        let parent_type = self.model_types[task.model_type_index].bind(self.py);
        let field = self.field(task);
        let target = &self.plan.objects[target_index];
        let model_type = self.plan.child_decode_type(
            self.py,
            parent_type,
            field,
            target,
            self.context.model_types,
        )?;
        Ok(self.intern_model_type(model_type.unbind()))
    }

    fn field(&self, task: FieldTask) -> &FieldWire {
        &self.plan.objects[task.object_index].fields[task.field_index]
    }

    #[inline]
    fn target_index(&self, task: FieldTask, description: &str) -> PyResult<usize> {
        self.field(task)
            .target_index
            .ok_or_else(|| PyValueError::new_err(format!("native {description} has no target")))
    }

    #[inline]
    fn union_target(&self, field: &FieldWire, tag: u64) -> PyResult<usize> {
        field
            .arms
            .iter()
            .find(|arm| arm.tag == tag)
            .map(|arm| arm.target_index)
            .ok_or_else(|| {
                self.plan
                    .invalid_error(self.py, format!("unknown union discriminator {tag}"))
            })
    }

    fn count_object(&mut self) -> PyResult<()> {
        if self.decoded_objects == MAX_DECODED_OBJECTS {
            return Err(self.plan.invalid_error(
                self.py,
                format!("FlatBuffer contains more than {MAX_DECODED_OBJECTS} decoded objects"),
            ));
        }
        self.decoded_objects += 1;
        Ok(())
    }

    fn push_table(
        &mut self,
        object_index: usize,
        model_type_index: usize,
        offset: usize,
    ) -> PyResult<()> {
        let object = &self.plan.objects[object_index];
        if object.is_struct {
            return Err(PyTypeError::new_err(format!(
                "{} is a struct, not a table",
                object.name
            )));
        }
        let table = self.plan.table_info(self.py, self.data, offset)?;
        self.push_object(object_index, model_type_index, ObjectLocation::Table(table))
    }

    fn push_struct(
        &mut self,
        object_index: usize,
        model_type_index: usize,
        offset: usize,
    ) -> PyResult<()> {
        let object = &self.plan.objects[object_index];
        if !object.is_struct {
            return Err(PyTypeError::new_err(format!(
                "{} is a table, not a struct",
                object.name
            )));
        }
        self.plan
            .require_span(self.py, self.data, offset, object.byte_size, "struct data")?;
        self.push_object(
            object_index,
            model_type_index,
            ObjectLocation::Struct { offset },
        )
    }

    fn push_object(
        &mut self,
        object_index: usize,
        model_type_index: usize,
        location: ObjectLocation,
    ) -> PyResult<()> {
        self.count_object()?;
        self.frames.push(DecodeFrame::Object(ObjectFrame {
            next_field: FieldTask {
                object_index,
                model_type_index,
                field_index: 0,
            },
            location,
            value_start: self.values.len(),
        }));
        Ok(())
    }

    fn finish_model(&mut self, task: FieldTask, value_start: usize) -> PyResult<()> {
        let object = &self.plan.objects[task.object_index];
        if self.values.len() - value_start != object.fields.len() {
            return Err(PyRuntimeError::new_err(
                "native materializer produced the wrong field count",
            ));
        }
        let kwargs = PyDict::new(self.py);
        for (field, value) in object.fields.iter().zip(&self.values[value_start..]) {
            kwargs.set_item(field.name.as_str(), value.bind(self.py))?;
        }
        let model = self.model_types[task.model_type_index]
            .bind(self.py)
            .call((), Some(&kwargs))?
            .unbind();
        self.values.truncate(value_start);
        self.values.push(model);
        Ok(())
    }

    fn finish_list(&mut self, value_start: usize) -> PyResult<()> {
        let value = PyList::new(self.py, &self.values[value_start..])?
            .into_any()
            .unbind();
        self.values.truncate(value_start);
        self.values.push(value);
        Ok(())
    }

    fn decode_scalar_sequence(
        &self,
        field: &FieldWire,
        start: usize,
        length: usize,
        scalar: ScalarKind,
    ) -> PyResult<Py<PyAny>> {
        if field.enum_type.is_none() {
            return self
                .plan
                .decode_numeric_vector(self.py, self.data, start, length, scalar);
        }
        let width = scalar_size(Some(scalar))?;
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let value = self
                .plan
                .read_scalar_value(self.py, self.data, start + index * width, scalar)?
                .into_py(self.py)?;
            values.push(self.plan.apply_enum(self.py, field, value)?);
        }
        Ok(PyList::new(self.py, values)?.into_any().unbind())
    }

    fn missing_field(&mut self, task: FieldTask) -> PyResult<()> {
        self.values
            .push(self.plan.missing_field(self.py, self.field(task))?);
        Ok(())
    }

    #[inline]
    fn position_or_missing(
        &mut self,
        task: FieldTask,
        table: TableInfo,
        size: usize,
    ) -> PyResult<Option<usize>> {
        let offset = self.field(task).offset;
        let position = self
            .plan
            .field_position(self.py, self.data, table, offset, size)?;
        if position.is_none() {
            self.missing_field(task)?;
        }
        Ok(position)
    }

    #[inline]
    fn vector_or_missing(
        &mut self,
        task: FieldTask,
        table: TableInfo,
        item_size: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let offset = self.field(task).offset;
        let vector = self
            .plan
            .vector_info(self.py, self.data, table, offset, item_size)?;
        if vector.is_none() {
            self.missing_field(task)?;
        }
        Ok(vector)
    }

    fn run(mut self) -> PyResult<Py<PyAny>> {
        while let Some(frame) = self.frames.pop() {
            match frame {
                DecodeFrame::Object(mut object) => {
                    let task = object.next_field;
                    if task.field_index == self.plan.objects[task.object_index].fields.len() {
                        self.finish_model(task, object.value_start)?;
                        continue;
                    }
                    let location = object.location;
                    object.next_field.field_index += 1;
                    self.frames.push(DecodeFrame::Object(object));
                    match location {
                        ObjectLocation::Table(table) => self.decode_table_field(task, table)?,
                        ObjectLocation::Struct { offset } => {
                            self.decode_struct_field(task, offset)?
                        }
                    }
                }
                DecodeFrame::ObjectVector(mut vector) => {
                    if vector.index == vector.length {
                        self.finish_list(vector.result_start)?;
                        continue;
                    }
                    let index = vector.index;
                    vector.index += 1;
                    self.frames.push(DecodeFrame::ObjectVector(vector));
                    match vector.element {
                        VectorElement::Table => {
                            let target = self.plan.offset_target(
                                self.py,
                                self.data,
                                vector.start + index * 4,
                                "table vector offset",
                            )?;
                            self.push_table(vector.object_index, vector.model_type_index, target)?;
                        }
                        VectorElement::Struct { stride } => self.push_struct(
                            vector.object_index,
                            vector.model_type_index,
                            vector.start + index * stride,
                        )?,
                    }
                }
                DecodeFrame::UnionVector(mut vector) => {
                    if vector.index == vector.length {
                        self.finish_list(vector.result_start)?;
                        continue;
                    }
                    let index = vector.index;
                    vector.index += 1;
                    self.frames.push(DecodeFrame::UnionVector(vector));
                    let tag = self
                        .plan
                        .read_scalar_value(
                            self.py,
                            self.data,
                            vector.type_start + index * vector.width,
                            vector.type_scalar,
                        )?
                        .as_u64()?;
                    let position = vector.value_start + index * 4;
                    if tag == 0 {
                        if self.plan.read_u32(self.py, self.data, position)? != 0 {
                            return Err(self
                                .plan
                                .invalid_error(self.py, "union vector NONE has a payload"));
                        }
                        self.values.push(self.py.None());
                        continue;
                    }
                    let field = self.field(vector.field);
                    let target_index = self.union_target(field, tag)?;
                    let target = self.plan.offset_target(
                        self.py,
                        self.data,
                        position,
                        "union vector offset",
                    )?;
                    let model_type_index = self.child_model_type(vector.field, target_index)?;
                    self.push_table(target_index, model_type_index, target)?;
                }
            }
        }
        if self.values.len() != 1 {
            return Err(PyRuntimeError::new_err(
                "native materializer did not produce one root model",
            ));
        }
        Ok(self.values.pop().expect("checked materializer result"))
    }

    fn decode_struct_field(&mut self, task: FieldTask, offset: usize) -> PyResult<()> {
        let field = self.field(task);
        match field.kind {
            FieldKind::Scalar => {
                let scalar = field.scalar.ok_or_else(|| {
                    PyValueError::new_err("native struct field has no scalar kind")
                })?;
                let value = self
                    .plan
                    .read_scalar_value(self.py, self.data, offset + field.offset, scalar)?
                    .into_py(self.py)?;
                self.values
                    .push(self.plan.apply_enum(self.py, field, value)?);
            }
            FieldKind::Struct => {
                let target_index = self.target_index(task, "struct field")?;
                let target_offset = offset.checked_add(field.offset).ok_or_else(|| {
                    self.plan
                        .bounds_error(self.py, "struct field offset overflows")
                })?;
                let child_type = self.child_model_type(task, target_index)?;
                self.push_struct(target_index, child_type, target_offset)?;
            }
            FieldKind::ArrayScalar => {
                let scalar = field.scalar.ok_or_else(|| {
                    PyValueError::new_err("native scalar array has no scalar kind")
                })?;
                let start = offset + field.offset;
                self.values.push(self.decode_scalar_sequence(
                    field,
                    start,
                    field.fixed_length,
                    scalar,
                )?);
            }
            FieldKind::ArrayStruct => {
                let target_index = self.target_index(task, "struct array")?;
                let start = offset + field.offset;
                let stride = field.element_size;
                let length = field.fixed_length;
                let child_type = self.child_model_type(task, target_index)?;
                self.frames
                    .push(DecodeFrame::ObjectVector(ObjectVectorFrame {
                        object_index: target_index,
                        model_type_index: child_type,
                        element: VectorElement::Struct { stride },
                        start,
                        length,
                        index: 0,
                        result_start: self.values.len(),
                    }));
            }
            _ => {
                return Err(PyNotImplementedError::new_err(
                    "unsupported native struct field",
                ));
            }
        }
        Ok(())
    }

    fn decode_table_field(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        match self.field(task).kind {
            FieldKind::Scalar => {
                let field = self.field(task);
                let scalar = field.scalar.ok_or_else(|| {
                    PyValueError::new_err("native scalar field has no scalar kind")
                })?;
                let position = self.plan.field_position(
                    self.py,
                    self.data,
                    table,
                    field.offset,
                    scalar_size(Some(scalar))?,
                )?;
                let value = match position {
                    Some(position) => self
                        .plan
                        .read_scalar_value(self.py, self.data, position, scalar)?,
                    None if field.optional => {
                        self.values.push(self.py.None());
                        return Ok(());
                    }
                    None => ScalarValue::from_default(scalar, &field.default)?,
                };
                let value = value.into_py(self.py)?;
                self.values
                    .push(self.plan.apply_enum(self.py, field, value)?);
            }
            FieldKind::String => {
                let Some(position) = self.position_or_missing(task, table, 4)? else {
                    return Ok(());
                };
                self.values
                    .push(self.plan.decode_string_at(self.py, self.data, position)?);
            }
            FieldKind::Table => {
                let Some(position) = self.position_or_missing(task, table, 4)? else {
                    return Ok(());
                };
                let target_index = self.target_index(task, "table field")?;
                let target =
                    self.plan
                        .offset_target(self.py, self.data, position, "table field offset")?;
                let child_type = self.child_model_type(task, target_index)?;
                self.push_table(target_index, child_type, target)?;
            }
            FieldKind::Struct => {
                let target_index = self.target_index(task, "struct field")?;
                let target_size = self.plan.objects[target_index].byte_size;
                let Some(position) = self.position_or_missing(task, table, target_size)? else {
                    return Ok(());
                };
                let child_type = self.child_model_type(task, target_index)?;
                self.push_struct(target_index, child_type, position)?;
            }
            FieldKind::VectorByte => {
                let Some((start, length)) = self.vector_or_missing(task, table, 1)? else {
                    return Ok(());
                };
                self.values.push(
                    PyBytes::new(self.py, &self.data[start..start + length])
                        .into_any()
                        .unbind(),
                );
            }
            FieldKind::VectorScalar => {
                self.decode_scalar_vector(task, table)?;
            }
            FieldKind::VectorString => {
                let Some((start, length)) = self.vector_or_missing(task, table, 4)? else {
                    return Ok(());
                };
                let mut values = Vec::with_capacity(length);
                for index in 0..length {
                    values.push(self.plan.decode_string_at(
                        self.py,
                        self.data,
                        start + index * 4,
                    )?);
                }
                self.values
                    .push(PyList::new(self.py, values)?.into_any().unbind());
            }
            FieldKind::VectorTable | FieldKind::VectorStruct => {
                self.push_object_vector(task, table)?;
            }
            FieldKind::Nested => {
                self.decode_nested(task, table)?;
            }
            FieldKind::Dynamic => {
                let field = self.field(task);
                let value = self.plan.decode_dynamic(
                    self.py,
                    field,
                    self.context.dynamic_overrides,
                    self.data,
                    table,
                )?;
                self.values.push(value);
            }
            FieldKind::Union => {
                self.decode_union_field(task, table)?;
            }
            FieldKind::UnionVector => {
                self.push_union_vector(task, table)?;
            }
            FieldKind::ArrayScalar | FieldKind::ArrayStruct => {
                return Err(PyValueError::new_err(
                    "fixed arrays may only be fields of structs",
                ));
            }
        }
        Ok(())
    }

    fn decode_scalar_vector(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let scalar = self
            .field(task)
            .scalar
            .ok_or_else(|| PyValueError::new_err("native scalar vector has no scalar kind"))?;
        let width = scalar_size(Some(scalar))?;
        let Some((start, length)) = self.vector_or_missing(task, table, width)? else {
            return Ok(());
        };
        let field = self.field(task);
        self.values
            .push(self.decode_scalar_sequence(field, start, length, scalar)?);
        Ok(())
    }

    fn push_object_vector(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let field = self.field(task);
        let target_index = self.target_index(task, "object vector")?;
        let (element, item_size) = match field.kind {
            FieldKind::VectorTable => (VectorElement::Table, 4),
            FieldKind::VectorStruct => {
                let stride = self.plan.objects[target_index].byte_size;
                (VectorElement::Struct { stride }, stride)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "native field is not an object vector",
                ));
            }
        };
        let Some((start, length)) = self.vector_or_missing(task, table, item_size)? else {
            return Ok(());
        };
        let child_type = self.child_model_type(task, target_index)?;
        self.frames
            .push(DecodeFrame::ObjectVector(ObjectVectorFrame {
                object_index: target_index,
                model_type_index: child_type,
                element,
                start,
                length,
                index: 0,
                result_start: self.values.len(),
            }));
        Ok(())
    }

    fn decode_nested(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let Some((start, length)) = self.vector_or_missing(task, table, 1)? else {
            return Ok(());
        };
        let target_index = self.target_index(task, "nested field")?;
        let child_type = self.child_model_type(task, target_index)?;
        let model_type = self.model_types[child_type].bind(self.py);
        let bytes = PyBytes::new(self.py, &self.data[start..start + length]);
        let value = match self.context.dynamic_overrides {
            Some(overrides) => {
                let kwargs = PyDict::new(self.py);
                kwargs.set_item("dynamic_overrides", overrides)?;
                model_type.call_method("from_flatbuffer", (bytes,), Some(&kwargs))?
            }
            None => model_type.call_method1("from_flatbuffer", (bytes,))?,
        };
        self.values.push(value.unbind());
        Ok(())
    }

    fn decode_union_field(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let field = self.field(task);
        let type_slot = field
            .type_slot
            .ok_or_else(|| PyValueError::new_err("union field has no type slot"))?;
        let type_scalar = field
            .type_scalar
            .ok_or_else(|| PyValueError::new_err("union field has no type scalar"))?;
        let type_position = self.plan.field_position(
            self.py,
            self.data,
            table,
            usize::from(vtable_offset(type_slot)),
            scalar_size(Some(type_scalar))?,
        )?;
        let tag = match type_position {
            Some(position) => self
                .plan
                .read_scalar_value(self.py, self.data, position, type_scalar)?
                .as_u64()?,
            None => 0,
        };
        let value_position =
            self.plan
                .field_position(self.py, self.data, table, field.offset, 4)?;
        if tag == 0 {
            if value_position.is_some() {
                return Err(self
                    .plan
                    .invalid_error(self.py, "union NONE discriminator has a payload"));
            }
            return self.missing_field(task);
        }
        let position = value_position.ok_or_else(|| {
            self.plan
                .invalid_error(self.py, format!("union discriminator {tag} has no payload"))
        })?;
        let target_index = self.union_target(field, tag)?;
        let target = self
            .plan
            .offset_target(self.py, self.data, position, "union field offset")?;
        let child_type = self.child_model_type(task, target_index)?;
        self.push_table(target_index, child_type, target)
    }

    fn push_union_vector(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let field = self.field(task);
        let type_slot = field
            .type_slot
            .ok_or_else(|| PyValueError::new_err("union vector has no type slot"))?;
        let type_scalar = field
            .type_scalar
            .ok_or_else(|| PyValueError::new_err("union vector has no type scalar"))?;
        let width = scalar_size(Some(type_scalar))?;
        let types = self.plan.vector_info(
            self.py,
            self.data,
            table,
            usize::from(vtable_offset(type_slot)),
            width,
        )?;
        let values = self
            .plan
            .vector_info(self.py, self.data, table, field.offset, 4)?;
        let (type_start, type_length, value_start, value_length) = match (types, values) {
            (None, None) => return self.missing_field(task),
            (Some((type_start, type_length)), Some((value_start, value_length))) => {
                (type_start, type_length, value_start, value_length)
            }
            _ => {
                return Err(self.plan.invalid_error(
                    self.py,
                    format!("union vector {:?} type/value presence differs", field.name),
                ));
            }
        };
        if type_length != value_length {
            return Err(self.plan.invalid_error(
                self.py,
                format!("union vector {:?} type/value lengths differ", field.name),
            ));
        }
        self.frames.push(DecodeFrame::UnionVector(UnionVectorFrame {
            field: task,
            type_scalar,
            type_start,
            value_start,
            length: value_length,
            width,
            index: 0,
            result_start: self.values.len(),
        }));
        Ok(())
    }
}

#[pymethods]
impl NativePlan {
    #[new]
    fn new(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut wire: ModuleWire = rmp_serde::from_slice(data.as_bytes())
            .map_err(|error| PyValueError::new_err(format!("invalid native plan: {error}")))?;
        if wire.version != 1 {
            return Err(PyValueError::new_err(format!(
                "unsupported native plan version {}",
                wire.version
            )));
        }
        let by_name: HashMap<_, _> = wire
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| (object.name.clone(), index))
            .collect();
        for (object_index, object) in wire.objects.iter_mut().enumerate() {
            object.index = object_index;
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
        let dynamic = data.py().import("msgspec_flatbuffers._dynamic")?;
        let dynamic_encoder = if has_dynamic {
            Some(dynamic.getattr("encode_dynamic")?.unbind())
        } else {
            None
        };
        let dynamic_registry = dynamic.getattr("dynamic_types")?.unbind();
        let numpy_empty = data.py().import("numpy")?.getattr("empty")?.unbind();
        let runtime = data.py().import("msgspec_flatbuffers._runtime")?;
        let buffer_bounds_error = runtime
            .getattr("BufferBoundsError")?
            .cast_into::<PyType>()?
            .unbind();
        let invalid_buffer_error = runtime
            .getattr("InvalidBufferError")?
            .cast_into::<PyType>()?
            .unbind();
        Ok(Self {
            identity: Arc::new(()),
            objects: wire.objects,
            by_name,
            bound_types: OnceLock::new(),
            model_subclass_cache: Mutex::new(HashMap::new()),
            dynamic_encoder,
            dynamic_registry,
            numpy_empty,
            buffer_bounds_error,
            invalid_buffer_error,
        })
    }

    fn bind_types(&self, types: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut by_pointer = HashMap::with_capacity(types.len());
        let mut by_name = HashMap::with_capacity(types.len());
        for (name, bound_type) in types.iter() {
            let name = name.extract::<String>()?;
            let bound_type = bound_type.cast::<PyType>()?;
            if let Some(object_index) = self.by_name.get(&name) {
                by_pointer.insert(bound_type.as_ptr() as usize, *object_index);
            }
            by_name.insert(name, bound_type.clone().unbind());
        }
        self.bound_types
            .set(BoundTypes {
                by_pointer,
                by_name,
            })
            .map_err(|_| PyRuntimeError::new_err("native model types are already bound"))
    }

    fn model_types(
        &self,
        generated: &Bound<'_, PyType>,
        requested: &Bound<'_, PyType>,
        bindings: &Bound<'_, PyAny>,
    ) -> PyResult<NativeModelTypes> {
        let bound_types = self
            .bound_types
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
        let bound_index = |model_type: &Bound<'_, PyType>, role: &str| {
            bound_types
                .by_pointer
                .get(&(model_type.as_ptr() as usize))
                .copied()
                .ok_or_else(|| PyTypeError::new_err(format!("{role} type is not bound")))
        };
        let root_index = bound_index(generated, "generated root")?;
        if !requested.is_subclass(generated)? {
            return Err(PyTypeError::new_err(format!(
                "{} must subclass {}",
                requested.name()?,
                generated.name()?,
            )));
        }

        let mut child_types = HashMap::new();
        for binding in bindings.try_iter()? {
            let binding = binding?.cast_into::<PyTuple>()?;
            if binding.len() != 5 {
                return Err(PyValueError::new_err(
                    "native model type bindings must contain five values",
                ));
            }
            let generated_parent = binding.get_item(0)?.cast_into::<PyType>()?;
            let requested_parent = binding.get_item(1)?.cast_into::<PyType>()?;
            let field_name = binding.get_item(2)?.extract::<String>()?;
            let generated_child = binding.get_item(3)?.cast_into::<PyType>()?;
            let requested_child = binding.get_item(4)?.cast_into::<PyType>()?;

            let parent_index = bound_index(&generated_parent, "generated parent")?;
            let child_index = bound_index(&generated_child, "generated child")?;
            if !requested_parent.is_subclass(&generated_parent)?
                || !requested_child.is_subclass(&generated_child)?
            {
                return Err(PyTypeError::new_err(
                    "native model type binding is not a subclass",
                ));
            }
            let parent = &self.objects[parent_index];
            let field = parent
                .fields
                .iter()
                .find(|field| field.name == field_name)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "native model field {:?}.{:?} does not exist",
                        parent.name, field_name,
                    ))
                })?;
            let accepts_child = field.target_index == Some(child_index)
                || field.arms.iter().any(|arm| arm.target_index == child_index);
            if !accepts_child {
                return Err(PyTypeError::new_err(format!(
                    "native model field {:?}.{:?} does not contain {}",
                    parent.name, field_name, self.objects[child_index].name,
                )));
            }
            let key = (requested_parent.as_ptr() as usize, field.slot, child_index);
            if let Some(existing) = child_types.insert(key, requested_child.clone().unbind())
                && existing.bind(requested.py()).as_ptr() != requested_child.as_ptr()
            {
                return Err(PyTypeError::new_err(
                    "native model field has conflicting subclass bindings",
                ));
            }
        }
        Ok(NativeModelTypes {
            plan_identity: Arc::clone(&self.identity),
            root_index,
            root_type: requested.clone().unbind(),
            child_types,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (root, buffer, *, identifier=None, offset=0, size_prefixed=false, check_identifier=true, model_types=None, dynamic_overrides=None))]
    fn unpack(
        &self,
        py: Python<'_>,
        root: &str,
        buffer: &Bound<'_, PyAny>,
        identifier: Option<&str>,
        offset: isize,
        size_prefixed: bool,
        check_identifier: bool,
        model_types: Option<PyRef<'_, NativeModelTypes>>,
        dynamic_overrides: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let offset = nonnegative_usize(offset, "offset")?;
        validate_identifier(identifier)?;
        let root = self.object(root)?;
        let model_types = model_types.as_deref();
        let (model_type, context) = self.prepare_decode(
            py,
            root,
            model_types,
            dynamic_overrides,
            "native model types belong to a different root",
        )?;
        let options = RootDecode {
            offset,
            size_prefixed,
            identifier,
            check_identifier,
        };
        with_input_bytes(buffer, |data| {
            self.decode_root(py, root, &model_type, context, data, options)
        })
    }

    #[pyo3(signature = (object, buffer, offset, *, model_types=None, dynamic_overrides=None))]
    fn unpack_view(
        &self,
        py: Python<'_>,
        object: &str,
        buffer: &Bound<'_, PyAny>,
        offset: isize,
        model_types: Option<PyRef<'_, NativeModelTypes>>,
        dynamic_overrides: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let offset = nonnegative_usize(offset, "offset")?;
        let object = self.object(object)?;
        let model_types = model_types.as_deref();
        let (model_type, context) = self.prepare_decode(
            py,
            object,
            model_types,
            dynamic_overrides,
            "native model types belong to a different view type",
        )?;
        with_input_bytes(buffer, |data| {
            if object.is_struct {
                self.decode_struct_at(py, object, &model_type, context, data, offset)
            } else {
                self.decode_table_at(py, object, &model_type, context, data, offset)
            }
        })
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
        let initial_size = nonnegative_usize(initial_size, "initial_size")?;
        validate_identifier(identifier)?;
        let root_object = self.object(root)?;
        let mut root_fields = self.load_fields(root_object, model)?;
        let finish_size = 4usize
            .saturating_add(identifier.map_or(0, |_| 4))
            .saturating_add(if size_prefixed { 4 } else { 0 });
        let estimate = self
            .estimate_table_fields(root_object, &mut root_fields)?
            .saturating_add(finish_size);
        let estimate = estimate.saturating_mul(101).saturating_add(99) / 100;
        let capacity = initial_size.max(estimate);
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
    module.add_class::<NativeModelTypes>()?;
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
