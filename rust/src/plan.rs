use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use bytemuck::{Pod, Zeroable, cast_slice, cast_slice_mut};
use flatbuffers::{
    FLATBUFFERS_MAX_BUFFER_SIZE, FlatBufferBuilder, Push, PushAlignment, TableFinishedWIPOffset,
    UnionWIPOffset, VOffsetT, WIPOffset,
};
use numpy::PyArrayMethods;
use pyo3::IntoPyObjectExt;
use pyo3::buffer::{Element, PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::{
    PyBufferError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyUnicodeDecodeError,
    PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{
    PyBytes, PyDict, PyInt, PyList, PyMemoryView, PyModule, PyString, PyTuple, PyType,
};
use rmpv::Value;
use serde::de::{
    DeserializeSeed, Error as DeserializeError, IgnoredAny, MapAccess, SeqAccess, Visitor,
};
use serde::ser::{Error as SerdeError, SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
    #[serde(skip)]
    key_field_index: Option<usize>,
    is_struct: bool,
    byte_size: usize,
    min_alignment: usize,
    fields: Vec<FieldWire>,
    serde_fields: Vec<SerdeFieldWire>,
    serde_tag_field: Option<String>,
    serde_tag: Option<String>,
}

impl ObjectWire {
    fn key_field(&self) -> Option<&FieldWire> {
        self.key_field_index
            .and_then(|index| self.fields.get(index))
    }
}

#[derive(Deserialize)]
struct SerdeFieldWire {
    attr_name: String,
    encode_name: String,
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
    Uuid,
    Decimal,
    Fallback,
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
    serde_nullable: bool,
    #[serde(default)]
    serde_element_nullable: bool,
    #[serde(default)]
    serde_python_int: bool,
    #[serde(default)]
    serde_omit_default: bool,
    #[serde(default)]
    key: bool,
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
    #[serde(default)]
    fallback_id: Option<String>,
}

enum TableKey {
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float32(f32),
    Float64(f64),
    String(String),
}

impl TableKey {
    fn extract(field: &FieldWire, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        match field.kind {
            FieldKind::String => Ok(Self::String(value.extract()?)),
            FieldKind::Scalar => match field
                .scalar
                .ok_or_else(|| PyValueError::new_err("key field has no scalar type"))?
            {
                ScalarKind::Bool => Ok(Self::Bool(value.extract()?)),
                ScalarKind::Int8 | ScalarKind::Int16 | ScalarKind::Int32 | ScalarKind::Int64 => {
                    Ok(Self::Signed(value.extract()?))
                }
                ScalarKind::Uint8
                | ScalarKind::Uint16
                | ScalarKind::Uint32
                | ScalarKind::Uint64 => Ok(Self::Unsigned(value.extract()?)),
                ScalarKind::Float32 => {
                    let value = value.extract::<f32>()?;
                    if value.is_nan() {
                        return Err(PyValueError::new_err("table keys cannot be NaN"));
                    }
                    Ok(Self::Float32(value))
                }
                ScalarKind::Float64 => {
                    let value = value.extract::<f64>()?;
                    if value.is_nan() {
                        return Err(PyValueError::new_err("table keys cannot be NaN"));
                    }
                    Ok(Self::Float64(value))
                }
            },
            _ => Err(PyValueError::new_err(
                "table key fields must be scalar or string fields",
            )),
        }
    }

    fn compare(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left.cmp(right),
            (Self::Signed(left), Self::Signed(right)) => left.cmp(right),
            (Self::Unsigned(left), Self::Unsigned(right)) => left.cmp(right),
            (Self::Float32(left), Self::Float32(right)) => {
                left.partial_cmp(right).expect("NaN keys rejected")
            }
            (Self::Float64(left), Self::Float64(right)) => {
                left.partial_cmp(right).expect("NaN keys rejected")
            }
            (Self::String(left), Self::String(right)) => left.cmp(right),
            _ => unreachable!("one FlatBuffers map has one key type"),
        }
    }
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
        dst[..size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
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
    serde_objects: Vec<Option<SerdeObject>>,
    serde_supported: bool,
}

struct SerdeObject {
    fields: Vec<SerdeField>,
    keyword_names: Py<PyTuple>,
    tag: Option<(String, String)>,
}

struct SerdeField {
    object_field_index: usize,
    attr_name: Py<PyString>,
    encode_name: String,
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
        // SAFETY: PyO3 provides a valid writable Py_buffer pointer. `slf` is
        // transferred to `view.obj`, keeping the immutable Vec allocation alive
        // until CPython releases the exported buffer.
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

const SIZE_ESTIMATE_WORK_LIMIT: usize = 6;
const SIZE_ESTIMATE_SAMPLE_LIMIT: usize = 6;
const SIZE_ESTIMATE_SEQUENCE_WORK_LIMIT: usize = 4;
const DEFAULT_STRING_SIZE_ESTIMATE: usize = 32;
const DEFAULT_TABLE_SIZE_ESTIMATE: usize = 64;

struct SizeEstimateBudget {
    remaining: usize,
}

impl SizeEstimateBudget {
    fn new() -> Self {
        Self {
            remaining: SIZE_ESTIMATE_WORK_LIMIT,
        }
    }

    fn consume(&mut self, work: usize) -> bool {
        if work > self.remaining {
            self.remaining = 0;
            return false;
        }
        self.remaining -= work;
        true
    }

    fn take_samples(&mut self, length: usize, work_per_sample: usize) -> usize {
        let samples = self.available_samples(length, work_per_sample);
        self.remaining -= samples * work_per_sample.max(1);
        samples
    }

    fn available_samples(&self, length: usize, work_per_sample: usize) -> usize {
        let work_per_sample = work_per_sample.max(1);
        let desired = length.min(SIZE_ESTIMATE_SAMPLE_LIMIT);
        let wanted = desired.min(SIZE_ESTIMATE_SEQUENCE_WORK_LIMIT / work_per_sample);
        wanted.min(self.remaining / work_per_sample)
    }
}

fn table_estimate_work(object: &ObjectWire) -> usize {
    object.fields.len().max(1)
}

fn mix_sample_seed(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sample_seed(object: &ObjectWire, field: &FieldWire, length: usize) -> u64 {
    mix_sample_seed(
        (object.index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ ((field.slot as u64) << 32)
            ^ length as u64,
    )
}

fn stratified_sample_index(length: usize, count: usize, index: usize, seed: u64) -> usize {
    if count == 1 {
        return length / 2;
    }
    if index == 0 {
        return 0;
    }
    if index + 1 == count {
        return length - 1;
    }
    let strata = count - 2;
    let width = length - 2;
    let stratum = index - 1;
    let start = 1 + stratum * width / strata;
    let end = 1 + (stratum + 1) * width / strata;
    start + mix_sample_seed(seed ^ index as u64) as usize % (end - start)
}

fn sampled_sequence_size(
    value: &Bound<'_, PyAny>,
    length: usize,
    sample_count: usize,
    seed: u64,
    default_item_size: usize,
    mut item_size: impl FnMut(Bound<'_, PyAny>) -> PyResult<usize>,
) -> PyResult<usize> {
    if length == 0 {
        return Ok(0);
    }
    let sample_count = sample_count.min(length);
    if sample_count == 0 {
        return Ok(length.saturating_mul(default_item_size));
    }
    let mut sampled = 0usize;
    for sample in 0..sample_count {
        let index = if sample_count == length {
            sample
        } else {
            stratified_sample_index(length, sample_count, sample, seed)
        };
        sampled = sampled.saturating_add(item_size(sequence_item(value, index)?)?);
    }
    Ok(sampled
        .saturating_mul(length)
        .saturating_add(sample_count - 1)
        / sample_count)
}

fn estimated_capacity(estimate: usize) -> usize {
    let headroom = (estimate.saturating_add(99) / 100).max(1);
    estimate.saturating_add(headroom)
}

fn offset_vector_size(length: usize) -> usize {
    aligned_payload_size(length.saturating_mul(4).saturating_add(4), 4)
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

#[derive(Clone, Copy)]
enum SerdeUuidFormat {
    Canonical,
    Hex,
    Bytes,
}

#[derive(Clone, Copy)]
struct SerdeEncodeContext<'a, 'py> {
    plan: &'a NativePlan,
    fallback_encoder: &'a Bound<'py, PyAny>,
    sorted: bool,
    decimal_as_number: bool,
    uuid_format: SerdeUuidFormat,
}

struct SerdeModel<'a, 'py> {
    context: SerdeEncodeContext<'a, 'py>,
    model: Bound<'py, PyAny>,
    object_index: usize,
}

struct SerdeFieldValue<'a, 'py> {
    context: SerdeEncodeContext<'a, 'py>,
    field: &'a FieldWire,
    value: Bound<'py, PyAny>,
}

struct SerdeScalarValue<'a, 'py> {
    value: &'a Bound<'py, PyAny>,
    scalar: ScalarKind,
}

impl Serialize for SerdeScalarValue<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.scalar {
            ScalarKind::Bool => {
                serializer.serialize_bool(self.value.extract::<bool>().map_err(S::Error::custom)?)
            }
            ScalarKind::Float32 | ScalarKind::Float64 => {
                serializer.serialize_f64(self.value.extract::<f64>().map_err(S::Error::custom)?)
            }
            ScalarKind::Uint8 | ScalarKind::Uint16 | ScalarKind::Uint32 | ScalarKind::Uint64 => {
                serializer.serialize_u64(self.value.extract::<u64>().map_err(S::Error::custom)?)
            }
            ScalarKind::Int8 | ScalarKind::Int16 | ScalarKind::Int32 | ScalarKind::Int64 => {
                serializer.serialize_i64(self.value.extract::<i64>().map_err(S::Error::custom)?)
            }
        }
    }
}

fn serialize_numpy<T, S>(value: &Bound<'_, PyAny>, serializer: S) -> Result<S::Ok, S::Error>
where
    T: Copy + numpy::Element + Serialize,
    S: Serializer,
{
    let array: numpy::PyReadonlyArray1<'_, T> = value.extract().map_err(|_| {
        S::Error::custom("numeric vector fields must be one-dimensional with the declared dtype")
    })?;
    match array.as_slice() {
        Ok(values) => values.serialize(serializer),
        Err(_) => array
            .as_array()
            .iter()
            .copied()
            .collect::<Vec<T>>()
            .serialize(serializer),
    }
}

fn serialize_numpy_field<S>(
    value: &Bound<'_, PyAny>,
    scalar: ScalarKind,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.len().map_err(S::Error::custom)? == 0 {
        return serializer.serialize_seq(Some(0))?.end();
    }
    match scalar {
        ScalarKind::Bool => serialize_numpy::<bool, S>(value, serializer),
        ScalarKind::Int8 => serialize_numpy::<i8, S>(value, serializer),
        ScalarKind::Uint8 => serialize_numpy::<u8, S>(value, serializer),
        ScalarKind::Int16 => serialize_numpy::<i16, S>(value, serializer),
        ScalarKind::Uint16 => serialize_numpy::<u16, S>(value, serializer),
        ScalarKind::Int32 => serialize_numpy::<i32, S>(value, serializer),
        ScalarKind::Uint32 => serialize_numpy::<u32, S>(value, serializer),
        ScalarKind::Int64 => serialize_numpy::<i64, S>(value, serializer),
        ScalarKind::Uint64 => serialize_numpy::<u64, S>(value, serializer),
        ScalarKind::Float32 => serialize_numpy::<f32, S>(value, serializer),
        ScalarKind::Float64 => serialize_numpy::<f64, S>(value, serializer),
    }
}

fn serialize_python_int<S>(value: &Bound<'_, PyAny>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if let Ok(value) = value.extract::<i64>() {
        return serializer.serialize_i64(value);
    }
    if let Ok(value) = value.extract::<u64>() {
        return serializer.serialize_u64(value);
    }
    let text = value.str().map_err(S::Error::custom)?;
    let raw: sonic_rs::LazyValue<'_> =
        sonic_rs::from_str(text.to_str().map_err(S::Error::custom)?).map_err(S::Error::custom)?;
    raw.serialize(serializer)
}

fn uuid_text(value: &Bound<'_, PyAny>, format: SerdeUuidFormat) -> PyResult<String> {
    match format {
        SerdeUuidFormat::Canonical => Ok(value.str()?.to_str()?.to_owned()),
        SerdeUuidFormat::Hex => value.getattr("hex")?.extract(),
        SerdeUuidFormat::Bytes => Err(PyValueError::new_err(
            "UUID bytes format is only valid for MessagePack",
        )),
    }
}

fn serialize_uuid<S>(
    value: &Bound<'_, PyAny>,
    format: SerdeUuidFormat,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&uuid_text(value, format).map_err(S::Error::custom)?)
}

fn serialize_decimal<S>(
    value: &Bound<'_, PyAny>,
    as_number: bool,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let text = value.str().map_err(S::Error::custom)?;
    let text = text.to_str().map_err(S::Error::custom)?;
    if !as_number {
        return serializer.serialize_str(text);
    }
    let raw: sonic_rs::LazyValue<'_> = sonic_rs::from_str(text).map_err(S::Error::custom)?;
    raw.serialize(serializer)
}

fn serde_field_is_default(field: &FieldWire, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !field.serde_omit_default {
        return Ok(false);
    }
    match &field.default {
        Value::Nil => Ok(value.is_none()),
        Value::Boolean(default) => Ok(value.extract::<bool>()? == *default),
        Value::Integer(default) => {
            if let Some(default) = default.as_i64() {
                Ok(value.extract::<i64>()? == default)
            } else if let Some(default) = default.as_u64() {
                Ok(value.extract::<u64>()? == default)
            } else {
                Err(PyTypeError::new_err("unsupported integer serde default"))
            }
        }
        Value::F32(default) => Ok(value.extract::<f32>()? == *default),
        Value::F64(default) => Ok(value.extract::<f64>()? == *default),
        Value::String(default) => Ok(value.extract::<&str>()?
            == default
                .as_str()
                .ok_or_else(|| PyTypeError::new_err("serde default is not valid UTF-8"))?),
        Value::Binary(default) => Ok(value.extract::<Vec<u8>>()? == *default),
        _ => Err(PyTypeError::new_err("unsupported native serde default")),
    }
}

fn serialize_serde_field<M>(
    map: &mut M,
    context: SerdeEncodeContext<'_, '_>,
    metadata_field: &SerdeField,
    field: &FieldWire,
    value: Bound<'_, PyAny>,
) -> Result<(), M::Error>
where
    M: SerializeMap,
{
    map.serialize_entry(
        &metadata_field.encode_name,
        &SerdeFieldValue {
            context,
            field,
            value,
        },
    )
}

impl Serialize for SerdeModel<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let metadata =
            serde_object(self.context.plan, self.object_index).map_err(S::Error::custom)?;
        let object = &self.context.plan.objects[self.object_index];
        let mut fields = Vec::with_capacity(metadata.fields.len());
        for metadata_field in &metadata.fields {
            let field = &object.fields[metadata_field.object_field_index];
            let value = self
                .model
                .getattr(metadata_field.attr_name.bind(self.model.py()))
                .map_err(S::Error::custom)?;
            if serde_field_is_default(field, &value).map_err(S::Error::custom)? {
                continue;
            }
            fields.push((metadata_field, field, value));
        }
        let mut map =
            serializer.serialize_map(Some(fields.len() + usize::from(metadata.tag.is_some())))?;
        if !self.context.sorted {
            if let Some((field, value)) = &metadata.tag {
                map.serialize_entry(field, value)?;
            }
            for (metadata_field, field, value) in fields {
                serialize_serde_field(&mut map, self.context, metadata_field, field, value)?;
            }
            return map.end();
        }
        fields.sort_by(|(left, _, _), (right, _, _)| left.encode_name.cmp(&right.encode_name));
        let tag_index = metadata.tag.as_ref().map(|(tag_field, _)| {
            fields.partition_point(|(field, _, _)| field.encode_name < *tag_field)
        });
        for index in 0..=fields.len() {
            if tag_index == Some(index)
                && let Some((field, value)) = &metadata.tag
            {
                map.serialize_entry(field, value)?;
            }
            let Some((metadata_field, field, value)) = fields.get(index) else {
                continue;
            };
            serialize_serde_field(&mut map, self.context, metadata_field, field, value.clone())?;
        }
        map.end()
    }
}

impl Serialize for SerdeFieldValue<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.value.is_none() {
            return serializer.serialize_none();
        }
        match self.field.kind {
            FieldKind::Scalar if self.field.serde_python_int => {
                serialize_python_int(&self.value, serializer)
            }
            FieldKind::Scalar => SerdeScalarValue {
                value: &self.value,
                scalar: self
                    .field
                    .scalar
                    .ok_or_else(|| S::Error::custom("scalar field has no scalar type"))?,
            }
            .serialize(serializer),
            FieldKind::String => {
                serializer.serialize_str(self.value.extract::<&str>().map_err(S::Error::custom)?)
            }
            FieldKind::Uuid => serialize_uuid(&self.value, self.context.uuid_format, serializer),
            FieldKind::Decimal => {
                serialize_decimal(&self.value, self.context.decimal_as_number, serializer)
            }
            FieldKind::Table | FieldKind::Struct | FieldKind::Nested | FieldKind::Union => {
                let object_index = self
                    .context
                    .plan
                    .model_object_index(&self.value)
                    .map_err(S::Error::custom)?;
                SerdeModel {
                    context: self.context,
                    model: self.value.clone(),
                    object_index,
                }
                .serialize(serializer)
            }
            FieldKind::VectorByte | FieldKind::VectorScalar | FieldKind::ArrayScalar => {
                let scalar = if self.field.kind == FieldKind::VectorByte {
                    ScalarKind::Uint8
                } else {
                    self.field.scalar.ok_or_else(|| {
                        S::Error::custom("numeric vector field has no scalar type")
                    })?
                };
                if self.field.enum_type.is_some() {
                    let values = self.value.cast::<PyList>().map_err(S::Error::custom)?;
                    let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                    for value in values {
                        sequence.serialize_element(&SerdeScalarValue {
                            value: &value,
                            scalar,
                        })?;
                    }
                    sequence.end()
                } else {
                    serialize_numpy_field(&self.value, scalar, serializer)
                }
            }
            FieldKind::VectorString => {
                let values = self.value.cast::<PyList>().map_err(S::Error::custom)?;
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence
                        .serialize_element(value.extract::<&str>().map_err(S::Error::custom)?)?;
                }
                sequence.end()
            }
            FieldKind::VectorTable
            | FieldKind::VectorStruct
            | FieldKind::ArrayStruct
            | FieldKind::UnionVector => {
                let values = self.value.cast::<PyList>().map_err(S::Error::custom)?;
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    if value.is_none() {
                        sequence.serialize_element(&Option::<u8>::None)?;
                    } else {
                        let object_index = self
                            .context
                            .plan
                            .model_object_index(&value)
                            .map_err(S::Error::custom)?;
                        sequence.serialize_element(&SerdeModel {
                            context: self.context,
                            model: value,
                            object_index,
                        })?;
                    }
                }
                sequence.end()
            }
            FieldKind::Dynamic => Err(S::Error::custom(
                "dynamic fields require the msgspec serde fallback",
            )),
            FieldKind::Fallback => {
                let encoded = self
                    .context
                    .fallback_encoder
                    .call1((&self.value,))
                    .map_err(S::Error::custom)?;
                let encoded = encoded.cast::<PyBytes>().map_err(S::Error::custom)?;
                let value: sonic_rs::LazyValue<'_> =
                    sonic_rs::from_slice(encoded.as_bytes()).map_err(S::Error::custom)?;
                value.serialize(serializer)
            }
        }
    }
}

fn msgpack_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("cannot encode MessagePack: {error}"))
}

fn with_numpy_slice<T>(
    value: &Bound<'_, PyAny>,
    operation: impl FnOnce(&[T]) -> PyResult<()>,
) -> PyResult<()>
where
    T: Copy + numpy::Element,
{
    let array: numpy::PyReadonlyArray1<'_, T> = value.extract().map_err(|_| {
        PyTypeError::new_err(
            "numeric vector fields must be one-dimensional with the declared dtype",
        )
    })?;
    match array.as_slice() {
        Ok(values) => operation(values),
        Err(_) => {
            let values = array.as_array().iter().copied().collect::<Vec<T>>();
            operation(&values)
        }
    }
}

fn write_messagepack_array(
    value: &Bound<'_, PyAny>,
    scalar: ScalarKind,
    output: &mut Vec<u8>,
) -> PyResult<()> {
    macro_rules! write_values {
        ($ty:ty, $write:expr) => {
            with_numpy_slice::<$ty>(value, |values| {
                rmp::encode::write_array_len(
                    output,
                    u32::try_from(values.len()).map_err(|_| {
                        PyValueError::new_err("numeric vector is too large for MessagePack")
                    })?,
                )
                .map_err(msgpack_error)?;
                for &item in values {
                    $write(output, item).map_err(msgpack_error)?;
                }
                Ok(())
            })
        };
    }

    match scalar {
        ScalarKind::Bool => write_values!(bool, rmp::encode::write_bool),
        ScalarKind::Int8 => write_values!(i8, |output: &mut Vec<u8>, value| {
            rmp::encode::write_sint(output, i64::from(value))
        }),
        ScalarKind::Uint8 => write_values!(u8, |output: &mut Vec<u8>, value| {
            rmp::encode::write_uint(output, u64::from(value))
        }),
        ScalarKind::Int16 => write_values!(i16, |output: &mut Vec<u8>, value| {
            rmp::encode::write_sint(output, i64::from(value))
        }),
        ScalarKind::Uint16 => write_values!(u16, |output: &mut Vec<u8>, value| {
            rmp::encode::write_uint(output, u64::from(value))
        }),
        ScalarKind::Int32 => write_values!(i32, |output: &mut Vec<u8>, value| {
            rmp::encode::write_sint(output, i64::from(value))
        }),
        ScalarKind::Uint32 => write_values!(u32, |output: &mut Vec<u8>, value| {
            rmp::encode::write_uint(output, u64::from(value))
        }),
        ScalarKind::Int64 => write_values!(i64, rmp::encode::write_sint),
        ScalarKind::Uint64 => write_values!(u64, rmp::encode::write_uint),
        ScalarKind::Float32 => write_values!(f32, rmp::encode::write_f32),
        ScalarKind::Float64 => write_values!(f64, rmp::encode::write_f64),
    }
}

fn write_messagepack_model(
    context: SerdeEncodeContext<'_, '_>,
    model: &Bound<'_, PyAny>,
    object_index: usize,
    output: &mut Vec<u8>,
) -> PyResult<()> {
    let metadata = serde_object(context.plan, object_index)?;
    let object = &context.plan.objects[object_index];
    let mut fields = Vec::with_capacity(metadata.fields.len());
    for metadata_field in &metadata.fields {
        let field = &object.fields[metadata_field.object_field_index];
        let value = model.getattr(metadata_field.attr_name.bind(model.py()))?;
        if serde_field_is_default(field, &value)? {
            continue;
        }
        fields.push((metadata_field, field, value));
    }
    let length = fields.len() + usize::from(metadata.tag.is_some());
    rmp::encode::write_map_len(
        output,
        u32::try_from(length)
            .map_err(|_| PyValueError::new_err("generated model has too many fields"))?,
    )
    .map_err(msgpack_error)?;
    if !context.sorted {
        if let Some((field, value)) = &metadata.tag {
            rmp::encode::write_str(output, field).map_err(msgpack_error)?;
            rmp::encode::write_str(output, value).map_err(msgpack_error)?;
        }
        for (metadata_field, field, value) in fields {
            rmp::encode::write_str(output, &metadata_field.encode_name).map_err(msgpack_error)?;
            write_messagepack_field(context, field, &value, output)?;
        }
        return Ok(());
    }
    fields.sort_by(|(left, _, _), (right, _, _)| left.encode_name.cmp(&right.encode_name));
    let tag_index = metadata.tag.as_ref().map(|(tag_field, _)| {
        fields.partition_point(|(field, _, _)| field.encode_name < *tag_field)
    });
    for index in 0..=fields.len() {
        if tag_index == Some(index)
            && let Some((field, value)) = &metadata.tag
        {
            rmp::encode::write_str(output, field).map_err(msgpack_error)?;
            rmp::encode::write_str(output, value).map_err(msgpack_error)?;
        }
        let Some((metadata_field, field, value)) = fields.get(index) else {
            continue;
        };
        rmp::encode::write_str(output, &metadata_field.encode_name).map_err(msgpack_error)?;
        write_messagepack_field(context, field, value, output)?;
    }
    Ok(())
}

fn write_messagepack_scalar(
    output: &mut Vec<u8>,
    scalar: ScalarKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match scalar {
        ScalarKind::Bool => {
            rmp::encode::write_bool(output, value.extract()?).map_err(msgpack_error)?
        }
        ScalarKind::Float32 | ScalarKind::Float64 => {
            rmp::encode::write_f64(output, value.extract()?).map_err(msgpack_error)?
        }
        ScalarKind::Uint8 | ScalarKind::Uint16 | ScalarKind::Uint32 | ScalarKind::Uint64 => {
            rmp::encode::write_uint(output, value.extract()?).map_err(msgpack_error)?;
        }
        ScalarKind::Int8 | ScalarKind::Int16 | ScalarKind::Int32 | ScalarKind::Int64 => {
            rmp::encode::write_sint(output, value.extract()?).map_err(msgpack_error)?;
        }
    }
    Ok(())
}

fn write_messagepack_field(
    context: SerdeEncodeContext<'_, '_>,
    field: &FieldWire,
    value: &Bound<'_, PyAny>,
    output: &mut Vec<u8>,
) -> PyResult<()> {
    if value.is_none() {
        rmp::encode::write_nil(output).map_err(msgpack_error)?;
        return Ok(());
    }
    match field.kind {
        FieldKind::Scalar if field.serde_python_int => {
            if let Ok(value) = value.extract::<i64>() {
                rmp::encode::write_sint(output, value).map_err(msgpack_error)?;
            } else {
                rmp::encode::write_uint(output, value.extract()?).map_err(msgpack_error)?;
            }
        }
        FieldKind::Scalar => write_messagepack_scalar(
            output,
            field
                .scalar
                .ok_or_else(|| PyRuntimeError::new_err("scalar field has no scalar type"))?,
            value,
        )?,
        FieldKind::String => {
            rmp::encode::write_str(output, value.extract()?).map_err(msgpack_error)?;
        }
        FieldKind::Uuid => match context.uuid_format {
            SerdeUuidFormat::Canonical | SerdeUuidFormat::Hex => {
                rmp::encode::write_str(output, &uuid_text(value, context.uuid_format)?)
                    .map_err(msgpack_error)?;
            }
            SerdeUuidFormat::Bytes => {
                let bytes = value.getattr("bytes")?.cast_into::<PyBytes>()?;
                rmp::encode::write_bin_len(
                    output,
                    u32::try_from(bytes.len()?).expect("UUID bytes fit in u32"),
                )
                .map_err(msgpack_error)?;
                output.extend_from_slice(bytes.as_bytes());
            }
        },
        FieldKind::Decimal => {
            if context.decimal_as_number {
                rmp::encode::write_f64(output, value.extract()?).map_err(msgpack_error)?;
            } else {
                rmp::encode::write_str(output, value.str()?.to_str()?).map_err(msgpack_error)?;
            }
        }
        FieldKind::Table | FieldKind::Struct | FieldKind::Nested | FieldKind::Union => {
            write_messagepack_model(
                context,
                value,
                context.plan.model_object_index(value)?,
                output,
            )?;
        }
        FieldKind::VectorByte | FieldKind::VectorScalar | FieldKind::ArrayScalar => {
            let scalar = if field.kind == FieldKind::VectorByte {
                ScalarKind::Uint8
            } else {
                field.scalar.ok_or_else(|| {
                    PyRuntimeError::new_err("numeric vector field has no scalar type")
                })?
            };
            if field.enum_type.is_some() {
                let values = value.cast::<PyList>()?;
                rmp::encode::write_array_len(
                    output,
                    u32::try_from(values.len())
                        .map_err(|_| PyValueError::new_err("enum vector is too large"))?,
                )
                .map_err(msgpack_error)?;
                for item in values {
                    write_messagepack_scalar(output, scalar, &item)?;
                }
            } else {
                write_messagepack_array(value, scalar, output)?;
            }
        }
        FieldKind::VectorString => {
            let values = value.cast::<PyList>()?;
            rmp::encode::write_array_len(
                output,
                u32::try_from(values.len())
                    .map_err(|_| PyValueError::new_err("string vector is too large"))?,
            )
            .map_err(msgpack_error)?;
            for item in values {
                rmp::encode::write_str(output, item.extract()?).map_err(msgpack_error)?;
            }
        }
        FieldKind::VectorTable
        | FieldKind::VectorStruct
        | FieldKind::ArrayStruct
        | FieldKind::UnionVector => {
            let values = value.cast::<PyList>()?;
            rmp::encode::write_array_len(
                output,
                u32::try_from(values.len())
                    .map_err(|_| PyValueError::new_err("object vector is too large"))?,
            )
            .map_err(msgpack_error)?;
            for item in values {
                if item.is_none() {
                    rmp::encode::write_nil(output).map_err(msgpack_error)?;
                } else {
                    write_messagepack_model(
                        context,
                        &item,
                        context.plan.model_object_index(&item)?,
                        output,
                    )?;
                }
            }
        }
        FieldKind::Dynamic => {
            return Err(PyNotImplementedError::new_err(
                "dynamic fields require the msgspec serde fallback",
            ));
        }
        FieldKind::Fallback => {
            let encoded = context.fallback_encoder.call1((value,))?;
            output.extend_from_slice(encoded.cast::<PyBytes>()?.as_bytes());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SerdeModelChoice<'a> {
    Known(usize),
    Union(&'a [ArmWire]),
}

#[derive(Clone, Copy)]
struct SerdeDecodeContext<'a, 'py> {
    plan: &'a NativePlan,
    py: Python<'py>,
    is_json: bool,
    strict: bool,
    fallback_decoders: &'a Bound<'py, PyDict>,
}

#[derive(Clone, Copy)]
struct SerdeModelSeed<'a, 'py> {
    context: SerdeDecodeContext<'a, 'py>,
    choice: SerdeModelChoice<'a>,
}

#[derive(Clone, Copy)]
struct SerdeFieldSeed<'a, 'py> {
    context: SerdeDecodeContext<'a, 'py>,
    field: &'a FieldWire,
}

struct ParsedSerdeField {
    encode_name: String,
    value: Py<PyAny>,
}

struct BufferedSerdeField {
    encode_name: String,
    encoded: Vec<u8>,
}

enum SerdeFieldResolution<'a> {
    Missing,
    Resolved(&'a FieldWire),
    Ambiguous,
}

fn serde_object(plan: &NativePlan, index: usize) -> PyResult<&SerdeObject> {
    let bound_types = plan
        .bound_types
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
    bound_types
        .serde_objects
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| PyRuntimeError::new_err("generated model has no serde metadata"))
}

fn serde_field_by_name<'a>(
    plan: &'a NativePlan,
    object_index: usize,
    encode_name: &str,
) -> PyResult<Option<(&'a SerdeField, &'a FieldWire)>> {
    let metadata = serde_object(plan, object_index)?;
    Ok(metadata
        .fields
        .iter()
        .find(|field| field.encode_name == encode_name)
        .map(|field| {
            (
                field,
                &plan.objects[object_index].fields[field.object_field_index],
            )
        }))
}

fn same_serde_field_shape(left: &FieldWire, right: &FieldWire) -> bool {
    left.kind == right.kind
        && left.scalar == right.scalar
        && left.target_index == right.target_index
        && left.fixed_length == right.fixed_length
        && left.fallback_id == right.fallback_id
        && left.serde_omit_default == right.serde_omit_default
        && left.arms.len() == right.arms.len()
        && left
            .arms
            .iter()
            .zip(&right.arms)
            .all(|(left, right)| left.target_index == right.target_index)
}

fn resolve_serde_field<'a>(
    plan: &'a NativePlan,
    choice: SerdeModelChoice<'a>,
    encode_name: &str,
) -> PyResult<SerdeFieldResolution<'a>> {
    match choice {
        SerdeModelChoice::Known(index) => {
            Ok(match serde_field_by_name(plan, index, encode_name)? {
                Some((_, field)) => SerdeFieldResolution::Resolved(field),
                None => SerdeFieldResolution::Missing,
            })
        }
        SerdeModelChoice::Union(arms) => {
            let mut selected = None;
            for arm in arms {
                let Some((_, field)) = serde_field_by_name(plan, arm.target_index, encode_name)?
                else {
                    continue;
                };
                if let Some(previous) = selected
                    && !same_serde_field_shape(previous, field)
                {
                    return Ok(SerdeFieldResolution::Ambiguous);
                }
                selected = Some(field);
            }
            Ok(match selected {
                Some(field) => SerdeFieldResolution::Resolved(field),
                None => SerdeFieldResolution::Missing,
            })
        }
    }
}

fn serde_choice_tag_field<'a>(
    plan: &'a NativePlan,
    choice: SerdeModelChoice<'a>,
) -> PyResult<Option<&'a str>> {
    let index = match choice {
        SerdeModelChoice::Known(index) => index,
        SerdeModelChoice::Union(arms) => match arms.first() {
            Some(arm) => arm.target_index,
            None => return Ok(None),
        },
    };
    Ok(serde_object(plan, index)?
        .tag
        .as_ref()
        .map(|(field, _)| field.as_str()))
}

fn serde_choice_tag(
    plan: &NativePlan,
    choice: SerdeModelChoice<'_>,
    tag: &str,
) -> PyResult<Option<usize>> {
    let matches = |index| -> PyResult<bool> {
        Ok(serde_object(plan, index)?
            .tag
            .as_ref()
            .is_some_and(|(_, value)| value == tag))
    };
    match choice {
        SerdeModelChoice::Known(index) => Ok(matches(index)?.then_some(index)),
        SerdeModelChoice::Union(arms) => {
            for arm in arms {
                if matches(arm.target_index)? {
                    return Ok(Some(arm.target_index));
                }
            }
            Ok(None)
        }
    }
}

fn serde_construct_model(
    plan: &NativePlan,
    py: Python<'_>,
    object_index: usize,
    parsed: Vec<ParsedSerdeField>,
) -> PyResult<Py<PyAny>> {
    let metadata = serde_object(plan, object_index)?;
    let model_type = plan.bound_type(py, &plan.objects[object_index].name)?;
    if parsed.len() == metadata.fields.len() {
        let positional = metadata
            .fields
            .iter()
            .map(|field| {
                parsed
                    .iter()
                    .find(|parsed| parsed.encode_name == field.encode_name)
                    .map(|parsed| parsed.value.clone_ref(py))
            })
            .collect::<Option<Vec<_>>>();
        if let Some(positional) = positional {
            let pointers = positional
                .iter()
                .map(|value| value.as_ptr())
                .collect::<Vec<_>>();
            // SAFETY: every pointer comes from a live owned Py object in
            // `positional`; `keyword_names` has the same length and remains live
            // for the call; vectorcall returns a new reference.
            let result = unsafe {
                ffi::PyObject_Vectorcall(
                    model_type.as_ptr(),
                    pointers.as_ptr(),
                    0,
                    metadata.keyword_names.bind(py).as_ptr(),
                )
            };
            // SAFETY: a non-null vectorcall result is a new owned reference.
            return unsafe { Bound::from_owned_ptr_or_err(py, result) }.map(Bound::unbind);
        }
    }

    let kwargs = PyDict::new(py);
    for parsed_field in parsed {
        let Some((field, _)) = serde_field_by_name(plan, object_index, &parsed_field.encode_name)?
        else {
            return Err(PyValueError::new_err(format!(
                "field {:?} does not belong to tagged model {}",
                parsed_field.encode_name, plan.objects[object_index].name
            )));
        };
        kwargs.set_item(field.attr_name.bind(py), parsed_field.value.bind(py))?;
    }
    model_type.call((), Some(&kwargs)).map(Bound::unbind)
}

fn deserialize_buffered_serde_field(
    context: SerdeDecodeContext<'_, '_>,
    field: &FieldWire,
    encoded: &[u8],
) -> PyResult<Py<PyAny>> {
    let seed = SerdeFieldSeed { context, field };
    if context.is_json {
        let mut deserializer = sonic_rs::Deserializer::from_slice(encoded);
        let value = seed.deserialize(&mut deserializer).map_err(|error| {
            PyValueError::new_err(format!("cannot decode buffered JSON field: {error}"))
        })?;
        deserializer.end().map_err(|error| {
            PyValueError::new_err(format!("cannot decode buffered JSON field: {error}"))
        })?;
        Ok(value)
    } else {
        let mut deserializer = rmp_serde::Deserializer::new(encoded);
        let value = seed.deserialize(&mut deserializer).map_err(|error| {
            PyValueError::new_err(format!("cannot decode buffered MessagePack field: {error}"))
        })?;
        if !deserializer.get_ref().is_empty() {
            return Err(PyValueError::new_err(
                "buffered MessagePack field contains trailing data",
            ));
        }
        Ok(value)
    }
}

fn decode_buffered_serde_fields(
    seed: SerdeModelSeed<'_, '_>,
    object_index: usize,
    buffered: &mut Vec<BufferedSerdeField>,
    parsed: &mut Vec<ParsedSerdeField>,
) -> PyResult<()> {
    for buffered_field in buffered.drain(..) {
        let Some((_, field)) =
            serde_field_by_name(seed.context.plan, object_index, &buffered_field.encode_name)?
        else {
            continue;
        };
        parsed.push(ParsedSerdeField {
            encode_name: buffered_field.encode_name,
            value: deserialize_buffered_serde_field(seed.context, field, &buffered_field.encoded)?,
        });
    }
    Ok(())
}

impl<'de> DeserializeSeed<'de> for SerdeModelSeed<'_, '_> {
    type Value = Py<PyAny>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(SerdeModelVisitor { seed: self })
    }
}

struct SerdeModelVisitor<'a, 'py> {
    seed: SerdeModelSeed<'a, 'py>,
}

impl<'de> Visitor<'de> for SerdeModelVisitor<'_, '_> {
    type Value = Py<PyAny>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a generated model object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let tag_field = serde_choice_tag_field(self.seed.context.plan, self.seed.choice)
            .map_err(A::Error::custom)?
            .map(str::to_owned);
        let mut selected = match self.seed.choice {
            SerdeModelChoice::Known(index) => Some(index),
            SerdeModelChoice::Union(_) => None,
        };
        let mut parsed = Vec::with_capacity(map.size_hint().unwrap_or(0));
        let mut buffered = Vec::new();
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            if tag_field.as_deref() == Some(key.as_ref()) {
                let tag = map.next_value::<String>()?;
                let object_index = serde_choice_tag(self.seed.context.plan, self.seed.choice, &tag)
                    .map_err(A::Error::custom)?
                    .ok_or_else(|| {
                        A::Error::custom(format!("unknown generated model tag {tag:?}"))
                    })?;
                selected = Some(object_index);
                decode_buffered_serde_fields(self.seed, object_index, &mut buffered, &mut parsed)
                    .map_err(A::Error::custom)?;
                continue;
            }
            let choice = selected.map_or(self.seed.choice, SerdeModelChoice::Known);
            let field = match resolve_serde_field(self.seed.context.plan, choice, key.as_ref())
                .map_err(A::Error::custom)?
            {
                SerdeFieldResolution::Missing => {
                    map.next_value::<IgnoredAny>()?;
                    continue;
                }
                SerdeFieldResolution::Resolved(field) => field,
                SerdeFieldResolution::Ambiguous => {
                    let encoded = if self.seed.context.is_json {
                        let value = map.next_value::<sonic_rs::LazyValue<'de>>()?;
                        value.as_raw_str().as_bytes().to_vec()
                    } else {
                        let value = map.next_value::<Value>()?;
                        rmp_serde::to_vec(&value).map_err(A::Error::custom)?
                    };
                    buffered.push(BufferedSerdeField {
                        encode_name: key.into_owned(),
                        encoded,
                    });
                    continue;
                }
            };
            let value = map.next_value_seed(SerdeFieldSeed {
                context: self.seed.context,
                field,
            })?;
            parsed.push(ParsedSerdeField {
                encode_name: key.into_owned(),
                value,
            });
        }
        let object_index = selected.ok_or_else(|| {
            A::Error::custom("tagged generated model object is missing its type tag")
        })?;
        serde_construct_model(
            self.seed.context.plan,
            self.seed.context.py,
            object_index,
            parsed,
        )
        .map_err(A::Error::custom)
    }
}

struct PythonIntVisitor<'py> {
    py: Python<'py>,
    coerce: bool,
}

impl<'de> Visitor<'de> for PythonIntVisitor<'_> {
    type Value = Py<PyAny>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an integer")
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        value.into_py_any(self.py).map_err(E::custom)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        value.into_py_any(self.py).map_err(E::custom)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        if !self.coerce || !value.is_finite() || value.fract() != 0.0 {
            return Err(E::custom("float is not coercible to an integer"));
        }
        if value < 0.0 {
            (value as i64).into_py_any(self.py).map_err(E::custom)
        } else {
            (value as u64).into_py_any(self.py).map_err(E::custom)
        }
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        if !self.coerce {
            return Err(E::custom("expected an integer"));
        }
        if let Ok(parsed) = value.parse::<i64>()
            && parsed.to_string() == value
        {
            return parsed.into_py_any(self.py).map_err(E::custom);
        }
        if let Ok(parsed) = value.parse::<u64>()
            && parsed.to_string() == value
        {
            return parsed.into_py_any(self.py).map_err(E::custom);
        }
        let parsed = sonic_rs::from_str::<f64>(value).map_err(E::custom)?;
        self.visit_f64(parsed)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        self.visit_str(&value)
    }
}

fn deserialize_python_int<'de, D>(
    deserializer: D,
    py: Python<'_>,
    is_json: bool,
    strict: bool,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    if is_json && strict {
        let value = sonic_rs::LazyValue::deserialize(deserializer)?;
        let raw = value.as_raw_str();
        if !matches!(raw.as_bytes().first(), Some(b'-' | b'0'..=b'9')) {
            return Err(D::Error::custom("expected an integer"));
        }
        if let Ok(value) = raw.parse::<i64>() {
            return value.into_py_any(py).map_err(D::Error::custom);
        }
        if let Ok(value) = raw.parse::<u64>() {
            return value.into_py_any(py).map_err(D::Error::custom);
        }
        return py
            .get_type::<PyInt>()
            .call1((raw,))
            .map(|value| value.into_any().unbind())
            .map_err(D::Error::custom);
    }
    deserializer.deserialize_any(PythonIntVisitor {
        py,
        coerce: !strict,
    })
}

fn scalar_from_integer(kind: ScalarKind, value: i128) -> Result<ScalarValue, String> {
    macro_rules! checked {
        ($variant:ident, $ty:ty) => {
            <$ty>::try_from(value)
                .map(ScalarValue::$variant)
                .map_err(|_| format!("integer {value} is out of range for {kind:?}"))
        };
    }
    match kind {
        ScalarKind::Bool if matches!(value, 0 | 1) => Ok(ScalarValue::Bool(value == 1)),
        ScalarKind::Bool => Err(format!("integer {value} is not a boolean")),
        ScalarKind::Int8 => checked!(Int8, i8),
        ScalarKind::Uint8 => checked!(Uint8, u8),
        ScalarKind::Int16 => checked!(Int16, i16),
        ScalarKind::Uint16 => checked!(Uint16, u16),
        ScalarKind::Int32 => checked!(Int32, i32),
        ScalarKind::Uint32 => checked!(Uint32, u32),
        ScalarKind::Int64 => checked!(Int64, i64),
        ScalarKind::Uint64 => checked!(Uint64, u64),
        ScalarKind::Float32 => Ok(ScalarValue::Float32(value as f32)),
        ScalarKind::Float64 => Ok(ScalarValue::Float64(value as f64)),
    }
}

fn scalar_from_i64(kind: ScalarKind, value: i64) -> Result<ScalarValue, String> {
    scalar_from_integer(kind, i128::from(value))
}

fn scalar_from_u64(kind: ScalarKind, value: u64) -> Result<ScalarValue, String> {
    scalar_from_integer(kind, i128::from(value))
}

fn scalar_from_f64(kind: ScalarKind, value: f64) -> Result<ScalarValue, String> {
    if matches!(kind, ScalarKind::Float32) {
        return Ok(ScalarValue::Float32(value as f32));
    }
    if matches!(kind, ScalarKind::Float64) {
        return Ok(ScalarValue::Float64(value));
    }
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(format!("float {value} is not an integer"));
    }
    if value < 0.0 {
        if value < i64::MIN as f64 {
            return Err(format!("float {value} is out of integer range"));
        }
        return scalar_from_i64(kind, value as i64);
    }
    if value >= 18_446_744_073_709_551_616.0 {
        return Err(format!("float {value} is out of integer range"));
    }
    scalar_from_u64(kind, value as u64)
}

fn scalar_from_str(kind: ScalarKind, value: &str) -> Result<ScalarValue, String> {
    if kind == ScalarKind::Bool {
        return match value.to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(ScalarValue::Bool(true)),
            "false" | "0" => Ok(ScalarValue::Bool(false)),
            _ => Err(format!("string {value:?} is not a boolean")),
        };
    }
    if matches!(kind, ScalarKind::Float32 | ScalarKind::Float64) {
        return value
            .parse::<f64>()
            .map_err(|_| format!("string {value:?} is not a float"))
            .and_then(|value| scalar_from_f64(kind, value));
    }
    if let Ok(parsed) = value.parse::<i64>()
        && parsed.to_string() == value
    {
        return scalar_from_i64(kind, parsed);
    }
    if let Ok(parsed) = value.parse::<u64>()
        && parsed.to_string() == value
    {
        return scalar_from_u64(kind, parsed);
    }
    let parsed = sonic_rs::from_str::<f64>(value)
        .map_err(|_| format!("string {value:?} is not an integer"))?;
    scalar_from_f64(kind, parsed)
}

struct CoercedScalarVisitor {
    kind: ScalarKind,
}

impl<'de> Visitor<'de> for CoercedScalarVisitor {
    type Value = ScalarValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "a value coercible to {:?}", self.kind)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        if self.kind == ScalarKind::Bool {
            Ok(ScalarValue::Bool(value))
        } else {
            Err(E::custom("boolean is not coercible to a numeric field"))
        }
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        scalar_from_i64(self.kind, value).map_err(E::custom)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        scalar_from_u64(self.kind, value).map_err(E::custom)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        scalar_from_f64(self.kind, value).map_err(E::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        scalar_from_str(self.kind, value).map_err(E::custom)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        self.visit_str(&value)
    }
}

fn deserialize_scalar_value<'de, D>(
    deserializer: D,
    kind: ScalarKind,
    strict: bool,
) -> Result<ScalarValue, D::Error>
where
    D: Deserializer<'de>,
{
    if !strict {
        return deserializer.deserialize_any(CoercedScalarVisitor { kind });
    }
    Ok(match kind {
        ScalarKind::Bool => ScalarValue::Bool(bool::deserialize(deserializer)?),
        ScalarKind::Int8 => ScalarValue::Int8(i8::deserialize(deserializer)?),
        ScalarKind::Uint8 => ScalarValue::Uint8(u8::deserialize(deserializer)?),
        ScalarKind::Int16 => ScalarValue::Int16(i16::deserialize(deserializer)?),
        ScalarKind::Uint16 => ScalarValue::Uint16(u16::deserialize(deserializer)?),
        ScalarKind::Int32 => ScalarValue::Int32(i32::deserialize(deserializer)?),
        ScalarKind::Uint32 => ScalarValue::Uint32(u32::deserialize(deserializer)?),
        ScalarKind::Int64 => ScalarValue::Int64(i64::deserialize(deserializer)?),
        ScalarKind::Uint64 => ScalarValue::Uint64(u64::deserialize(deserializer)?),
        ScalarKind::Float32 | ScalarKind::Float64 => {
            ScalarValue::Float64(f64::deserialize(deserializer)?)
        }
    })
}

fn deserialize_uuid<'de, D>(
    deserializer: D,
    plan: &NativePlan,
    py: Python<'_>,
    is_json: bool,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    if is_json {
        let value = String::deserialize(deserializer)?;
        return plan
            .construct_uuid_text(py, &value)
            .map_err(D::Error::custom);
    }
    match Value::deserialize(deserializer)? {
        Value::String(value) => plan
            .construct_uuid_text(
                py,
                value
                    .as_str()
                    .ok_or_else(|| D::Error::custom("UUID string is not valid UTF-8"))?,
            )
            .map_err(D::Error::custom),
        Value::Binary(value) => plan
            .construct_uuid_bytes(py, &value)
            .map_err(D::Error::custom),
        _ => Err(D::Error::custom("expected a UUID string or 16-byte value")),
    }
}

fn deserialize_decimal<'de, D>(
    deserializer: D,
    plan: &NativePlan,
    py: Python<'_>,
    is_json: bool,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    let text = if is_json {
        let value = sonic_rs::LazyValue::deserialize(deserializer)?;
        let raw = value.as_raw_str();
        if raw.starts_with('"') {
            sonic_rs::from_str::<String>(raw).map_err(D::Error::custom)?
        } else {
            raw.to_owned()
        }
    } else {
        match Value::deserialize(deserializer)? {
            Value::String(value) => value
                .as_str()
                .ok_or_else(|| D::Error::custom("Decimal string is not valid UTF-8"))?
                .to_owned(),
            Value::Integer(value) => value.to_string(),
            Value::F32(value) => value.to_string(),
            Value::F64(value) => value.to_string(),
            _ => return Err(D::Error::custom("expected a Decimal string or number")),
        }
    };
    plan.construct_decimal(py, &text).map_err(D::Error::custom)
}

impl SerdeFieldSeed<'_, '_> {
    fn deserialize_value<'de, D>(self, deserializer: D) -> Result<Py<PyAny>, D::Error>
    where
        D: Deserializer<'de>,
    {
        match self.field.kind {
            FieldKind::Scalar if self.field.serde_python_int => deserialize_python_int(
                deserializer,
                self.context.py,
                self.context.is_json,
                self.context.strict,
            ),
            FieldKind::Scalar => {
                let scalar = self
                    .field
                    .scalar
                    .ok_or_else(|| D::Error::custom("scalar field has no scalar type"))?;
                let value = deserialize_scalar_value(deserializer, scalar, self.context.strict)?;
                let value = value.into_py(self.context.py).map_err(D::Error::custom)?;
                self.context
                    .plan
                    .apply_enum(self.context.py, self.field, value)
                    .map_err(D::Error::custom)
            }
            FieldKind::String => String::deserialize(deserializer)
                .and_then(|value| value.into_py_any(self.context.py).map_err(D::Error::custom)),
            FieldKind::Uuid => deserialize_uuid(
                deserializer,
                self.context.plan,
                self.context.py,
                self.context.is_json,
            ),
            FieldKind::Decimal => deserialize_decimal(
                deserializer,
                self.context.plan,
                self.context.py,
                self.context.is_json,
            ),
            FieldKind::Table | FieldKind::Struct | FieldKind::Nested => {
                let target = self
                    .field
                    .target_index
                    .ok_or_else(|| D::Error::custom("model field has no target"))?;
                SerdeModelSeed {
                    context: self.context,
                    choice: SerdeModelChoice::Known(target),
                }
                .deserialize(deserializer)
            }
            FieldKind::Union => SerdeModelSeed {
                context: self.context,
                choice: SerdeModelChoice::Union(&self.field.arms),
            }
            .deserialize(deserializer),
            FieldKind::VectorScalar | FieldKind::ArrayScalar => {
                if self.field.enum_type.is_some() {
                    return deserialize_enum_vector(
                        deserializer,
                        self.context.plan,
                        self.context.py,
                        self.field,
                        self.context.strict,
                    );
                }
                let scalar = self
                    .field
                    .scalar
                    .ok_or_else(|| D::Error::custom("numeric vector field has no scalar type"))?;
                deserialize_numpy_array(
                    deserializer,
                    self.context.py,
                    scalar,
                    self.field.fixed_length,
                    self.context.strict,
                )
            }
            FieldKind::VectorString => {
                Vec::<String>::deserialize(deserializer).and_then(|values| {
                    PyList::new(self.context.py, values)
                        .map(|value| value.into_any().unbind())
                        .map_err(D::Error::custom)
                })
            }
            FieldKind::VectorTable | FieldKind::VectorStruct | FieldKind::ArrayStruct => {
                let target = self
                    .field
                    .target_index
                    .ok_or_else(|| D::Error::custom("model vector field has no target"))?;
                SerdeObjectVectorSeed {
                    context: self.context,
                    choice: SerdeModelChoice::Known(target),
                    fixed_length: self.field.fixed_length,
                    element_nullable: self.field.serde_element_nullable,
                }
                .deserialize(deserializer)
            }
            FieldKind::UnionVector => SerdeObjectVectorSeed {
                context: self.context,
                choice: SerdeModelChoice::Union(&self.field.arms),
                fixed_length: self.field.fixed_length,
                element_nullable: self.field.serde_element_nullable,
            }
            .deserialize(deserializer),
            FieldKind::VectorByte | FieldKind::Dynamic => Err(D::Error::custom(
                "field requires the msgspec serde fallback",
            )),
            FieldKind::Fallback => {
                let fallback_id = self
                    .field
                    .fallback_id
                    .as_deref()
                    .ok_or_else(|| D::Error::custom("fallback field has no callback id"))?;
                let decoder = self
                    .context
                    .fallback_decoders
                    .get_item(fallback_id)
                    .map_err(D::Error::custom)?
                    .ok_or_else(|| {
                        D::Error::custom(format!(
                            "serde fallback decoder {fallback_id:?} is not bound"
                        ))
                    })?;
                if self.context.is_json {
                    let value = sonic_rs::LazyValue::deserialize(deserializer)?;
                    decoder
                        .call1((PyBytes::new(self.context.py, value.as_raw_str().as_bytes()),))
                        .map(Bound::unbind)
                        .map_err(D::Error::custom)
                } else {
                    let value = Value::deserialize(deserializer)?;
                    let encoded = rmp_serde::to_vec(&value).map_err(D::Error::custom)?;
                    decoder
                        .call1((PyBytes::new(self.context.py, &encoded),))
                        .map(Bound::unbind)
                        .map_err(D::Error::custom)
                }
            }
        }
    }
}

impl<'de> DeserializeSeed<'de> for SerdeFieldSeed<'_, '_> {
    type Value = Py<PyAny>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if self.field.serde_nullable {
            deserializer.deserialize_option(SerdeFieldOptionVisitor { seed: self })
        } else {
            self.deserialize_value(deserializer)
        }
    }
}

struct SerdeFieldOptionVisitor<'a, 'py> {
    seed: SerdeFieldSeed<'a, 'py>,
}

impl<'de> Visitor<'de> for SerdeFieldOptionVisitor<'_, '_> {
    type Value = Py<PyAny>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a generated model field")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        Ok(self.seed.context.py.None())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        self.visit_none()
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.seed.deserialize_value(deserializer)
    }
}

#[derive(Clone, Copy)]
struct SerdeObjectVectorSeed<'a, 'py> {
    context: SerdeDecodeContext<'a, 'py>,
    choice: SerdeModelChoice<'a>,
    fixed_length: usize,
    element_nullable: bool,
}

impl<'de> DeserializeSeed<'de> for SerdeObjectVectorSeed<'_, '_> {
    type Value = Py<PyAny>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SerdeObjectVectorVisitor { seed: self })
    }
}

struct SerdeObjectVectorVisitor<'a, 'py> {
    seed: SerdeObjectVectorSeed<'a, 'py>,
}

#[derive(Clone, Copy)]
struct OptionalSerdeModelSeed<'a, 'py> {
    seed: SerdeModelSeed<'a, 'py>,
}

impl<'de> DeserializeSeed<'de> for OptionalSerdeModelSeed<'_, '_> {
    type Value = Py<PyAny>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_option(OptionalSerdeModelVisitor { seed: self.seed })
    }
}

struct OptionalSerdeModelVisitor<'a, 'py> {
    seed: SerdeModelSeed<'a, 'py>,
}

impl<'de> Visitor<'de> for OptionalSerdeModelVisitor<'_, '_> {
    type Value = Py<PyAny>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a generated model object or null")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        Ok(self.seed.context.py.None())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: DeserializeError,
    {
        self.visit_none()
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.seed.deserialize(deserializer)
    }
}

impl<'de> Visitor<'de> for SerdeObjectVectorVisitor<'_, '_> {
    type Value = Py<PyAny>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a vector of generated model objects")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        loop {
            let model_seed = SerdeModelSeed {
                context: self.seed.context,
                choice: self.seed.choice,
            };
            let value = if self.seed.element_nullable {
                sequence.next_element_seed(OptionalSerdeModelSeed { seed: model_seed })?
            } else {
                sequence.next_element_seed(model_seed)?
            };
            match value {
                Some(value) => values.push(value),
                None => break,
            }
        }
        if self.seed.fixed_length != 0 && values.len() != self.seed.fixed_length {
            return Err(A::Error::custom(format!(
                "fixed object array requires {} values, got {}",
                self.seed.fixed_length,
                values.len()
            )));
        }
        PyList::new(self.seed.context.py, values)
            .map(|value| value.into_any().unbind())
            .map_err(A::Error::custom)
    }
}

fn finish_numpy_values<T, E>(
    py: Python<'_>,
    values: Vec<T>,
    fixed_length: usize,
) -> Result<Py<PyAny>, E>
where
    T: numpy::Element,
    E: DeserializeError,
{
    if fixed_length != 0 && values.len() != fixed_length {
        return Err(E::custom(format!(
            "fixed numeric array requires {fixed_length} values, got {}",
            values.len()
        )));
    }
    Ok(numpy::PyArray1::from_slice(py, &values).into_any().unbind())
}

fn scalar_from_value(kind: ScalarKind, value: Value, strict: bool) -> Result<ScalarValue, String> {
    match value {
        Value::Boolean(value) if kind == ScalarKind::Bool => Ok(ScalarValue::Bool(value)),
        Value::Integer(value) if !strict || kind != ScalarKind::Bool => {
            if let Some(value) = value.as_i64() {
                scalar_from_i64(kind, value)
            } else if let Some(value) = value.as_u64() {
                scalar_from_u64(kind, value)
            } else {
                Err("integer is outside the supported range".to_owned())
            }
        }
        Value::F32(value)
            if !strict || matches!(kind, ScalarKind::Float32 | ScalarKind::Float64) =>
        {
            scalar_from_f64(kind, f64::from(value))
        }
        Value::F64(value)
            if !strict || matches!(kind, ScalarKind::Float32 | ScalarKind::Float64) =>
        {
            scalar_from_f64(kind, value)
        }
        Value::String(value) if !strict => scalar_from_str(
            kind,
            value
                .as_str()
                .ok_or_else(|| "numeric string is not valid UTF-8".to_owned())?,
        ),
        Value::Nil if kind == ScalarKind::Float32 => Ok(ScalarValue::Float32(f32::NAN)),
        Value::Nil if kind == ScalarKind::Float64 => Ok(ScalarValue::Float64(f64::NAN)),
        _ if strict => Err(format!("expected {kind:?}")),
        _ => Err(format!("value is not coercible to {kind:?}")),
    }
}

fn deserialize_enum_vector<'de, D>(
    deserializer: D,
    plan: &NativePlan,
    py: Python<'_>,
    field: &FieldWire,
    strict: bool,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    let scalar = field
        .scalar
        .ok_or_else(|| D::Error::custom("enum vector has no scalar type"))?;
    let raw = Vec::<Value>::deserialize(deserializer)?;
    if field.fixed_length != 0 && raw.len() != field.fixed_length {
        return Err(D::Error::custom(format!(
            "fixed enum array requires {} values, got {}",
            field.fixed_length,
            raw.len()
        )));
    }
    let mut values = Vec::with_capacity(raw.len());
    for value in raw {
        let value = scalar_from_value(scalar, value, strict)
            .map_err(D::Error::custom)?
            .into_py(py)
            .map_err(D::Error::custom)?;
        values.push(
            plan.apply_enum(py, field, value)
                .map_err(D::Error::custom)?,
        );
    }
    PyList::new(py, values)
        .map(|value| value.into_any().unbind())
        .map_err(D::Error::custom)
}

fn finish_coerced_numpy<E>(
    py: Python<'_>,
    values: Vec<ScalarValue>,
    kind: ScalarKind,
    fixed_length: usize,
) -> Result<Py<PyAny>, E>
where
    E: DeserializeError,
{
    macro_rules! finish {
        ($variant:ident, $ty:ty) => {{
            let values = values
                .into_iter()
                .map(|value| match value {
                    ScalarValue::$variant(value) => Ok(value),
                    _ => Err(E::custom("coerced numeric vector changed scalar type")),
                })
                .collect::<Result<Vec<$ty>, E>>()?;
            finish_numpy_values(py, values, fixed_length)
        }};
    }
    match kind {
        ScalarKind::Bool => finish!(Bool, bool),
        ScalarKind::Int8 => finish!(Int8, i8),
        ScalarKind::Uint8 => finish!(Uint8, u8),
        ScalarKind::Int16 => finish!(Int16, i16),
        ScalarKind::Uint16 => finish!(Uint16, u16),
        ScalarKind::Int32 => finish!(Int32, i32),
        ScalarKind::Uint32 => finish!(Uint32, u32),
        ScalarKind::Int64 => finish!(Int64, i64),
        ScalarKind::Uint64 => finish!(Uint64, u64),
        ScalarKind::Float32 => finish!(Float32, f32),
        ScalarKind::Float64 => finish!(Float64, f64),
    }
}

fn deserialize_numpy_values<'de, T, D>(
    deserializer: D,
    py: Python<'_>,
    fixed_length: usize,
) -> Result<Py<PyAny>, D::Error>
where
    T: Deserialize<'de> + numpy::Element,
    D: Deserializer<'de>,
{
    finish_numpy_values(py, Vec::<T>::deserialize(deserializer)?, fixed_length)
}

fn deserialize_numpy_array<'de, D>(
    deserializer: D,
    py: Python<'_>,
    scalar: ScalarKind,
    fixed_length: usize,
    strict: bool,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    if !strict {
        let values = Vec::<Value>::deserialize(deserializer)?
            .into_iter()
            .map(|value| scalar_from_value(scalar, value, false).map_err(D::Error::custom))
            .collect::<Result<Vec<_>, _>>()?;
        return finish_coerced_numpy(py, values, scalar, fixed_length);
    }
    match scalar {
        ScalarKind::Bool => deserialize_numpy_values::<bool, D>(deserializer, py, fixed_length),
        ScalarKind::Int8 => deserialize_numpy_values::<i8, D>(deserializer, py, fixed_length),
        ScalarKind::Uint8 => deserialize_numpy_values::<u8, D>(deserializer, py, fixed_length),
        ScalarKind::Int16 => deserialize_numpy_values::<i16, D>(deserializer, py, fixed_length),
        ScalarKind::Uint16 => deserialize_numpy_values::<u16, D>(deserializer, py, fixed_length),
        ScalarKind::Int32 => deserialize_numpy_values::<i32, D>(deserializer, py, fixed_length),
        ScalarKind::Uint32 => deserialize_numpy_values::<u32, D>(deserializer, py, fixed_length),
        ScalarKind::Int64 => deserialize_numpy_values::<i64, D>(deserializer, py, fixed_length),
        ScalarKind::Uint64 => deserialize_numpy_values::<u64, D>(deserializer, py, fixed_length),
        ScalarKind::Float32 => {
            let values = Vec::<Option<f32>>::deserialize(deserializer)?
                .into_iter()
                .map(|value| value.unwrap_or(f32::NAN))
                .collect();
            finish_numpy_values(py, values, fixed_length)
        }
        ScalarKind::Float64 => {
            let values = Vec::<Option<f64>>::deserialize(deserializer)?
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect();
            finish_numpy_values(py, values, fixed_length)
        }
    }
}

fn initialized_numpy_array<'py, T>(
    py: Python<'py>,
    length: usize,
    initialize: impl FnOnce(&mut [T]),
) -> Bound<'py, numpy::PyArray1<T>>
where
    T: numpy::Element,
{
    let array = numpy::PyArray1::<T>::zeros(py, length, false);
    {
        let mut writable = array.readwrite();
        let values = writable
            .as_slice_mut()
            .expect("a newly allocated one-dimensional array is contiguous");
        initialize(values);
    }
    array
}

#[pyclass(module = "msgspec_flatbuffers._native", frozen)]
pub struct NativePlan {
    identity: Arc<()>,
    objects: Vec<ObjectWire>,
    by_name: HashMap<String, usize>,
    bound_types: OnceLock<BoundTypes>,
    model_subclass_cache: Mutex<HashMap<usize, BoundModelSubclass>>,
    dynamic_encoder: Option<Py<PyAny>>,
    nested_encoder: Option<Py<PyAny>>,
    model_decoder: Option<Py<PyAny>>,
    dynamic_registry: Py<PyAny>,
    uuid_type: Option<Py<PyType>>,
    decimal_type: Option<Py<PyType>>,
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

    fn sorted_table_map_values<'py>(
        &self,
        field: &FieldWire,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let target = self.target_object(field)?;
        let key_field = target
            .key_field()
            .ok_or_else(|| PyRuntimeError::new_err("keyed table has no key field"))?;
        let mapping = value.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "keyed table field {:?} requires a dict",
                field.name
            ))
        })?;
        let mut entries = Vec::with_capacity(mapping.len());
        for (mapping_key, item) in mapping {
            let key = TableKey::extract(key_field, &mapping_key)?;
            let model_key = item.getattr(key_field.name.as_str())?;
            if !mapping_key.eq(&model_key)? {
                return Err(PyValueError::new_err(format!(
                    "keyed table field {:?} key does not match {}.{}",
                    field.name, target.name, key_field.name
                )));
            }
            entries.push((key, item));
        }
        entries.sort_by(|(left, _), (right, _)| left.compare(right));
        if entries
            .windows(2)
            .any(|items| items[0].0.compare(&items[1].0) == Ordering::Equal)
        {
            return Err(PyValueError::new_err(format!(
                "keyed table field {:?} has duplicate encoded keys",
                field.name
            )));
        }
        Ok(PyList::new(value.py(), entries.into_iter().map(|(_, item)| item))?.into_any())
    }

    fn bound_type<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyType>> {
        self.bound_types
            .get()
            .and_then(|types| types.by_name.get(name))
            .map(|value| value.bind(py).clone())
            .ok_or_else(|| PyRuntimeError::new_err(format!("native type {name:?} is not bound")))
    }

    fn construct_uuid_text(&self, py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
        self.uuid_type
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("UUID type is not loaded"))?
            .bind(py)
            .call1((value,))
            .map(Bound::into_any)
            .map(Bound::unbind)
    }

    fn construct_uuid_bytes(&self, py: Python<'_>, value: &[u8]) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("bytes", PyBytes::new(py, value))?;
        self.uuid_type
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("UUID type is not loaded"))?
            .bind(py)
            .call((), Some(&kwargs))
            .map(Bound::into_any)
            .map(Bound::unbind)
    }

    fn construct_decimal(&self, py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
        self.decimal_type
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Decimal type is not loaded"))?
            .bind(py)
            .call1((value,))
            .map(Bound::into_any)
            .map(Bound::unbind)
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
        let bytes = &data[start..start + byte_length];
        macro_rules! array {
            ($ty:ty) => {
                initialized_numpy_array::<$ty>(py, length, |values| {
                    cast_slice_mut(values).copy_from_slice(bytes);
                })
                .into_any()
            };
        }
        let array = match scalar {
            ScalarKind::Bool => initialized_numpy_array(py, length, |values| {
                for (value, &byte) in values.iter_mut().zip(bytes) {
                    *value = byte != 0;
                }
            })
            .into_any(),
            ScalarKind::Int8 => array!(i8),
            ScalarKind::Uint8 => array!(u8),
            ScalarKind::Int16 => array!(i16),
            ScalarKind::Uint16 => array!(u16),
            ScalarKind::Int32 => array!(i32),
            ScalarKind::Uint32 => array!(u32),
            ScalarKind::Int64 => array!(i64),
            ScalarKind::Uint64 => array!(u64),
            ScalarKind::Float32 => array!(f32),
            ScalarKind::Float64 => array!(f64),
        };
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
        let model_type = match dynamic_overrides {
            Some(overrides) => {
                overrides.call_method1("get", (generated_type.clone(), generated_type))?
            }
            None => generated_type,
        };
        let model = self.decode_model_value(py, &bytes, &model_type, dynamic_overrides)?;
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

    fn encode_nested_value<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.nested_encoder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("native nested encoder is not loaded"))?
            .bind(value.py())
            .call1((value,))
    }

    fn decode_model_value<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyBytes>,
        model_type: &Bound<'py, PyAny>,
        dynamic_overrides: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("type", model_type)?;
        if let Some(overrides) = dynamic_overrides {
            kwargs.set_item("dynamic_overrides", overrides)?;
        }
        self.model_decoder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("native model decoder is not loaded"))?
            .bind(py)
            .call((data,), Some(&kwargs))
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
                let mut value = model.getattr(field.name.as_str())?;
                if !value.is_none()
                    && field.kind == FieldKind::VectorTable
                    && self.target_object(field)?.key_field().is_some()
                {
                    value = self.sorted_table_map_values(field, &value)?;
                }
                Ok(FieldState {
                    value,
                    offset: None,
                    discriminator: None,
                    prepared: None,
                    scalar: None,
                })
            })
            .collect()
    }

    fn estimate_table(
        &self,
        object: &ObjectWire,
        model: &Bound<'_, PyAny>,
        budget: &mut SizeEstimateBudget,
    ) -> PyResult<usize> {
        if !budget.consume(table_estimate_work(object)) {
            return Ok(DEFAULT_TABLE_SIZE_ESTIMATE);
        }
        let mut fields = self.load_fields(object, model)?;
        self.estimate_table_fields(object, &mut fields, budget)
    }

    fn estimate_table_fields(
        &self,
        object: &ObjectWire,
        fields: &mut [FieldState<'_>],
        budget: &mut SizeEstimateBudget,
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
                    self.estimate_table(target, value, budget)?
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
                    let sample_count = budget.take_samples(length, 1);
                    let strings = sampled_sequence_size(
                        value,
                        length,
                        sample_count,
                        sample_seed(object, field, length),
                        DEFAULT_STRING_SIZE_ESTIMATE,
                        |item| {
                            let length = item.cast::<PyString>()?.to_str()?.len();
                            Ok(aligned_payload_size(length.saturating_add(5), 4))
                        },
                    )?;
                    offset_vector_size(length).saturating_add(strings)
                }
                FieldKind::VectorTable => {
                    size = size.saturating_add(7);
                    let length = sequence_len(value)?;
                    let target = self.target_object(field)?;
                    let sample_count =
                        budget.available_samples(length, table_estimate_work(target));
                    let tables = sampled_sequence_size(
                        value,
                        length,
                        sample_count,
                        sample_seed(object, field, length),
                        DEFAULT_TABLE_SIZE_ESTIMATE,
                        |item| self.estimate_table(target, &item, budget),
                    )?;
                    offset_vector_size(length).saturating_add(tables)
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
                    let data = self.encode_nested_value(value)?;
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
                    self.estimate_table(&self.objects[arm.target_index], value, budget)?
                }
                FieldKind::UnionVector => {
                    size = size.saturating_add(7);
                    if let Some(type_slot) = field.type_slot {
                        include_slot(&mut highest_slot, type_slot);
                        size = size.saturating_add(7);
                    }
                    let length = sequence_len(value)?;
                    let width = scalar_size(field.type_scalar)?;
                    let sample_count = budget.available_samples(length, 1);
                    let tables = sampled_sequence_size(
                        value,
                        length,
                        sample_count,
                        sample_seed(object, field, length),
                        DEFAULT_TABLE_SIZE_ESTIMATE,
                        |item| {
                            if item.is_none() {
                                return Ok(0);
                            }
                            let arm = self.union_arm(field, &item)?;
                            self.estimate_table(&self.objects[arm.target_index], &item, budget)
                        },
                    )?;
                    offset_vector_size(length)
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
                FieldKind::Uuid | FieldKind::Decimal | FieldKind::Fallback => {
                    return Err(PyValueError::new_err(
                        "serde fallback fields cannot be encoded as FlatBuffers",
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

    // Keep estimation code out of the hot explicit-capacity path in `pack`.
    #[inline(never)]
    fn estimate_initial_capacity(
        &self,
        root_object: &ObjectWire,
        root_fields: &mut [FieldState<'_>],
        identifier: Option<&str>,
        size_prefixed: bool,
    ) -> PyResult<usize> {
        let mut budget = SizeEstimateBudget::new();
        let finish_size = 4usize
            .saturating_add(identifier.map_or(0, |_| 4))
            .saturating_add(if size_prefixed { 4 } else { 0 });
        let estimate = self
            .estimate_table_fields(root_object, root_fields, &mut budget)?
            .saturating_add(finish_size);
        Ok(estimated_capacity(estimate))
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
                        _ => self.encode_nested_value(value)?,
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
                FieldKind::Uuid | FieldKind::Decimal | FieldKind::Fallback => {
                    return Err(PyValueError::new_err(
                        "serde fallback fields cannot be encoded as FlatBuffers",
                    ));
                }
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
                FieldKind::Uuid | FieldKind::Decimal | FieldKind::Fallback => {
                    return Err(PyValueError::new_err(
                        "serde fallback fields cannot be encoded as FlatBuffers",
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

    fn finish_map(&mut self, value_start: usize, object_index: usize) -> PyResult<()> {
        let object = &self.plan.objects[object_index];
        let key_field = object
            .key_field()
            .ok_or_else(|| PyRuntimeError::new_err("keyed table has no key field"))?;
        let mapping = PyDict::new(self.py);
        let mut previous: Option<TableKey> = None;
        for value in &self.values[value_start..] {
            let key_value = value.bind(self.py).getattr(key_field.name.as_str())?;
            let key = TableKey::extract(key_field, &key_value).map_err(|_| {
                self.plan.invalid_error(
                    self.py,
                    format!("keyed table {} has an invalid key", object.name),
                )
            })?;
            if previous
                .as_ref()
                .is_some_and(|previous| previous.compare(&key) != Ordering::Less)
            {
                return Err(self.plan.invalid_error(
                    self.py,
                    format!(
                        "keyed table vector for {} is not strictly sorted",
                        object.name
                    ),
                ));
            }
            mapping.set_item(key_value, value.bind(self.py))?;
            previous = Some(key);
        }
        self.values.truncate(value_start);
        self.values.push(mapping.into_any().unbind());
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
                        if self.plan.objects[vector.object_index].key_field().is_some() {
                            self.finish_map(vector.result_start, vector.object_index)?;
                        } else {
                            self.finish_list(vector.result_start)?;
                        }
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
            FieldKind::Uuid | FieldKind::Decimal | FieldKind::Fallback => {
                return Err(PyValueError::new_err(
                    "serde fallback fields cannot be decoded as FlatBuffers",
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
        let value = self.plan.decode_model_value(
            self.py,
            &bytes,
            model_type.as_any(),
            self.context.dynamic_overrides,
        )?;
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
        if wire.version != 2 {
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
            let mut key_fields = object
                .fields
                .iter()
                .enumerate()
                .filter(|(_, field)| field.key);
            object.key_field_index = key_fields.next().map(|(index, _)| index);
            if key_fields.next().is_some() {
                return Err(PyValueError::new_err(format!(
                    "native table {:?} has multiple key fields",
                    object.name
                )));
            }
            if let Some(key_field) = object.key_field()
                && (object.is_struct
                    || !matches!(key_field.kind, FieldKind::Scalar | FieldKind::String))
            {
                return Err(PyValueError::new_err(format!(
                    "native key field on {:?} must be a table scalar or string",
                    object.name
                )));
            }
        }
        let has_dynamic = wire
            .objects
            .iter()
            .flat_map(|object| &object.fields)
            .any(|field| field.kind == FieldKind::Dynamic);
        let has_nested = wire
            .objects
            .iter()
            .flat_map(|object| &object.fields)
            .any(|field| field.kind == FieldKind::Nested);
        let has_uuid = wire
            .objects
            .iter()
            .flat_map(|object| &object.fields)
            .any(|field| field.kind == FieldKind::Uuid);
        let has_decimal = wire
            .objects
            .iter()
            .flat_map(|object| &object.fields)
            .any(|field| field.kind == FieldKind::Decimal);
        let dynamic = data.py().import("msgspec_flatbuffers._dynamic")?;
        let dynamic_encoder = if has_dynamic {
            Some(dynamic.getattr("encode_dynamic")?.unbind())
        } else {
            None
        };
        let flatbuffer = if has_nested || has_dynamic {
            Some(data.py().import("msgspec_flatbuffers._flatbuffer")?)
        } else {
            None
        };
        let nested_encoder = match &flatbuffer {
            Some(module) if has_nested => Some(module.getattr("encode")?.unbind()),
            _ => None,
        };
        let model_decoder = match &flatbuffer {
            Some(module) => Some(module.getattr("decode")?.unbind()),
            None => None,
        };
        let dynamic_registry = dynamic.getattr("dynamic_types")?.unbind();
        let uuid_type = if has_uuid {
            Some(
                data.py()
                    .import("uuid")?
                    .getattr("UUID")?
                    .cast_into::<PyType>()?
                    .unbind(),
            )
        } else {
            None
        };
        let decimal_type = if has_decimal {
            Some(
                data.py()
                    .import("decimal")?
                    .getattr("Decimal")?
                    .cast_into::<PyType>()?
                    .unbind(),
            )
        } else {
            None
        };
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
            nested_encoder,
            model_decoder,
            dynamic_registry,
            uuid_type,
            decimal_type,
            buffer_bounds_error,
            invalid_buffer_error,
        })
    }

    fn bind_types(&self, types: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut by_pointer = HashMap::with_capacity(types.len());
        let mut by_name = HashMap::with_capacity(types.len());
        let mut serde_objects: Vec<Option<SerdeObject>> =
            (0..self.objects.len()).map(|_| None).collect();
        let mut serde_supported = true;
        for (name, bound_type) in types.iter() {
            let name = name.extract::<String>()?;
            let bound_type = bound_type.cast::<PyType>()?;
            if let Some(object_index) = self.by_name.get(&name) {
                by_pointer.insert(bound_type.as_ptr() as usize, *object_index);
                let object = &self.objects[*object_index];
                let mut fields = Vec::with_capacity(object.serde_fields.len());
                for serde_field in &object.serde_fields {
                    let Some(object_field_index) = object
                        .fields
                        .iter()
                        .position(|field| field.name == serde_field.attr_name)
                    else {
                        serde_supported = false;
                        continue;
                    };
                    fields.push(SerdeField {
                        object_field_index,
                        attr_name: PyString::intern(bound_type.py(), &serde_field.attr_name)
                            .unbind(),
                        encode_name: serde_field.encode_name.clone(),
                    });
                }
                if object.fields.iter().any(|field| {
                    matches!(field.kind, FieldKind::Dynamic | FieldKind::VectorByte)
                        || (field.kind == FieldKind::VectorTable
                            && field
                                .target_index
                                .is_some_and(|index| self.objects[index].key_field().is_some()))
                }) {
                    serde_supported = false;
                }
                let tag = match (&object.serde_tag_field, &object.serde_tag) {
                    (Some(field), Some(value)) => Some((field.clone(), value.clone())),
                    (None, None) => None,
                    _ => {
                        return Err(PyValueError::new_err(
                            "native serde tag metadata is incomplete",
                        ));
                    }
                };
                let keyword_names = PyTuple::new(
                    bound_type.py(),
                    fields
                        .iter()
                        .map(|field| field.attr_name.clone_ref(bound_type.py())),
                )?
                .unbind();
                serde_objects[*object_index] = Some(SerdeObject {
                    fields,
                    keyword_names,
                    tag,
                });
            }
            by_name.insert(name, bound_type.clone().unbind());
        }
        if serde_objects.iter().any(Option::is_none) {
            serde_supported = false;
        }
        self.bound_types
            .set(BoundTypes {
                by_pointer,
                by_name,
                serde_objects,
                serde_supported,
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (root, model, is_json, *, fallback_encoder, order=None, decimal_format="string", uuid_format="canonical"))]
    fn encode_serde<'py>(
        &self,
        py: Python<'py>,
        root: &str,
        model: &Bound<'py, PyAny>,
        is_json: bool,
        fallback_encoder: &Bound<'py, PyAny>,
        order: Option<&str>,
        decimal_format: &str,
        uuid_format: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let root = self.object(root)?;
        let object_index = self.model_object_index(model)?;
        if object_index != root.index {
            return Err(PyTypeError::new_err(
                "serde model does not match the requested generated root",
            ));
        }
        let bound_types = self
            .bound_types
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
        if !bound_types.serde_supported {
            return Err(PyNotImplementedError::new_err(
                "generated model graph requires the msgspec serde fallback",
            ));
        }
        let context = SerdeEncodeContext {
            plan: self,
            fallback_encoder,
            sorted: order == Some("sorted"),
            decimal_as_number: decimal_format == "number",
            uuid_format: match uuid_format {
                "canonical" => SerdeUuidFormat::Canonical,
                "hex" => SerdeUuidFormat::Hex,
                "bytes" if !is_json => SerdeUuidFormat::Bytes,
                _ => return Err(PyValueError::new_err("unsupported UUID format")),
            },
        };
        let encoded = if is_json {
            let value = SerdeModel {
                context,
                model: model.clone(),
                object_index,
            };
            sonic_rs::to_vec(&value)
                .map_err(|error| PyTypeError::new_err(format!("cannot encode JSON: {error}")))?
        } else {
            let mut output = Vec::new();
            write_messagepack_model(context, model, object_index, &mut output)?;
            output
        };
        Ok(PyBytes::new(py, &encoded))
    }

    #[pyo3(signature = (root, buffer, is_json, *, strict=true, fallback_decoders))]
    fn decode_serde(
        &self,
        py: Python<'_>,
        root: &str,
        buffer: &Bound<'_, PyAny>,
        is_json: bool,
        strict: bool,
        fallback_decoders: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let root = self.object(root)?;
        let bound_types = self
            .bound_types
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
        if !bound_types.serde_supported {
            return Err(PyNotImplementedError::new_err(
                "generated model graph requires the msgspec serde fallback",
            ));
        }
        with_input_bytes(buffer, |data| {
            let seed = SerdeModelSeed {
                context: SerdeDecodeContext {
                    plan: self,
                    py,
                    is_json,
                    strict,
                    fallback_decoders,
                },
                choice: SerdeModelChoice::Known(root.index),
            };
            if is_json {
                let mut deserializer = sonic_rs::Deserializer::from_slice(data);
                let value = seed.deserialize(&mut deserializer).map_err(|error| {
                    PyValueError::new_err(format!("cannot decode JSON: {error}"))
                })?;
                deserializer.end().map_err(|error| {
                    PyValueError::new_err(format!("cannot decode JSON: {error}"))
                })?;
                Ok(value)
            } else {
                let mut deserializer = rmp_serde::Deserializer::new(data);
                let value = seed.deserialize(&mut deserializer).map_err(|error| {
                    PyValueError::new_err(format!("cannot decode MessagePack: {error}"))
                })?;
                if !deserializer.get_ref().is_empty() {
                    return Err(PyValueError::new_err(
                        "MessagePack document contains trailing data",
                    ));
                }
                Ok(value)
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
        if initial_size > FLATBUFFERS_MAX_BUFFER_SIZE {
            return Err(PyValueError::new_err(
                "initial FlatBuffer size exceeds the 2 GiB format limit",
            ));
        }
        let root_object = self.object(root)?;
        let mut root_fields = self.load_fields(root_object, model)?;
        let capacity = if initial_size > 0 {
            initial_size
        } else {
            self.estimate_initial_capacity(
                root_object,
                &mut root_fields,
                identifier,
                size_prefixed,
            )?
        };
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
    use std::collections::HashSet;

    use super::{
        BufferEndian, ScalarKind, estimated_capacity, scalar_format, stratified_sample_index,
    };

    #[test]
    fn stratified_samples_are_distinct_and_cover_endpoints() {
        for length in 1..100 {
            for count in 1..=length.min(6) {
                let indices = (0..count)
                    .map(|index| stratified_sample_index(length, count, index, 42))
                    .collect::<Vec<_>>();
                assert!(indices.iter().all(|&index| index < length));
                assert_eq!(indices.iter().copied().collect::<HashSet<_>>().len(), count);
                if count > 1 {
                    assert_eq!(indices[0], 0);
                    assert_eq!(indices[count - 1], length - 1);
                }
            }
        }
    }

    #[test]
    fn estimates_receive_one_percent_headroom() {
        assert_eq!(estimated_capacity(16), 17);
        assert_eq!(estimated_capacity(16_000), 16_160);
    }

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
