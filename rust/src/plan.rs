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

mod buffer;
mod decoding;
mod encoding;
mod serde_decode;
mod serde_encode;

use buffer::NativeBuffer;

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

    fn resolve_references(
        &mut self,
        object_index: usize,
        object_indices: &HashMap<String, usize>,
    ) -> PyResult<()> {
        self.index = object_index;
        for field in &mut self.fields {
            if let Some(target) = field.target.take() {
                field.target_index =
                    Some(object_indices.get(&target).copied().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "native field target {target:?} does not exist"
                        ))
                    })?);
            }
            for arm in &mut field.arms {
                arm.target_index = object_indices.get(&arm.target).copied().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "native union target {:?} does not exist",
                        arm.target
                    ))
                })?;
                arm.target.clear();
            }
        }
        Ok(())
    }

    fn resolve_key_field(&mut self) -> PyResult<()> {
        let mut key_fields = self
            .fields
            .iter()
            .enumerate()
            .filter(|(_, field)| field.key);
        self.key_field_index = key_fields.next().map(|(index, _)| index);
        if key_fields.next().is_some() {
            return Err(PyValueError::new_err(format!(
                "native table {:?} has multiple key fields",
                self.name
            )));
        }
        if let Some(key_field) = self.key_field()
            && (self.is_struct || !matches!(key_field.kind, FieldKind::Scalar | FieldKind::String))
        {
            return Err(PyValueError::new_err(format!(
                "native key field on {:?} must be a table scalar or string",
                self.name
            )));
        }
        Ok(())
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

#[derive(Deserialize)]
struct ArmWire {
    tag: u64,
    target: String,
    #[serde(skip)]
    target_index: usize,
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

impl SerdeObject {
    fn for_bound_type(
        plan: &NativePlan,
        object: &ObjectWire,
        bound_type: &Bound<'_, PyType>,
    ) -> PyResult<(Self, bool)> {
        let py = bound_type.py();
        let mut fields = Vec::with_capacity(object.serde_fields.len());
        let mut supported = true;

        for serde_field in &object.serde_fields {
            let Some(object_field_index) = object
                .fields
                .iter()
                .position(|field| field.name == serde_field.attr_name)
            else {
                supported = false;
                continue;
            };
            fields.push(SerdeField {
                object_field_index,
                attr_name: PyString::intern(py, &serde_field.attr_name).unbind(),
                encode_name: serde_field.encode_name.clone(),
            });
        }

        if object.fields.iter().any(|field| {
            matches!(field.kind, FieldKind::Dynamic | FieldKind::VectorByte)
                || (field.kind == FieldKind::VectorTable
                    && field
                        .target_index
                        .is_some_and(|index| plan.objects[index].key_field().is_some()))
        }) {
            supported = false;
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
        let keyword_names =
            PyTuple::new(py, fields.iter().map(|field| field.attr_name.clone_ref(py)))?.unbind();

        Ok((
            Self {
                fields,
                keyword_names,
                tag,
            },
            supported,
        ))
    }
}

struct SerdeField {
    object_field_index: usize,
    attr_name: Py<PyString>,
    encode_name: String,
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

struct BoundModelSubclass {
    _type_owner: Py<PyType>,
    object_index: usize,
}

type ChildTypeKey = (usize, u16, usize);

#[pyclass(module = "msgspec_serde._native", frozen)]
pub struct NativeModelTypes {
    plan_identity: Arc<()>,
    root_index: usize,
    root_type: Py<PyType>,
    child_types: HashMap<ChildTypeKey, Py<PyType>>,
}

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
fn vtable_offset(slot: u16) -> VOffsetT {
    4 + slot * 2
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

#[pyclass(module = "msgspec_serde._native", frozen)]
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
        let object_index = *object_index;
        let generated_type = self.bound_type(py, &self.objects[object_index].name)?;
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
                object_index,
            },
        );
        Ok(object_index)
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
            object.resolve_references(object_index, &by_name)?;
            object.resolve_key_field()?;
        }
        let has_field_kind = |kind| {
            wire.objects
                .iter()
                .flat_map(|object| &object.fields)
                .any(|field| field.kind == kind)
        };
        let has_dynamic = has_field_kind(FieldKind::Dynamic);
        let has_nested = has_field_kind(FieldKind::Nested);
        let has_uuid = has_field_kind(FieldKind::Uuid);
        let has_decimal = has_field_kind(FieldKind::Decimal);
        let py = data.py();
        let dynamic = py.import("msgspec_serde._dynamic")?;
        let dynamic_encoder = if has_dynamic {
            Some(dynamic.getattr("encode_dynamic")?.unbind())
        } else {
            None
        };
        let flatbuffer = if has_nested || has_dynamic {
            Some(py.import("msgspec_serde._flatbuffer")?)
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
                py.import("uuid")?
                    .getattr("UUID")?
                    .cast_into::<PyType>()?
                    .unbind(),
            )
        } else {
            None
        };
        let decimal_type = if has_decimal {
            Some(
                py.import("decimal")?
                    .getattr("Decimal")?
                    .cast_into::<PyType>()?
                    .unbind(),
            )
        } else {
            None
        };
        let runtime = py.import("msgspec_serde._runtime")?;
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
                let (serde_object, object_supported) =
                    SerdeObject::for_bound_type(self, object, bound_type)?;
                serde_supported &= object_supported;
                serde_objects[*object_index] = Some(serde_object);
            }
            by_name.insert(name, bound_type.clone().unbind());
        }
        serde_supported &= serde_objects.iter().all(Option::is_some);
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
        decoding::unpack(
            self,
            py,
            root,
            buffer,
            identifier,
            offset,
            size_prefixed,
            check_identifier,
            model_types,
            dynamic_overrides,
        )
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
        decoding::unpack_view(
            self,
            py,
            object,
            buffer,
            offset,
            model_types,
            dynamic_overrides,
        )
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
        serde_encode::encode(
            self,
            py,
            root,
            model,
            is_json,
            fallback_encoder,
            order,
            decimal_format,
            uuid_format,
        )
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
        serde_decode::decode(self, py, root, buffer, is_json, strict, fallback_decoders)
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
        encoding::pack(
            self,
            py,
            root,
            model,
            identifier,
            size_prefixed,
            initial_size,
        )
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
