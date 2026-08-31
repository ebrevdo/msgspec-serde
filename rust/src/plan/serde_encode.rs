use super::*;

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

enum SerdeModelEntry<'a, 'py> {
    Tag {
        name: &'a str,
        value: &'a str,
    },
    Field {
        metadata: &'a SerdeField,
        wire: &'a FieldWire,
        value: Bound<'py, PyAny>,
    },
}

impl SerdeModelEntry<'_, '_> {
    fn name(&self) -> &str {
        match self {
            Self::Tag { name, .. } => name,
            Self::Field { metadata, .. } => &metadata.encode_name,
        }
    }
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

fn serialize_scalar_list<S>(
    value: &Bound<'_, PyAny>,
    scalar: ScalarKind,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let values = value.cast::<PyList>().map_err(S::Error::custom)?;
    let mut sequence = serializer.serialize_seq(Some(values.len()))?;
    for value in values {
        sequence.serialize_element(&SerdeScalarValue {
            value: &value,
            scalar,
        })?;
    }
    sequence.end()
}

fn serialize_string_list<S>(value: &Bound<'_, PyAny>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let values = value.cast::<PyList>().map_err(S::Error::custom)?;
    let mut sequence = serializer.serialize_seq(Some(values.len()))?;
    for value in values {
        sequence.serialize_element(value.extract::<&str>().map_err(S::Error::custom)?)?;
    }
    sequence.end()
}

fn serialize_model_list<S>(
    context: SerdeEncodeContext<'_, '_>,
    value: &Bound<'_, PyAny>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let values = value.cast::<PyList>().map_err(S::Error::custom)?;
    let mut sequence = serializer.serialize_seq(Some(values.len()))?;
    for model in values {
        if model.is_none() {
            sequence.serialize_element(&Option::<u8>::None)?;
            continue;
        }
        let object_index = context
            .plan
            .model_object_index(&model)
            .map_err(S::Error::custom)?;
        sequence.serialize_element(&SerdeModel {
            context,
            model,
            object_index,
        })?;
    }
    sequence.end()
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
        Value::String(default) => {
            let default = default
                .as_str()
                .ok_or_else(|| PyTypeError::new_err("serde default is not valid UTF-8"))?;
            Ok(value.extract::<&str>()? == default)
        }
        Value::Binary(default) => Ok(value.extract::<Vec<u8>>()? == *default),
        _ => Err(PyTypeError::new_err("unsupported native serde default")),
    }
}

fn serde_model_entries<'a, 'py>(
    plan: &'a NativePlan,
    model: &Bound<'py, PyAny>,
    object_index: usize,
    sorted: bool,
) -> PyResult<Vec<SerdeModelEntry<'a, 'py>>> {
    let metadata = serde_object(plan, object_index)?;
    let object = &plan.objects[object_index];
    let mut entries =
        Vec::with_capacity(metadata.fields.len() + usize::from(metadata.tag.is_some()));
    if let Some((name, value)) = &metadata.tag {
        entries.push(SerdeModelEntry::Tag { name, value });
    }
    for metadata in &metadata.fields {
        let wire = &object.fields[metadata.object_field_index];
        let value = model.getattr(metadata.attr_name.bind(model.py()))?;
        if !serde_field_is_default(wire, &value)? {
            entries.push(SerdeModelEntry::Field {
                metadata,
                wire,
                value,
            });
        }
    }
    if sorted {
        entries.sort_by(|left, right| left.name().cmp(right.name()));
    }
    Ok(entries)
}

impl Serialize for SerdeModel<'_, '_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let entries = serde_model_entries(
            self.context.plan,
            &self.model,
            self.object_index,
            self.context.sorted,
        )
        .map_err(S::Error::custom)?;
        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for entry in entries {
            match entry {
                SerdeModelEntry::Tag { name, value } => map.serialize_entry(name, value)?,
                SerdeModelEntry::Field {
                    metadata,
                    wire,
                    value,
                } => map.serialize_entry(
                    &metadata.encode_name,
                    &SerdeFieldValue {
                        context: self.context,
                        field: wire,
                        value,
                    },
                )?,
            }
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
                    serialize_scalar_list(&self.value, scalar, serializer)
                } else {
                    serialize_numpy_field(&self.value, scalar, serializer)
                }
            }
            FieldKind::VectorString => serialize_string_list(&self.value, serializer),
            FieldKind::VectorTable
            | FieldKind::VectorStruct
            | FieldKind::ArrayStruct
            | FieldKind::UnionVector => serialize_model_list(self.context, &self.value, serializer),
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

fn write_messagepack_array_len(
    output: &mut Vec<u8>,
    len: usize,
    too_large: &'static str,
) -> PyResult<()> {
    let len = u32::try_from(len).map_err(|_| PyValueError::new_err(too_large))?;
    rmp::encode::write_array_len(output, len)
        .map(|_| ())
        .map_err(msgpack_error)
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
                write_messagepack_array_len(
                    output,
                    values.len(),
                    "numeric vector is too large for MessagePack",
                )?;
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
    let entries = serde_model_entries(context.plan, model, object_index, context.sorted)?;
    rmp::encode::write_map_len(
        output,
        u32::try_from(entries.len())
            .map_err(|_| PyValueError::new_err("generated model has too many fields"))?,
    )
    .map_err(msgpack_error)?;
    for entry in entries {
        match entry {
            SerdeModelEntry::Tag { name, value } => {
                rmp::encode::write_str(output, name).map_err(msgpack_error)?;
                rmp::encode::write_str(output, value).map_err(msgpack_error)?;
            }
            SerdeModelEntry::Field {
                metadata,
                wire,
                value,
            } => {
                rmp::encode::write_str(output, &metadata.encode_name).map_err(msgpack_error)?;
                write_messagepack_field(context, wire, &value, output)?;
            }
        }
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

fn write_messagepack_python_int(output: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(value) = value.extract::<i64>() {
        rmp::encode::write_sint(output, value).map_err(msgpack_error)?;
    } else {
        rmp::encode::write_uint(output, value.extract()?).map_err(msgpack_error)?;
    }
    Ok(())
}

fn write_messagepack_uuid(
    output: &mut Vec<u8>,
    value: &Bound<'_, PyAny>,
    format: SerdeUuidFormat,
) -> PyResult<()> {
    match format {
        SerdeUuidFormat::Canonical | SerdeUuidFormat::Hex => {
            rmp::encode::write_str(output, &uuid_text(value, format)?).map_err(msgpack_error)?;
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
    }
    Ok(())
}

fn write_messagepack_decimal(
    output: &mut Vec<u8>,
    value: &Bound<'_, PyAny>,
    as_number: bool,
) -> PyResult<()> {
    if as_number {
        rmp::encode::write_f64(output, value.extract()?).map_err(msgpack_error)?;
    } else {
        rmp::encode::write_str(output, value.str()?.to_str()?).map_err(msgpack_error)?;
    }
    Ok(())
}

fn write_messagepack_scalar_list(
    output: &mut Vec<u8>,
    value: &Bound<'_, PyAny>,
    scalar: ScalarKind,
) -> PyResult<()> {
    let values = value.cast::<PyList>()?;
    write_messagepack_array_len(output, values.len(), "enum vector is too large")?;
    for value in values {
        write_messagepack_scalar(output, scalar, &value)?;
    }
    Ok(())
}

fn write_messagepack_string_list(output: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let values = value.cast::<PyList>()?;
    write_messagepack_array_len(output, values.len(), "string vector is too large")?;
    for value in values {
        rmp::encode::write_str(output, value.extract()?).map_err(msgpack_error)?;
    }
    Ok(())
}

fn write_messagepack_model_list(
    context: SerdeEncodeContext<'_, '_>,
    output: &mut Vec<u8>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let models = value.cast::<PyList>()?;
    write_messagepack_array_len(output, models.len(), "object vector is too large")?;
    for model in models {
        if model.is_none() {
            rmp::encode::write_nil(output).map_err(msgpack_error)?;
            continue;
        }
        write_messagepack_model(
            context,
            &model,
            context.plan.model_object_index(&model)?,
            output,
        )?;
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
            write_messagepack_python_int(output, value)?;
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
        FieldKind::Uuid => write_messagepack_uuid(output, value, context.uuid_format)?,
        FieldKind::Decimal => {
            write_messagepack_decimal(output, value, context.decimal_as_number)?;
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
                write_messagepack_scalar_list(output, value, scalar)?;
            } else {
                write_messagepack_array(value, scalar, output)?;
            }
        }
        FieldKind::VectorString => write_messagepack_string_list(output, value)?,
        FieldKind::VectorTable
        | FieldKind::VectorStruct
        | FieldKind::ArrayStruct
        | FieldKind::UnionVector => write_messagepack_model_list(context, output, value)?,
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

#[allow(clippy::too_many_arguments)]
pub(super) fn encode<'py>(
    plan: &NativePlan,
    py: Python<'py>,
    root: &str,
    model: &Bound<'py, PyAny>,
    is_json: bool,
    fallback_encoder: &Bound<'py, PyAny>,
    order: Option<&str>,
    decimal_format: &str,
    uuid_format: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let root = plan.object(root)?;
    let object_index = plan.model_object_index(model)?;
    if object_index != root.index {
        return Err(PyTypeError::new_err(
            "serde model does not match the requested generated root",
        ));
    }
    let bound_types = plan
        .bound_types
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("native model types are not bound"))?;
    if !bound_types.serde_supported {
        return Err(PyNotImplementedError::new_err(
            "generated model graph requires the msgspec serde fallback",
        ));
    }
    let context = SerdeEncodeContext {
        plan,
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
