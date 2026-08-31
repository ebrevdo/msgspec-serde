use super::buffer::with_input_bytes;
use super::*;

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

enum CanonicalInteger {
    Signed(i64),
    Unsigned(u64),
}

fn parse_canonical_integer(value: &str) -> Option<CanonicalInteger> {
    if let Ok(parsed) = value.parse::<i64>()
        && parsed.to_string() == value
    {
        return Some(CanonicalInteger::Signed(parsed));
    }
    if let Ok(parsed) = value.parse::<u64>()
        && parsed.to_string() == value
    {
        return Some(CanonicalInteger::Unsigned(parsed));
    }
    None
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

fn deserialize_json<'de, S>(seed: S, data: &'de [u8], error_context: &str) -> PyResult<Py<PyAny>>
where
    S: DeserializeSeed<'de, Value = Py<PyAny>>,
{
    let mut deserializer = sonic_rs::Deserializer::from_slice(data);
    let value = seed
        .deserialize(&mut deserializer)
        .map_err(|error| PyValueError::new_err(format!("{error_context}: {error}")))?;
    deserializer
        .end()
        .map_err(|error| PyValueError::new_err(format!("{error_context}: {error}")))?;
    Ok(value)
}

fn deserialize_messagepack<'de, S>(
    seed: S,
    data: &'de [u8],
    error_context: &str,
    trailing_data_error: &'static str,
) -> PyResult<Py<PyAny>>
where
    S: DeserializeSeed<'de, Value = Py<PyAny>>,
{
    let mut deserializer = rmp_serde::Deserializer::new(data);
    let value = seed
        .deserialize(&mut deserializer)
        .map_err(|error| PyValueError::new_err(format!("{error_context}: {error}")))?;
    if !deserializer.get_ref().is_empty() {
        return Err(PyValueError::new_err(trailing_data_error));
    }
    Ok(value)
}

fn deserialize_buffered_serde_field(
    context: SerdeDecodeContext<'_, '_>,
    field: &FieldWire,
    encoded: &[u8],
) -> PyResult<Py<PyAny>> {
    let seed = SerdeFieldSeed { context, field };
    if context.is_json {
        deserialize_json(seed, encoded, "cannot decode buffered JSON field")
    } else {
        deserialize_messagepack(
            seed,
            encoded,
            "cannot decode buffered MessagePack field",
            "buffered MessagePack field contains trailing data",
        )
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

fn buffer_next_serde_value<'de, A>(map: &mut A, is_json: bool) -> Result<Vec<u8>, A::Error>
where
    A: MapAccess<'de>,
{
    if is_json {
        let value = map.next_value::<sonic_rs::LazyValue<'de>>()?;
        Ok(value.as_raw_str().as_bytes().to_vec())
    } else {
        let value = map.next_value::<Value>()?;
        rmp_serde::to_vec(&value).map_err(A::Error::custom)
    }
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
                    buffered.push(BufferedSerdeField {
                        encode_name: key.into_owned(),
                        encoded: buffer_next_serde_value(&mut map, self.seed.context.is_json)?,
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
        match parse_canonical_integer(value) {
            Some(CanonicalInteger::Signed(parsed)) => {
                return parsed.into_py_any(self.py).map_err(E::custom);
            }
            Some(CanonicalInteger::Unsigned(parsed)) => {
                return parsed.into_py_any(self.py).map_err(E::custom);
            }
            None => {}
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
    match kind {
        ScalarKind::Float32 => return Ok(ScalarValue::Float32(value as f32)),
        ScalarKind::Float64 => return Ok(ScalarValue::Float64(value)),
        _ => {}
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
    match parse_canonical_integer(value) {
        Some(CanonicalInteger::Signed(parsed)) => return scalar_from_i64(kind, parsed),
        Some(CanonicalInteger::Unsigned(parsed)) => return scalar_from_u64(kind, parsed),
        None => {}
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

fn deserialize_fallback<'de, D>(
    deserializer: D,
    context: SerdeDecodeContext<'_, '_>,
    field: &FieldWire,
) -> Result<Py<PyAny>, D::Error>
where
    D: Deserializer<'de>,
{
    let fallback_id = field
        .fallback_id
        .as_deref()
        .ok_or_else(|| D::Error::custom("fallback field has no callback id"))?;
    let decoder = context
        .fallback_decoders
        .get_item(fallback_id)
        .map_err(D::Error::custom)?
        .ok_or_else(|| {
            D::Error::custom(format!(
                "serde fallback decoder {fallback_id:?} is not bound"
            ))
        })?;
    if context.is_json {
        let value = sonic_rs::LazyValue::deserialize(deserializer)?;
        decoder
            .call1((PyBytes::new(context.py, value.as_raw_str().as_bytes()),))
            .map(Bound::unbind)
            .map_err(D::Error::custom)
    } else {
        let value = Value::deserialize(deserializer)?;
        let encoded = rmp_serde::to_vec(&value).map_err(D::Error::custom)?;
        decoder
            .call1((PyBytes::new(context.py, &encoded),))
            .map(Bound::unbind)
            .map_err(D::Error::custom)
    }
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
            FieldKind::Fallback => deserialize_fallback(deserializer, self.context, self.field),
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
        let model_seed = SerdeModelSeed {
            context: self.seed.context,
            choice: self.seed.choice,
        };
        loop {
            let value = if self.seed.element_nullable {
                sequence.next_element_seed(OptionalSerdeModelSeed { seed: model_seed })?
            } else {
                sequence.next_element_seed(model_seed)?
            };
            let Some(value) = value else {
                break;
            };
            values.push(value);
        }
        validate_fixed_length::<A::Error>("object", self.seed.fixed_length, values.len())?;
        PyList::new(self.seed.context.py, values)
            .map(|value| value.into_any().unbind())
            .map_err(A::Error::custom)
    }
}

fn validate_fixed_length<E>(
    element_kind: &str,
    fixed_length: usize,
    actual_length: usize,
) -> Result<(), E>
where
    E: DeserializeError,
{
    if fixed_length != 0 && actual_length != fixed_length {
        return Err(E::custom(format!(
            "fixed {element_kind} array requires {fixed_length} values, got {actual_length}"
        )));
    }
    Ok(())
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
    validate_fixed_length::<E>("numeric", fixed_length, values.len())?;
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
    validate_fixed_length::<D::Error>("enum", field.fixed_length, raw.len())?;
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

pub(super) fn initialized_numpy_array<'py, T>(
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

pub(super) fn decode(
    plan: &NativePlan,
    py: Python<'_>,
    root: &str,
    buffer: &Bound<'_, PyAny>,
    is_json: bool,
    strict: bool,
    fallback_decoders: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let root = plan.object(root)?;
    let bound_types = plan
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
                plan,
                py,
                is_json,
                strict,
                fallback_decoders,
            },
            choice: SerdeModelChoice::Known(root.index),
        };
        if is_json {
            deserialize_json(seed, data, "cannot decode JSON")
        } else {
            deserialize_messagepack(
                seed,
                data,
                "cannot decode MessagePack",
                "MessagePack document contains trailing data",
            )
        }
    })
}
