use super::buffer::{
    NativeBuffer, buffer_byte_length, check_byte_buffer, checked_byte_buffer, extract_bytes,
    immutable_buffer_subslice, nonnegative_usize, validate_identifier,
};
use super::*;

pub(super) enum TableKey {
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float32(f32),
    Float64(f64),
    String(String),
}

impl TableKey {
    pub(super) fn extract(field: &FieldWire, value: &Bound<'_, PyAny>) -> PyResult<Self> {
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

    pub(super) fn compare(&self, other: &Self) -> Ordering {
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

fn is_offset_field(kind: FieldKind) -> bool {
    matches!(
        kind,
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
    )
}

fn skip_absent_field(field: &FieldWire, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !value.is_none() {
        return Ok(false);
    }
    if field.required {
        return Err(PyTypeError::new_err(format!(
            "required field {:?} is absent",
            field.name
        )));
    }
    Ok(true)
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

fn push_discriminator(
    builder: &mut FlatBufferBuilder<'_>,
    field: &FieldWire,
    discriminator: Option<Discriminator>,
) -> PyResult<()> {
    match field.kind {
        FieldKind::Dynamic => {
            let Some(Discriminator::Offset(type_offset)) = discriminator else {
                return Err(PyTypeError::new_err("dynamic FlatBuffer has no type tag"));
            };
            let type_slot = vtable_offset(
                field
                    .type_slot
                    .ok_or_else(|| PyValueError::new_err("dynamic field has no type slot"))?,
            );
            builder.push_slot_always(type_slot, type_offset);
        }
        FieldKind::Union => {
            let Some(Discriminator::Union(tag)) = discriminator else {
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
            let Some(Discriminator::Offset(type_offset)) = discriminator else {
                return Err(PyTypeError::new_err(
                    "union vector has no discriminator vector",
                ));
            };
            let type_slot = vtable_offset(
                field
                    .type_slot
                    .ok_or_else(|| PyValueError::new_err("union vector has no type slot"))?,
            );
            builder.push_slot_always(type_slot, type_offset);
        }
        _ => {}
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

impl NativePlan {
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

    pub(super) fn decode_model_value<'py>(
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

    fn build_offset_field<'fbb, 'py>(
        &self,
        builder: &mut FlatBufferBuilder<'fbb>,
        field: &FieldWire,
        value: &Bound<'py, PyAny>,
        prepared: Option<PreparedValue<'py>>,
        vector_length_patches: &mut Vec<(u32, u32)>,
    ) -> PyResult<(AnyOffset, Option<Discriminator>)> {
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
                        let allowed_prefix = field.allowed_prefix.as_deref().ok_or_else(|| {
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
        Ok((offset, discriminator))
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
            if !is_offset_field(field.kind) {
                continue;
            }
            let state = &mut states[index];
            let prepared = state.prepared.take();
            let value = &state.value;
            if skip_absent_field(field, value)? {
                continue;
            }
            let (offset, discriminator) =
                self.build_offset_field(builder, field, value, prepared, vector_length_patches)?;
            state.offset = Some(offset);
            state.discriminator = discriminator;
        }

        let table = builder.start_table();
        for (field, state) in object.fields.iter().zip(&states) {
            let slot = vtable_offset(field.slot);
            if let Some(offset) = state.offset {
                builder.push_slot_always(slot, offset);
                push_discriminator(builder, field, state.discriminator)?;
                continue;
            }
            if field.kind != FieldKind::Scalar && skip_absent_field(field, &state.value)? {
                continue;
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

pub(super) fn pack<'py>(
    plan: &NativePlan,
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
    let root_object = plan.object(root)?;
    let mut root_fields = plan.load_fields(root_object, model)?;
    let capacity = if initial_size > 0 {
        initial_size
    } else {
        plan.estimate_initial_capacity(root_object, &mut root_fields, identifier, size_prefixed)?
    };
    if capacity > FLATBUFFERS_MAX_BUFFER_SIZE {
        return Err(PyValueError::new_err(
            "estimated FlatBuffer size exceeds the 2 GiB format limit",
        ));
    }
    let mut builder = FlatBufferBuilder::with_capacity(capacity);
    let mut vector_length_patches = Vec::new();
    let root = plan.build_table(
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
