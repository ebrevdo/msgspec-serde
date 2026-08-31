use super::buffer::{nonnegative_usize, validate_identifier, with_input_bytes};
use super::encoding::TableKey;
use super::serde_decode::initialized_numpy_array;
use super::*;

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

impl NativePlan {
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

    pub(super) fn apply_enum(
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
        macro_rules! native_array {
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
            ScalarKind::Int8 => native_array!(i8),
            ScalarKind::Uint8 => native_array!(u8),
            ScalarKind::Int16 => native_array!(i16),
            ScalarKind::Uint16 => native_array!(u16),
            ScalarKind::Int32 => native_array!(i32),
            ScalarKind::Uint32 => native_array!(u32),
            ScalarKind::Int64 => native_array!(i64),
            ScalarKind::Uint64 => native_array!(u64),
            ScalarKind::Float32 => native_array!(f32),
            ScalarKind::Float64 => native_array!(f64),
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

    fn child_model_type_index(&mut self, task: FieldTask, target_index: usize) -> PyResult<usize> {
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

    fn push_object_vector_frame(
        &mut self,
        object_index: usize,
        model_type_index: usize,
        element: VectorElement,
        start: usize,
        length: usize,
    ) {
        self.frames
            .push(DecodeFrame::ObjectVector(ObjectVectorFrame {
                object_index,
                model_type_index,
                element,
                start,
                length,
                index: 0,
                result_start: self.values.len(),
            }));
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
        let mut previous_key: Option<TableKey> = None;
        for value in &self.values[value_start..] {
            let key_value = value.bind(self.py).getattr(key_field.name.as_str())?;
            let key = TableKey::extract(key_field, &key_value).map_err(|_| {
                self.plan.invalid_error(
                    self.py,
                    format!("keyed table {} has an invalid key", object.name),
                )
            })?;
            if previous_key
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
            previous_key = Some(key);
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

    fn push_missing_field(&mut self, task: FieldTask) -> PyResult<()> {
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
            self.push_missing_field(task)?;
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
            self.push_missing_field(task)?;
        }
        Ok(vector)
    }

    fn run(mut self) -> PyResult<Py<PyAny>> {
        while let Some(frame) = self.frames.pop() {
            match frame {
                DecodeFrame::Object(frame) => self.resume_object(frame)?,
                DecodeFrame::ObjectVector(frame) => self.resume_object_vector(frame)?,
                DecodeFrame::UnionVector(frame) => self.resume_union_vector(frame)?,
            }
        }
        if self.values.len() != 1 {
            return Err(PyRuntimeError::new_err(
                "native materializer did not produce one root model",
            ));
        }
        Ok(self.values.pop().expect("checked materializer result"))
    }

    fn resume_object(&mut self, mut frame: ObjectFrame) -> PyResult<()> {
        let task = frame.next_field;
        if task.field_index == self.plan.objects[task.object_index].fields.len() {
            return self.finish_model(task, frame.value_start);
        }

        frame.next_field.field_index += 1;
        self.frames.push(DecodeFrame::Object(frame));
        match frame.location {
            ObjectLocation::Table(table) => self.decode_table_field(task, table),
            ObjectLocation::Struct { offset } => self.decode_struct_field(task, offset),
        }
    }

    fn resume_object_vector(&mut self, mut frame: ObjectVectorFrame) -> PyResult<()> {
        if frame.index == frame.length {
            if self.plan.objects[frame.object_index].key_field().is_some() {
                return self.finish_map(frame.result_start, frame.object_index);
            }
            return self.finish_list(frame.result_start);
        }

        let index = frame.index;
        frame.index += 1;
        self.frames.push(DecodeFrame::ObjectVector(frame));
        match frame.element {
            VectorElement::Table => {
                let target = self.plan.offset_target(
                    self.py,
                    self.data,
                    frame.start + index * 4,
                    "table vector offset",
                )?;
                self.push_table(frame.object_index, frame.model_type_index, target)
            }
            VectorElement::Struct { stride } => self.push_struct(
                frame.object_index,
                frame.model_type_index,
                frame.start + index * stride,
            ),
        }
    }

    fn resume_union_vector(&mut self, mut frame: UnionVectorFrame) -> PyResult<()> {
        if frame.index == frame.length {
            return self.finish_list(frame.result_start);
        }

        let index = frame.index;
        frame.index += 1;
        self.frames.push(DecodeFrame::UnionVector(frame));
        let tag = self
            .plan
            .read_scalar_value(
                self.py,
                self.data,
                frame.type_start + index * frame.width,
                frame.type_scalar,
            )?
            .as_u64()?;
        let position = frame.value_start + index * 4;
        if tag == 0 {
            if self.plan.read_u32(self.py, self.data, position)? != 0 {
                return Err(self
                    .plan
                    .invalid_error(self.py, "union vector NONE has a payload"));
            }
            self.values.push(self.py.None());
            return Ok(());
        }

        let field = self.field(frame.field);
        let target_index = self.union_target(field, tag)?;
        let target =
            self.plan
                .offset_target(self.py, self.data, position, "union vector offset")?;
        let model_type_index = self.child_model_type_index(frame.field, target_index)?;
        self.push_table(target_index, model_type_index, target)
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
                let model_type_index = self.child_model_type_index(task, target_index)?;
                self.push_struct(target_index, model_type_index, target_offset)?;
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
                let model_type_index = self.child_model_type_index(task, target_index)?;
                self.push_object_vector_frame(
                    target_index,
                    model_type_index,
                    VectorElement::Struct { stride },
                    start,
                    length,
                );
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
                self.decode_scalar_field(task, table)?;
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
                let model_type_index = self.child_model_type_index(task, target_index)?;
                self.push_table(target_index, model_type_index, target)?;
            }
            FieldKind::Struct => {
                let target_index = self.target_index(task, "struct field")?;
                let target_size = self.plan.objects[target_index].byte_size;
                let Some(position) = self.position_or_missing(task, table, target_size)? else {
                    return Ok(());
                };
                let model_type_index = self.child_model_type_index(task, target_index)?;
                self.push_struct(target_index, model_type_index, position)?;
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
                self.decode_string_vector(task, table)?;
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

    fn decode_scalar_field(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let field = self.field(task);
        let scalar = field
            .scalar
            .ok_or_else(|| PyValueError::new_err("native scalar field has no scalar kind"))?;
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
        Ok(())
    }

    fn decode_string_vector(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let Some((start, length)) = self.vector_or_missing(task, table, 4)? else {
            return Ok(());
        };
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            values.push(
                self.plan
                    .decode_string_at(self.py, self.data, start + index * 4)?,
            );
        }
        self.values
            .push(PyList::new(self.py, values)?.into_any().unbind());
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
        let model_type_index = self.child_model_type_index(task, target_index)?;
        self.push_object_vector_frame(target_index, model_type_index, element, start, length);
        Ok(())
    }

    fn decode_nested(&mut self, task: FieldTask, table: TableInfo) -> PyResult<()> {
        let Some((start, length)) = self.vector_or_missing(task, table, 1)? else {
            return Ok(());
        };
        let target_index = self.target_index(task, "nested field")?;
        let model_type_index = self.child_model_type_index(task, target_index)?;
        let model_type = self.model_types[model_type_index].bind(self.py);
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
            return self.push_missing_field(task);
        }
        let position = value_position.ok_or_else(|| {
            self.plan
                .invalid_error(self.py, format!("union discriminator {tag} has no payload"))
        })?;
        let target_index = self.union_target(field, tag)?;
        let target = self
            .plan
            .offset_target(self.py, self.data, position, "union field offset")?;
        let model_type_index = self.child_model_type_index(task, target_index)?;
        self.push_table(target_index, model_type_index, target)
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
        let type_vector = self.plan.vector_info(
            self.py,
            self.data,
            table,
            usize::from(vtable_offset(type_slot)),
            width,
        )?;
        let value_vector = self
            .plan
            .vector_info(self.py, self.data, table, field.offset, 4)?;
        let (type_start, type_length, value_start, value_length) = match (type_vector, value_vector)
        {
            (None, None) => return self.push_missing_field(task),
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

#[allow(clippy::too_many_arguments)]
pub(super) fn unpack(
    plan: &NativePlan,
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
    let root = plan.object(root)?;
    let model_types = model_types.as_deref();
    let (model_type, context) = plan.prepare_decode(
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
        plan.decode_root(py, root, &model_type, context, data, options)
    })
}

pub(super) fn unpack_view(
    plan: &NativePlan,
    py: Python<'_>,
    object: &str,
    buffer: &Bound<'_, PyAny>,
    offset: isize,
    model_types: Option<PyRef<'_, NativeModelTypes>>,
    dynamic_overrides: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let offset = nonnegative_usize(offset, "offset")?;
    let object = plan.object(object)?;
    let model_types = model_types.as_deref();
    let (model_type, context) = plan.prepare_decode(
        py,
        object,
        model_types,
        dynamic_overrides,
        "native model types belong to a different view type",
    )?;
    with_input_bytes(buffer, |data| {
        if object.is_struct {
            plan.decode_struct_at(py, object, &model_type, context, data, offset)
        } else {
            plan.decode_table_at(py, object, &model_type, context, data, offset)
        }
    })
}
