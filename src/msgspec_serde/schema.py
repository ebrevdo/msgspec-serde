"""Represent a reflected FlatBuffers schema as normalized msgspec models.

Example:
    Describe a schema directly:

    >>> table = ObjectDefinition(name="example.Monster", fields=())
    >>> schema = Schema(objects=(table,), enums=(), root_table=table.name)
    >>> schema.root is table
    True
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

import msgspec


class BaseType(IntEnum):
    """Identify a scalar or aggregate type in a reflected schema.

    Attributes:
        NONE: No type.
        UTYPE: An unsigned union discriminator.
        BOOL: A Boolean value.
        BYTE: A signed 8-bit integer.
        UBYTE: An unsigned 8-bit integer.
        SHORT: A signed 16-bit integer.
        USHORT: An unsigned 16-bit integer.
        INT: A signed 32-bit integer.
        UINT: An unsigned 32-bit integer.
        LONG: A signed 64-bit integer.
        ULONG: An unsigned 64-bit integer.
        FLOAT: A 32-bit floating-point value.
        DOUBLE: A 64-bit floating-point value.
        STRING: A UTF-8 string.
        VECTOR: A variable-length vector.
        OBJECT: A table or struct.
        UNION: A union value.
        ARRAY: A fixed-length array.
        VECTOR64: A vector that uses 64-bit offsets.

    Example:
        Inspect the reflected type of an ``int`` field:

        >>> BaseType.INT.value
        7
    """

    NONE = 0
    UTYPE = 1
    BOOL = 2
    BYTE = 3
    UBYTE = 4
    SHORT = 5
    USHORT = 6
    INT = 7
    UINT = 8
    LONG = 9
    ULONG = 10
    FLOAT = 11
    DOUBLE = 12
    STRING = 13
    VECTOR = 14
    OBJECT = 15
    UNION = 16
    ARRAY = 17
    VECTOR64 = 18


class AdvancedFeature(IntFlag):
    """Describe advanced features used by a binary FlatBuffers schema.

    Attributes:
        NONE: No advanced features.
        ARRAYS: Fixed-length arrays.
        UNIONS: Advanced union features.
        OPTIONAL_SCALARS: Optional scalar fields.
        DEFAULT_VECTORS_AND_STRINGS: Nonempty vector or string defaults.

    Example:
        Combine feature flags:

        >>> features = AdvancedFeature.ARRAYS | AdvancedFeature.UNIONS
        >>> AdvancedFeature.ARRAYS in features
        True
    """

    NONE = 0
    ARRAYS = 1
    UNIONS = 2
    OPTIONAL_SCALARS = 4
    DEFAULT_VECTORS_AND_STRINGS = 8


class TypeReference(msgspec.Struct, frozen=True, kw_only=True):
    """Describe a field type and its referenced schema definition.

    Attributes:
        base_type: The field's scalar or aggregate type.
        element: The element type for a vector or array.
        index: The referenced object or enum index, or ``-1`` when absent.
        fixed_length: The element count for a fixed-length array.
        base_size: The encoded size of the base type, in bytes.
        element_size: The encoded size of each vector or array element.

    Example:
        Describe a vector of 32-bit integers:

        >>> type_ref = TypeReference(
        ...     base_type=BaseType.VECTOR,
        ...     element=BaseType.INT,
        ...     element_size=4,
        ... )
        >>> type_ref.element
        <BaseType.INT: 7>
    """

    base_type: BaseType
    element: BaseType = BaseType.NONE
    index: int = -1
    fixed_length: int = 0
    base_size: int = 4
    element_size: int = 0


class SchemaAttribute(msgspec.Struct, frozen=True, kw_only=True):
    """Store one FlatBuffers schema attribute.

    Attributes:
        key: The attribute name.
        value: The optional attribute value.

    Example:
        Create a marker attribute:

        >>> attribute = SchemaAttribute(key="dynamic_extension")
        >>> attribute.value is None
        True
    """

    key: str
    value: str | None = None


class EnumValue(msgspec.Struct, frozen=True, kw_only=True):
    """Describe one enum value or union discriminator.

    Attributes:
        name: The value name from the IDL.
        value: The encoded integer value.
        union_type: The union arm type, when this value belongs to a union.
        documentation: Documentation lines from the IDL.
        attributes: Attributes attached to the value.

    Example:
        Create a regular enum value:

        >>> value = EnumValue(name="RED", value=1)
        >>> value.value
        1
    """

    name: str
    value: int
    union_type: TypeReference | None = None
    documentation: tuple[str, ...] = ()
    attributes: tuple[SchemaAttribute, ...] = ()


class EnumDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Describe an enum or union declared in FlatBuffers IDL.

    Attributes:
        name: The fully qualified FlatBuffers name.
        values: Values in the definition.
        underlying_type: The integer storage type.
        is_union: Whether this definition describes a union.
        attributes: Attributes attached to the definition.
        documentation: Documentation lines from the IDL.
        declaration_file: The source file recorded by ``flatc``.

    Example:
        Describe a byte-backed enum:

        >>> definition = EnumDefinition(
        ...     name="example.Color",
        ...     values=(EnumValue(name="RED", value=1),),
        ...     underlying_type=TypeReference(base_type=BaseType.UBYTE),
        ... )
        >>> definition.name
        'example.Color'
    """

    name: str
    values: tuple[EnumValue, ...]
    underlying_type: TypeReference
    is_union: bool = False
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class FieldDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Describe one field in a FlatBuffers table or struct.

    Attributes:
        name: The field name from the IDL.
        type: The field's reflected type.
        id: The numeric field identifier.
        offset: The table vtable offset or inline struct offset.
        default_integer: The default for integral and Boolean fields.
        default_real: The default for floating-point fields.
        deprecated: Whether the field is deprecated.
        required: Whether the field must be present.
        key: Whether the field is a table key.
        optional: Whether a scalar field is optional.
        padding: The number of padding bytes after an inline struct field.
        offset64: Whether the field uses 64-bit offsets.
        nested_flatbuffer: The root type of a nested FlatBuffer byte vector.
        dynamic_flatbuffer: The discriminator field for a dynamic payload.
        dynamic_allow: The allowed dynamic type namespace pattern.
        attributes: Attributes attached to the field.
        documentation: Documentation lines from the IDL.

    Example:
        Describe the first 32-bit integer field in a table:

        >>> field = FieldDefinition(
        ...     name="count",
        ...     type=TypeReference(base_type=BaseType.INT),
        ...     id=0,
        ...     offset=4,
        ... )
        >>> field.default_integer
        0
    """

    name: str
    type: TypeReference
    id: int
    offset: int
    default_integer: int = 0
    default_real: float = 0.0
    deprecated: bool = False
    required: bool = False
    key: bool = False
    optional: bool = False
    padding: int = 0
    offset64: bool = False
    nested_flatbuffer: str | None = None
    dynamic_flatbuffer: str | None = None
    dynamic_allow: str | None = None
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()


class ObjectDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Describe a FlatBuffers table or struct.

    Attributes:
        name: The fully qualified FlatBuffers name.
        fields: Fields declared by the object.
        is_struct: Whether the object is an inline struct rather than a table.
        min_alignment: The object's minimum alignment in bytes.
        byte_size: The encoded size of an inline struct.
        attributes: Attributes attached to the object.
        documentation: Documentation lines from the IDL.
        declaration_file: The source file recorded by ``flatc``.

    Example:
        Describe an empty table:

        >>> definition = ObjectDefinition(name="example.Empty", fields=())
        >>> definition.is_struct
        False
    """

    name: str
    fields: tuple[FieldDefinition, ...]
    is_struct: bool = False
    min_alignment: int = 0
    byte_size: int = 0
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class RpcCallDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Describe one RPC method from a FlatBuffers service.

    Attributes:
        name: The RPC method name.
        request: The fully qualified request table name.
        response: The fully qualified response table name.
        attributes: Attributes attached to the method.
        documentation: Documentation lines from the IDL.

    Example:
        Describe an RPC method:

        >>> call = RpcCallDefinition(
        ...     name="GetMonster",
        ...     request="example.GetMonsterRequest",
        ...     response="example.Monster",
        ... )
        >>> call.response
        'example.Monster'
    """

    name: str
    request: str
    response: str
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()


class ServiceDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Describe a FlatBuffers RPC service.

    Attributes:
        name: The fully qualified service name.
        calls: RPC methods declared by the service.
        attributes: Attributes attached to the service.
        documentation: Documentation lines from the IDL.
        declaration_file: The source file recorded by ``flatc``.

    Example:
        Describe a service with no methods:

        >>> service = ServiceDefinition(name="example.Monsters", calls=())
        >>> service.calls
        ()
    """

    name: str
    calls: tuple[RpcCallDefinition, ...]
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class SchemaFile(msgspec.Struct, frozen=True, kw_only=True):
    """Record one source file included in a schema compilation.

    Attributes:
        filename: The filename relative to the project root.
        included_filenames: Files included by this source file.

    Example:
        Record a schema include:

        >>> source = SchemaFile(
        ...     filename="monster.fbs",
        ...     included_filenames=("weapon.fbs",),
        ... )
        >>> source.included_filenames
        ('weapon.fbs',)
    """

    filename: str
    included_filenames: tuple[str, ...] = ()


class Schema(msgspec.Struct, frozen=True, kw_only=True):
    """Store a normalized FlatBuffers schema for code generation.

    Attributes:
        objects: Table and struct definitions.
        enums: Enum and union definitions.
        root_table: The fully qualified root table name, if declared.
        file_identifier: The four-character file identifier, if declared.
        file_extension: The recommended file extension, if declared.
        services: RPC service definitions.
        advanced_features: Advanced features used by the schema.
        files: Source files included in the compilation.

    Example:
        Create a schema with one root table:

        >>> table = ObjectDefinition(name="example.Monster", fields=())
        >>> schema = Schema(objects=(table,), enums=(), root_table=table.name)
        >>> schema.root is table
        True
    """

    objects: tuple[ObjectDefinition, ...]
    enums: tuple[EnumDefinition, ...]
    root_table: str | None = None
    file_identifier: str | None = None
    file_extension: str | None = None
    services: tuple[ServiceDefinition, ...] = ()
    advanced_features: AdvancedFeature = AdvancedFeature.NONE
    files: tuple[SchemaFile, ...] = ()

    def object(self, name: str) -> ObjectDefinition:
        """Return an object by its fully qualified FlatBuffers name.

        Args:
            name: The table or struct name to find.

        Returns:
            The matching object definition.

        Raises:
            KeyError: No object has the requested name.

        Example:
            Look up a table definition:

            >>> table = ObjectDefinition(name="example.Monster", fields=())
            >>> Schema(objects=(table,), enums=()).object(table.name) is table
            True
        """

        for item in self.objects:
            if item.name == name:
                return item
        raise KeyError(name)

    def enum(self, name: str) -> EnumDefinition:
        """Return an enum by its fully qualified FlatBuffers name.

        Args:
            name: The enum or union name to find.

        Returns:
            The matching enum definition.

        Raises:
            KeyError: No enum has the requested name.

        Example:
            Look up an enum definition:

            >>> color = EnumDefinition(
            ...     name="example.Color",
            ...     values=(),
            ...     underlying_type=TypeReference(base_type=BaseType.UBYTE),
            ... )
            >>> Schema(objects=(), enums=(color,)).enum(color.name) is color
            True
        """

        for item in self.enums:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def root(self) -> ObjectDefinition | None:
        """The root table definition, if the schema declares one.

        Example:
            Read the optional root definition:

            >>> Schema(objects=(), enums=()).root is None
            True
        """

        if self.root_table is None:
            return None
        return self.object(self.root_table)
