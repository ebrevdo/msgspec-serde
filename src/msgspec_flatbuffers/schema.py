"""Normalized msgspec models for a reflected FlatBuffers schema."""

from __future__ import annotations

from enum import IntEnum, IntFlag

import msgspec


class BaseType(IntEnum):
    """Scalar and aggregate types defined by ``reflection.fbs``."""

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
    """Feature flags recorded in a binary FlatBuffers schema."""

    NONE = 0
    ARRAYS = 1
    UNIONS = 2
    OPTIONAL_SCALARS = 4
    DEFAULT_VECTORS_AND_STRINGS = 8


class TypeReference(msgspec.Struct, frozen=True, kw_only=True):
    """A field type, including references into schema object or enum lists."""

    base_type: BaseType
    element: BaseType = BaseType.NONE
    index: int = -1
    fixed_length: int = 0
    base_size: int = 4
    element_size: int = 0


class SchemaAttribute(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    value: str | None = None


class EnumValue(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    value: int
    union_type: TypeReference | None = None
    documentation: tuple[str, ...] = ()
    attributes: tuple[SchemaAttribute, ...] = ()


class EnumDefinition(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    values: tuple[EnumValue, ...]
    underlying_type: TypeReference
    is_union: bool = False
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class FieldDefinition(msgspec.Struct, frozen=True, kw_only=True):
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
    name: str
    fields: tuple[FieldDefinition, ...]
    is_struct: bool = False
    min_alignment: int = 0
    byte_size: int = 0
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class RpcCallDefinition(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    request: str
    response: str
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()


class ServiceDefinition(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    calls: tuple[RpcCallDefinition, ...]
    attributes: tuple[SchemaAttribute, ...] = ()
    documentation: tuple[str, ...] = ()
    declaration_file: str | None = None


class SchemaFile(msgspec.Struct, frozen=True, kw_only=True):
    filename: str
    included_filenames: tuple[str, ...] = ()


class Schema(msgspec.Struct, frozen=True, kw_only=True):
    """A normalized FlatBuffers schema suitable for code generation."""

    objects: tuple[ObjectDefinition, ...]
    enums: tuple[EnumDefinition, ...]
    root_table: str | None = None
    file_identifier: str | None = None
    file_extension: str | None = None
    services: tuple[ServiceDefinition, ...] = ()
    advanced_features: AdvancedFeature = AdvancedFeature.NONE
    files: tuple[SchemaFile, ...] = ()

    def object(self, name: str) -> ObjectDefinition:
        """Return an object by its fully-qualified FlatBuffers name."""

        for item in self.objects:
            if item.name == name:
                return item
        raise KeyError(name)

    def enum(self, name: str) -> EnumDefinition:
        """Return an enum by its fully-qualified FlatBuffers name."""

        for item in self.enums:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def root(self) -> ObjectDefinition | None:
        """The root table definition, if the schema declares one."""

        if self.root_table is None:
            return None
        return self.object(self.root_table)
