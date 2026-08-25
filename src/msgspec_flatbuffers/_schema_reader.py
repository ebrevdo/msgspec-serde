"""Convert the generated ``reflection.fbs`` API into normalized models."""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import TypeVar

from ._reflection.Enum import Enum as ReflectedEnum
from ._reflection.EnumVal import EnumVal as ReflectedEnumValue
from ._reflection.Field import Field as ReflectedField
from ._reflection.KeyValue import KeyValue as ReflectedKeyValue
from ._reflection.Object import Object as ReflectedObject
from ._reflection.RPCCall import RPCCall as ReflectedRpcCall
from ._reflection.Schema import Schema as ReflectedSchema
from ._reflection.SchemaFile import SchemaFile as ReflectedSchemaFile
from ._reflection.Service import Service as ReflectedService
from ._reflection.Type import Type as ReflectedType
from .schema import (
    AdvancedFeature,
    BaseType,
    EnumDefinition,
    EnumValue,
    FieldDefinition,
    ObjectDefinition,
    RpcCallDefinition,
    Schema,
    SchemaAttribute,
    SchemaFile,
    ServiceDefinition,
    TypeReference,
)

_T = TypeVar("_T")
_KNOWN_ADVANCED_FEATURES = (
    AdvancedFeature.ARRAYS
    | AdvancedFeature.UNIONS
    | AdvancedFeature.OPTIONAL_SCALARS
    | AdvancedFeature.DEFAULT_VECTORS_AND_STRINGS
)


class InvalidSchemaError(ValueError):
    """Raised when a binary schema is malformed or unsupported."""


def _required(value: _T | None, description: str) -> _T:
    if value is None:
        raise InvalidSchemaError(f"binary schema is missing required {description}")
    return value


def _text(value: object | None, description: str) -> str:
    value = _required(value, description)
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    raise InvalidSchemaError(f"{description} is not text")


def _optional_text(value: object | None, description: str) -> str | None:
    if value is None:
        return None
    return _text(value, description)


def _documents(
    length: int,
    getter: Callable[[int], object],
    description: str,
) -> tuple[str, ...]:
    return tuple(
        _text(getter(index), description).strip() for index in range(length)
    )


def _attributes(
    length: int,
    getter: Callable[[int], ReflectedKeyValue | None],
) -> tuple[SchemaAttribute, ...]:
    result: list[SchemaAttribute] = []
    for index in range(length):
        item = _required(getter(index), "attribute")
        result.append(
            SchemaAttribute(
                key=_text(item.Key(), "attribute key"),
                value=_optional_text(item.Value(), "attribute value"),
            )
        )
    return tuple(result)


def _attribute_value(
    attributes: tuple[SchemaAttribute, ...],
    key: str,
) -> str | None:
    matches = [attribute for attribute in attributes if attribute.key == key]
    if len(matches) > 1:
        raise InvalidSchemaError(f"field has duplicate {key!r} attributes")
    if not matches:
        return None
    value = matches[0].value
    if value is None or not value.strip():
        raise InvalidSchemaError(f"field attribute {key!r} requires a value")
    return value


def _base_type(value: int) -> BaseType:
    try:
        return BaseType(value)
    except ValueError as error:
        raise InvalidSchemaError(f"unknown FlatBuffers base type {value}") from error


def _type(item: ReflectedType) -> TypeReference:
    return TypeReference(
        base_type=_base_type(item.BaseType()),
        element=_base_type(item.Element()),
        index=item.Index(),
        fixed_length=item.FixedLength(),
        base_size=item.BaseSize(),
        element_size=item.ElementSize(),
    )


def _field(item: ReflectedField) -> FieldDefinition:
    field_type = _required(item.Type(), "field type")
    attributes = _attributes(item.AttributesLength(), item.Attributes)
    return FieldDefinition(
        name=_text(item.Name(), "field name"),
        type=_type(field_type),
        id=item.Id(),
        offset=item.Offset(),
        default_integer=item.DefaultInteger(),
        default_real=item.DefaultReal(),
        deprecated=item.Deprecated(),
        required=item.Required(),
        key=item.Key(),
        optional=item.Optional(),
        padding=item.Padding(),
        offset64=item.Offset64(),
        nested_flatbuffer=_attribute_value(attributes, "nested_flatbuffer"),
        dynamic_flatbuffer=_attribute_value(attributes, "dynamic_flatbuffer"),
        dynamic_allow=_attribute_value(attributes, "dynamic_allow"),
        attributes=attributes,
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "field documentation",
        ),
    )


def _object(item: ReflectedObject) -> ObjectDefinition:
    fields = tuple(
        _field(_required(item.Fields(index), "object field"))
        for index in range(item.FieldsLength())
    )
    return ObjectDefinition(
        name=_text(item.Name(), "object name"),
        fields=fields,
        is_struct=item.IsStruct(),
        min_alignment=item.Minalign(),
        byte_size=item.Bytesize(),
        attributes=_attributes(item.AttributesLength(), item.Attributes),
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "object documentation",
        ),
        declaration_file=_optional_text(item.DeclarationFile(), "declaration file"),
    )


def _enum_value(item: ReflectedEnumValue) -> EnumValue:
    union_type = item.UnionType()
    return EnumValue(
        name=_text(item.Name(), "enum value name"),
        value=item.Value(),
        union_type=None if union_type is None else _type(union_type),
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "enum value documentation",
        ),
        attributes=_attributes(item.AttributesLength(), item.Attributes),
    )


def _enum(item: ReflectedEnum) -> EnumDefinition:
    underlying_type = _required(item.UnderlyingType(), "enum underlying type")
    values = tuple(
        _enum_value(_required(item.Values(index), "enum value"))
        for index in range(item.ValuesLength())
    )
    return EnumDefinition(
        name=_text(item.Name(), "enum name"),
        values=values,
        underlying_type=_type(underlying_type),
        is_union=item.IsUnion(),
        attributes=_attributes(item.AttributesLength(), item.Attributes),
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "enum documentation",
        ),
        declaration_file=_optional_text(item.DeclarationFile(), "declaration file"),
    )


def _rpc_call(item: ReflectedRpcCall) -> RpcCallDefinition:
    request = _required(item.Request(), "RPC request type")
    response = _required(item.Response(), "RPC response type")
    return RpcCallDefinition(
        name=_text(item.Name(), "RPC call name"),
        request=_text(request.Name(), "RPC request name"),
        response=_text(response.Name(), "RPC response name"),
        attributes=_attributes(item.AttributesLength(), item.Attributes),
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "RPC documentation",
        ),
    )


def _service(item: ReflectedService) -> ServiceDefinition:
    calls = tuple(
        _rpc_call(_required(item.Calls(index), "RPC call"))
        for index in range(item.CallsLength())
    )
    return ServiceDefinition(
        name=_text(item.Name(), "service name"),
        calls=calls,
        attributes=_attributes(item.AttributesLength(), item.Attributes),
        documentation=_documents(
            item.DocumentationLength(),
            item.Documentation,
            "service documentation",
        ),
        declaration_file=_optional_text(item.DeclarationFile(), "declaration file"),
    )


def _schema_file(item: ReflectedSchemaFile) -> SchemaFile:
    included_filenames = tuple(
        _text(item.IncludedFilenames(index), "included filename")
        for index in range(item.IncludedFilenamesLength())
    )
    return SchemaFile(
        filename=_text(item.Filename(), "schema filename"),
        included_filenames=included_filenames,
    )


def _parse_schema(root: ReflectedSchema) -> Schema:
    raw_features = root.AdvancedFeatures()
    unknown_features = raw_features & ~int(_KNOWN_ADVANCED_FEATURES)
    if unknown_features:
        raise InvalidSchemaError(
            f"binary schema uses unknown advanced feature bits 0x{unknown_features:x}"
        )

    objects = tuple(
        _object(_required(root.Objects(index), "object"))
        for index in range(root.ObjectsLength())
    )
    enums = tuple(
        _enum(_required(root.Enums(index), "enum"))
        for index in range(root.EnumsLength())
    )
    services = tuple(
        _service(_required(root.Services(index), "service"))
        for index in range(root.ServicesLength())
    )
    files = tuple(
        _schema_file(_required(root.FbsFiles(index), "schema file"))
        for index in range(root.FbsFilesLength())
    )
    root_table = root.RootTable()
    file_identifier = _optional_text(root.FileIdent(), "file identifier") or None
    file_extension = _optional_text(root.FileExt(), "file extension") or None

    return Schema(
        objects=objects,
        enums=enums,
        root_table=(
            None
            if root_table is None
            else _text(root_table.Name(), "root table name")
        ),
        file_identifier=file_identifier,
        file_extension=file_extension,
        services=services,
        advanced_features=AdvancedFeature(raw_features),
        files=files,
    )


def parse_bfbs(buffer: bytes | bytearray | memoryview) -> Schema:
    """Parse a binary FlatBuffers schema into normalized msgspec models."""

    data = bytes(buffer)
    if not ReflectedSchema.SchemaBufferHasIdentifier(data, 0):
        raise InvalidSchemaError("buffer does not have the BFBS file identifier")
    try:
        return _parse_schema(ReflectedSchema.GetRootAs(data))
    except InvalidSchemaError:
        raise
    except (IndexError, OverflowError, TypeError, ValueError, struct.error) as error:
        raise InvalidSchemaError("malformed binary FlatBuffers schema") from error
