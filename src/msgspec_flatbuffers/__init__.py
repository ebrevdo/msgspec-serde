"""Lazy FlatBuffers views and msgspec model generation."""

from importlib.metadata import PackageNotFoundError, version

from ._runtime import (
    BufferBoundsError,
    CachedVector,
    InvalidBufferError,
    OpenIntEnum,
    StringVector,
    StructVector,
    StructView,
    TableVector,
    TableView,
    UnionDispatch,
    UnionVector,
    build_byte_vector,
    build_offset_vector,
    build_scalar_vector,
    build_string_vector,
)
from .compiler import (
    FlatcError,
    FlatcNotFoundError,
    InvalidSchemaError,
    compile_schema,
    compile_schema_to_bfbs,
    load_bfbs,
    parse_bfbs,
)
from ._conversion import dec_hook, enc_hook
from ._dynamic import (
    DynamicType,
    DynamicValue,
    DynamicView,
    dynamic_types,
    encode_dynamic,
    register_dynamic_module,
    register_dynamic_type,
)
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
from .generator import GenerationError, generate, render_module

try:
    __version__ = version("msgspec-flatbuffers")
except PackageNotFoundError:  # pragma: no cover - direct source-tree import
    __version__ = "0.0.0"

__all__ = [
    "AdvancedFeature",
    "BaseType",
    "BufferBoundsError",
    "CachedVector",
    "EnumDefinition",
    "EnumValue",
    "DynamicType",
    "DynamicValue",
    "DynamicView",
    "FieldDefinition",
    "FlatcError",
    "FlatcNotFoundError",
    "GenerationError",
    "InvalidSchemaError",
    "InvalidBufferError",
    "ObjectDefinition",
    "OpenIntEnum",
    "RpcCallDefinition",
    "Schema",
    "SchemaAttribute",
    "SchemaFile",
    "ServiceDefinition",
    "StringVector",
    "StructVector",
    "StructView",
    "TableVector",
    "TableView",
    "TypeReference",
    "UnionDispatch",
    "UnionVector",
    "__version__",
    "compile_schema",
    "compile_schema_to_bfbs",
    "dec_hook",
    "dynamic_types",
    "enc_hook",
    "encode_dynamic",
    "build_byte_vector",
    "build_offset_vector",
    "build_scalar_vector",
    "build_string_vector",
    "generate",
    "load_bfbs",
    "parse_bfbs",
    "render_module",
    "register_dynamic_module",
    "register_dynamic_type",
]
