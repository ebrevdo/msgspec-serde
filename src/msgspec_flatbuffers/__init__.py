"""Lazy FlatBuffers views and msgspec model generation."""

from importlib.metadata import PackageNotFoundError, version

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
from .generator import GenerationError, generate, render_module
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

try:
    __version__ = version("msgspec-flatbuffers")
except PackageNotFoundError:  # pragma: no cover - direct source-tree import
    __version__ = "0.0.0"

__all__ = [
    "AdvancedFeature",
    "BaseType",
    "BufferBoundsError",
    "CachedVector",
    "DynamicType",
    "DynamicValue",
    "DynamicView",
    "EnumDefinition",
    "EnumValue",
    "FieldDefinition",
    "FlatcError",
    "FlatcNotFoundError",
    "GenerationError",
    "InvalidBufferError",
    "InvalidSchemaError",
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
    "build_byte_vector",
    "build_offset_vector",
    "build_scalar_vector",
    "build_string_vector",
    "compile_schema",
    "compile_schema_to_bfbs",
    "dec_hook",
    "dynamic_types",
    "enc_hook",
    "encode_dynamic",
    "generate",
    "load_bfbs",
    "parse_bfbs",
    "register_dynamic_module",
    "register_dynamic_type",
    "render_module",
]
