"""Lazy FlatBuffers views and msgspec model generation."""

import warnings

from . import flatbuffer, json, msgpack
from ._dynamic import (
    DynamicType,
    DynamicValue,
    DynamicView,
    dynamic_types,
    register_dynamic_module,
    register_dynamic_type,
)
from ._overrides import DynamicModelOverrides
from ._runtime import (
    BufferBoundsError,
    CachedVector,
    InvalidBufferError,
    OpenIntEnum,
    StringVector,
    StructVector,
    StructView,
    TableMap,
    TableVector,
    TableView,
    UnionDispatch,
    UnionVector,
)
from ._version import __version__, _major_minor
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


class GeneratedCodeVersionError(ImportError):
    """Raised when generated code and the runtime use different major versions."""


class GeneratedCodeVersionWarning(UserWarning):
    """Issued when generated code is newer than its same-major runtime."""


warn_on_older_runtime = True
_warned_version_pairs: set[tuple[str, str]] = set()


def _check_generated_code_version(generated_version: str) -> None:
    """Validate a generated module against the installed runtime version."""

    if __version__ == "0.0.0":
        return
    generated_major, generated_minor = _major_minor(generated_version)
    runtime_major, runtime_minor = _major_minor(__version__)
    if generated_major != runtime_major:
        raise GeneratedCodeVersionError(
            "generated code from msgspec-flatbuffers "
            f"{generated_version} cannot run with msgspec-flatbuffers "
            f"{__version__}: major versions differ"
        )
    if runtime_minor >= generated_minor or not warn_on_older_runtime:
        return
    pair = (generated_version, __version__)
    if pair in _warned_version_pairs:
        return
    _warned_version_pairs.add(pair)
    warnings.warn(
        "generated code was produced by msgspec-flatbuffers "
        f"{generated_version}, but runtime {__version__} is older; "
        "forward compatibility is not guaranteed",
        GeneratedCodeVersionWarning,
        stacklevel=2,
    )


__all__ = [
    "AdvancedFeature",
    "BaseType",
    "BufferBoundsError",
    "CachedVector",
    "DynamicModelOverrides",
    "DynamicType",
    "DynamicValue",
    "DynamicView",
    "EnumDefinition",
    "EnumValue",
    "FieldDefinition",
    "FlatcError",
    "FlatcNotFoundError",
    "GeneratedCodeVersionError",
    "GeneratedCodeVersionWarning",
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
    "TableMap",
    "TableVector",
    "TableView",
    "TypeReference",
    "UnionDispatch",
    "UnionVector",
    "__version__",
    "compile_schema",
    "compile_schema_to_bfbs",
    "dynamic_types",
    "flatbuffer",
    "generate",
    "json",
    "load_bfbs",
    "msgpack",
    "parse_bfbs",
    "register_dynamic_module",
    "register_dynamic_type",
    "render_module",
    "warn_on_older_runtime",
]
