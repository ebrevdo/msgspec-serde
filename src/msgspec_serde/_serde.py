"""Native plan compilation shared by the JSON and MessagePack codecs."""

from __future__ import annotations

import decimal
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from types import UnionType
from typing import Any, TypeGuard, Union, cast, get_args, get_origin

import msgspec
import numpy as np

from ._conversion import _ndarray_dtype, decode_fallback, encode_fallback
from ._native import NativePlan

_SUPPORTED_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    }
)

_DTYPE_NAMES = {np.dtype(name): name for name in _SUPPORTED_DTYPES}
_JSON_DEPTH_ERROR = "Encountered nesting of JSON maps and arrays"


_NativeBinding = tuple[NativePlan, str]

_NATIVE_SERDE_BINDINGS: dict[type[msgspec.Struct], _NativeBinding | None] = {}
_NATIVE_SERDE_DECODERS: dict[type[msgspec.Struct], _NativeBinding | None] = {}
_AUTO_SERDE_BINDINGS: dict[
    type[msgspec.Struct],
    _CompiledSerdeBinding | str,
] = {}


class _UnsupportedSerdePlan(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _CompiledSerdeBinding:
    plan: NativePlan
    object_name: str
    fallback_fields: tuple[str, ...]
    fallback_annotations: tuple[tuple[str, Any], ...]


def _generated_model_binding(model_type: type[msgspec.Struct]) -> Any | None:
    from ._flatbuffer import _model_binding

    try:
        return _model_binding(model_type)
    except TypeError:
        return None


def _resolve_native_binding(
    model_type: type[msgspec.Struct],
    *,
    for_decode: bool,
) -> _NativeBinding | None:
    generated = _generated_model_binding(model_type)
    if generated is None:
        compiled = _get_auto_serde_binding(model_type)
        if compiled is None:
            return None
        return compiled.plan, compiled.object_name
    if model_type is not generated.generated_type and (
        for_decode
        or not _generated_subclass_wire_matches(
            generated.generated_type,
            model_type,
        )
    ):
        return None
    return generated.plan, generated.object_name


def _generated_subclass_wire_matches(
    generated_type: type[msgspec.Struct],
    model_type: type[msgspec.Struct],
) -> bool:
    generated_config = generated_type.__struct_config__
    model_config = model_type.__struct_config__
    if any(
        getattr(generated_config, option) != getattr(model_config, option)
        for option in ("array_like", "omit_defaults", "tag", "tag_field")
    ):
        return False
    generated_fields = tuple(
        (field.name, field.encode_name)
        for field in msgspec.structs.fields(generated_type)
    )
    model_fields = tuple(
        (field.name, field.encode_name) for field in msgspec.structs.fields(model_type)
    )
    return model_fields == generated_fields


def _cached_native_binding(
    model_type: type[msgspec.Struct],
    *,
    for_decode: bool,
) -> _NativeBinding | None:
    cache = _NATIVE_SERDE_DECODERS if for_decode else _NATIVE_SERDE_BINDINGS
    try:
        return cache[model_type]
    except KeyError:
        binding = _resolve_native_binding(model_type, for_decode=for_decode)
        cache[model_type] = binding
        return binding


def encode_native_model(
    value: Any,
    *,
    is_json: bool,
    fallback_encoder: Any,
    order: str | None,
    decimal_format: str,
    uuid_format: str,
) -> bytes | None:
    """Encode one Struct through its cached native plan when supported."""

    if not isinstance(value, msgspec.Struct):
        return None
    model_type = type(value)
    binding = _cached_native_binding(model_type, for_decode=False)
    if binding is None:
        return None
    plan, object_name = binding
    try:
        return plan.encode_serde(
            object_name,
            value,
            is_json,
            fallback_encoder=fallback_encoder,
            order=order,
            decimal_format=decimal_format,
            uuid_format=uuid_format,
        )
    except NotImplementedError:
        _NATIVE_SERDE_BINDINGS[model_type] = None
        return None


def decode_native_model(
    buffer: Any,
    model_type: type[msgspec.Struct],
    *,
    is_json: bool,
    strict: bool,
    fallback_decoders: dict[str, Any],
) -> msgspec.Struct | None:
    """Decode one Struct through its cached native plan when supported."""

    binding = _cached_native_binding(model_type, for_decode=True)
    if binding is None:
        return None
    plan, object_name = binding
    try:
        return cast(
            msgspec.Struct,
            plan.decode_serde(
                object_name,
                buffer,
                is_json,
                strict=strict,
                fallback_decoders=fallback_decoders,
            ),
        )
    except NotImplementedError:
        _NATIVE_SERDE_DECODERS[model_type] = None
        return None
    except ValueError as error:
        if is_json and _JSON_DEPTH_ERROR in str(error):
            return None
        raise


def write_encoded(buffer: bytearray, encoded: bytes, offset: int) -> None:
    """Write one pre-encoded document with msgspec encode_into semantics."""

    if offset == -1:
        buffer.extend(encoded)
        return
    if offset < 0:
        raise ValueError("offset must be >= -1")
    if offset > len(buffer):
        buffer.extend(b"\x00" * (offset - len(buffer)))
    buffer[offset:] = encoded


def _serde_type_name(model_type: type[msgspec.Struct]) -> str:
    return f"{model_type.__module__}.{model_type.__qualname__}"


class _SerdePlanCompiler:
    __slots__ = (
        "fallback_annotations",
        "fallback_fields",
        "fallback_ids",
        "model_types",
        "objects",
    )

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.model_types: dict[str, type[msgspec.Struct]] = {}
        self.fallback_annotations: dict[str, Any] = {}
        self.fallback_fields: list[str] = []
        self.fallback_ids: dict[Any, str] = {}

    def compile(self, root_type: type[msgspec.Struct]) -> _CompiledSerdeBinding:
        if not self._compile_object(root_type):
            raise _UnsupportedSerdePlan(
                f"{root_type.__qualname__} uses a Struct-wide configuration "
                "that requires msgspec fallback"
            )
        plan = NativePlan(
            msgspec.msgpack.encode(
                {
                    "version": 2,
                    "objects": list(self.objects.values()),
                }
            )
        )
        plan.bind_types(self.model_types)
        return _CompiledSerdeBinding(
            plan,
            _serde_type_name(root_type),
            tuple(self.fallback_fields),
            tuple(self.fallback_annotations.items()),
        )

    def _compile_object(self, model_type: type[msgspec.Struct]) -> bool:
        name = _serde_type_name(model_type)
        if name in self.objects:
            return True
        config = model_type.__struct_config__
        if (
            config.array_like
            or config.forbid_unknown_fields
            or (config.tag is not None and not isinstance(config.tag, str))
        ):
            return False

        fields = msgspec.structs.fields(model_type)
        if any(
            field.default is msgspec.UNSET
            or _annotation_contains(field.type, msgspec.UnsetType)
            for field in fields
        ):
            return False
        if config.omit_defaults and any(
            not _supported_serde_default(field)
            for field in fields
            if field.default is not msgspec.NODEFAULT
            or field.default_factory is not msgspec.NODEFAULT
        ):
            return False

        self.objects[name] = {}
        self.model_types[name] = model_type
        native_fields: list[dict[str, Any]] = []
        serde_fields: list[dict[str, str]] = []
        for slot, field in enumerate(fields):
            serde_fields.append(
                {
                    "attr_name": field.name,
                    "encode_name": field.encode_name,
                }
            )
            required = (
                field.default is msgspec.NODEFAULT
                and field.default_factory is msgspec.NODEFAULT
            )
            omit_default = config.omit_defaults and not required
            common: dict[str, Any] = {
                "name": field.name,
                "slot": slot,
                "offset": 0,
                "default": field.default if omit_default else None,
                "optional": not required,
                "required": required,
                "serde_nullable": _is_nullable(field.type),
                "serde_omit_default": omit_default,
            }
            field_plan = self._field_plan(
                field.type,
                fallback_id=f"{name}:{field.name}",
            )
            common.update(field_plan)
            native_fields.append(common)

        tag = config.tag
        self.objects[name] = {
            "name": name,
            "is_struct": False,
            "byte_size": 0,
            "min_alignment": 1,
            "fields": native_fields,
            "serde_fields": serde_fields,
            "serde_tag_field": config.tag_field if tag is not None else None,
            "serde_tag": tag,
        }
        return True

    def _field_plan(self, annotation: Any, *, fallback_id: str) -> dict[str, Any]:
        dtype = _ndarray_dtype(annotation)
        if dtype is not None:
            dtype_name = _DTYPE_NAMES.get(dtype)
            if dtype_name is not None:
                return {"kind": "vector_scalar", "scalar": dtype_name}
            return self._fallback(annotation, fallback_id)

        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin in (Union, UnionType):
            return self._union_field_plan(annotation, arguments, fallback_id)

        if annotation is int:
            return {
                "kind": "scalar",
                "scalar": "int64",
                "serde_python_int": True,
            }

        scalar = _serde_scalar(annotation)
        if scalar is not None:
            return {"kind": "scalar", "scalar": scalar}
        if annotation is str:
            return {"kind": "string"}
        if annotation is uuid.UUID:
            return {"kind": "uuid"}
        if annotation is decimal.Decimal:
            return {"kind": "decimal"}
        if _is_struct_type(annotation):
            if self._compile_object(annotation):
                return {"kind": "table", "target": _serde_type_name(annotation)}
            return self._fallback(annotation, fallback_id)

        if origin is list and len(arguments) == 1:
            return self._list_field_plan(annotation, arguments[0], fallback_id)

        return self._fallback(annotation, fallback_id)

    def _union_field_plan(
        self,
        annotation: Any,
        arguments: tuple[Any, ...],
        fallback_id: str,
    ) -> dict[str, Any]:
        arms = [argument for argument in arguments if argument is not type(None)]
        if len(arms) == 1:
            return self._field_plan(arms[0], fallback_id=fallback_id)
        tagged_arms = self._tagged_struct_arms(arms)
        if tagged_arms is not None:
            return self._union_plan("union", tagged_arms)
        return self._fallback(annotation, fallback_id)

    def _list_field_plan(
        self,
        annotation: Any,
        item: Any,
        fallback_id: str,
    ) -> dict[str, Any]:
        if item is str:
            return {"kind": "vector_string"}
        if _is_struct_type(item):
            if self._compile_object(item):
                return {
                    "kind": "vector_table",
                    "target": _serde_type_name(item),
                }
            return self._fallback(annotation, fallback_id)
        if get_origin(item) in (Union, UnionType):
            item_nullable = _is_nullable(item)
            arms = self._tagged_struct_arms(
                argument for argument in get_args(item) if argument is not type(None)
            )
            if arms is not None:
                plan = self._union_plan("union_vector", arms)
                plan["serde_element_nullable"] = item_nullable
                return plan
        return self._fallback(annotation, fallback_id)

    def _tagged_struct_arms(
        self,
        annotations: Iterable[Any],
    ) -> list[type[msgspec.Struct]] | None:
        arms = list(annotations)
        if not arms:
            return None
        for annotation in arms:
            if (
                not _is_struct_type(annotation)
                or not self._compile_object(annotation)
                or not isinstance(annotation.__struct_config__.tag, str)
            ):
                return None
        return arms

    @staticmethod
    def _union_plan(
        kind: str,
        arms: list[type[msgspec.Struct]],
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "arms": [
                {"tag": index + 1, "target": _serde_type_name(arm)}
                for index, arm in enumerate(arms)
            ],
        }

    def _fallback(self, annotation: Any, fallback_id: str) -> dict[str, Any]:
        self.fallback_fields.append(fallback_id)
        callback_id = self.fallback_ids.get(annotation)
        if callback_id is None:
            callback_id = f"fallback:{len(self.fallback_ids)}"
            self.fallback_ids[annotation] = callback_id
            self.fallback_annotations[callback_id] = annotation
        return {"kind": "fallback", "fallback_id": callback_id}


def _supported_serde_default(field: Any) -> bool:
    if field.default_factory is not msgspec.NODEFAULT:
        return False
    default = field.default
    if default is msgspec.NODEFAULT:
        return True
    if type(default) in (type(None), bool, float, str, bytes):
        return True
    return type(default) is int and -(2**63) <= default < 2**64


def _serde_scalar(annotation: Any) -> str | None:
    if annotation is bool:
        return "bool"
    if annotation is float:
        return "float64"
    return None


def _is_struct_type(annotation: Any) -> TypeGuard[type[msgspec.Struct]]:
    return isinstance(annotation, type) and issubclass(annotation, msgspec.Struct)


def _is_nullable(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, UnionType) and type(None) in get_args(
        annotation
    )


def _annotation_contains(annotation: Any, target: Any) -> bool:
    return annotation is target or any(
        _annotation_contains(argument, target) for argument in get_args(annotation)
    )


def _compile_serde_binding(
    model_type: type[msgspec.Struct],
) -> _CompiledSerdeBinding | str:
    try:
        return _SerdePlanCompiler().compile(model_type)
    except _UnsupportedSerdePlan as error:
        return str(error)


def _cached_auto_serde_binding(
    model_type: type[msgspec.Struct],
) -> _CompiledSerdeBinding | str:
    binding = _AUTO_SERDE_BINDINGS.get(model_type)
    if binding is None:
        binding = _compile_serde_binding(model_type)
        _AUTO_SERDE_BINDINGS[model_type] = binding
    return binding


def _get_auto_serde_binding(
    model_type: type[msgspec.Struct],
) -> _CompiledSerdeBinding | None:
    binding = _cached_auto_serde_binding(model_type)
    return None if isinstance(binding, str) else binding


def serde_fallback_annotations(
    model_type: type[msgspec.Struct],
) -> tuple[tuple[str, Any], ...]:
    """Return callback ids and annotations for native fallback fields."""

    if _generated_model_binding(model_type) is None:
        binding = _get_auto_serde_binding(model_type)
        return () if binding is None else binding.fallback_annotations
    return ()


def register_serde_type(model_type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Compile and cache a native serde plan for one Struct graph."""

    if not isinstance(model_type, type) or not issubclass(model_type, msgspec.Struct):
        raise TypeError("serde registration requires a msgspec.Struct type")
    generated = _generated_model_binding(model_type)
    if generated is not None:
        exact_generated_type = model_type is generated.generated_type
        native_encode = exact_generated_type or _generated_subclass_wire_matches(
            generated.generated_type,
            model_type,
        )
        cached = (generated.plan, generated.object_name)
        _NATIVE_SERDE_BINDINGS[model_type] = cached if native_encode else None
        _NATIVE_SERDE_DECODERS[model_type] = cached if exact_generated_type else None
        return ()

    binding = _cached_auto_serde_binding(model_type)
    if isinstance(binding, str):
        raise TypeError(binding)
    cached = (binding.plan, binding.object_name)
    _NATIVE_SERDE_BINDINGS[model_type] = cached
    _NATIVE_SERDE_DECODERS[model_type] = cached
    return binding.fallback_fields


def make_enc_hook(
    *,
    user_hook: Any = None,
) -> Any:
    """Build an encoder hook with package types ahead of a user hook."""

    if user_hook is None:
        return encode_fallback

    def enc_hook(value: Any) -> Any:
        try:
            return encode_fallback(value)
        except NotImplementedError:
            return user_hook(value)

    return enc_hook


def make_dec_hook(user_hook: Any = None) -> Any:
    """Build a decoder hook with a user hook ahead of package types."""

    if user_hook is None:
        return decode_fallback

    def dec_hook(annotation: Any, value: Any) -> Any:
        try:
            return user_hook(annotation, value)
        except NotImplementedError:
            return decode_fallback(annotation, value)

    return dec_hook
