"""Type registry and implementation for the public FlatBuffer codecs."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import msgspec

from ._models import resolve_model_types, validate_model_subclass
from ._native import NativeModelTypes, NativePlan
from ._runtime import TableView

if TYPE_CHECKING:
    from ._overrides import DynamicModelOverrides

_DecodedT = TypeVar("_DecodedT")


@dataclass(frozen=True, slots=True)
class _ModelBinding:
    plan: NativePlan
    object_name: str
    generated_type: type[msgspec.Struct]
    identifier: str | None


_MODEL_BINDINGS: dict[type[msgspec.Struct], _ModelBinding] = {}
_RESOLVED_MODEL_BINDINGS: dict[type[msgspec.Struct], _ModelBinding] = {}


def register_type(
    model_type: type[msgspec.Struct],
    plan: NativePlan,
    object_name: str,
    identifier: str | None,
) -> None:
    """Register one generated table model with its native schema plan."""

    if not isinstance(model_type, type) or not issubclass(model_type, msgspec.Struct):
        raise TypeError("FlatBuffer model types must extend msgspec.Struct")
    if not isinstance(plan, NativePlan):
        raise TypeError("FlatBuffer model plans must be NativePlan instances")
    if not isinstance(object_name, str) or not object_name:
        raise TypeError("FlatBuffer object names must be nonempty strings")
    if identifier is not None and (
        not isinstance(identifier, str) or len(identifier.encode()) != 4
    ):
        raise TypeError("FlatBuffer file identifiers must contain four bytes")

    binding = _ModelBinding(plan, object_name, model_type, identifier)
    existing = _MODEL_BINDINGS.get(model_type)
    if existing is not None and existing != binding:
        raise ValueError(
            f"FlatBuffer model {model_type.__qualname__} is already registered"
        )
    _MODEL_BINDINGS[model_type] = binding
    _RESOLVED_MODEL_BINDINGS.clear()


def _model_binding(model_type: type[msgspec.Struct]) -> _ModelBinding:
    binding = _MODEL_BINDINGS.get(model_type)
    if binding is not None:
        return binding
    binding = _RESOLVED_MODEL_BINDINGS.get(model_type)
    if binding is not None:
        return binding

    matches: list[_ModelBinding] = []
    for base in model_type.__mro__[1:]:
        candidate = _MODEL_BINDINGS.get(base)
        if candidate is not None and candidate not in matches:
            matches.append(candidate)
    if not matches:
        raise TypeError(
            f"{model_type.__qualname__} is not a registered FlatBuffer model"
        )
    if len(matches) != 1:
        raise TypeError(
            f"{model_type.__qualname__} has multiple registered FlatBuffer bases"
        )
    binding = matches[0]
    validate_model_subclass(binding.generated_type, model_type)
    _RESOLVED_MODEL_BINDINGS[model_type] = binding
    return binding


class Encoder:
    """Encode registered generated models as FlatBuffers."""

    __slots__ = ("_initial_size", "_size_prefixed")

    def __init__(
        self,
        *,
        size_prefixed: bool = False,
        initial_size: int = 0,
    ) -> None:
        self._size_prefixed = size_prefixed
        self._initial_size = initial_size

    def encode(self, value: msgspec.Struct) -> memoryview:
        """Encode one registered model."""

        if not isinstance(value, msgspec.Struct):
            raise TypeError("FlatBuffer encoders require a msgspec.Struct model")
        binding = _model_binding(type(value))
        return binding.plan.pack(
            binding.object_name,
            value,
            identifier=binding.identifier,
            size_prefixed=self._size_prefixed,
            initial_size=self._initial_size,
        )


class Decoder:
    """Decode FlatBuffers as a materialized model or lazy table view."""

    __slots__ = (
        "_binding",
        "_check_identifier",
        "_dynamic_overrides",
        "_model_types",
        "_offset",
        "_size_prefixed",
        "_target_type",
    )

    def __init__(
        self,
        type: type[_DecodedT],
        *,
        offset: int = 0,
        size_prefixed: bool = False,
        check_identifier: bool = True,
        dynamic_overrides: DynamicModelOverrides | None = None,
    ) -> None:
        if not isinstance(type, builtins.type):
            raise TypeError("FlatBuffer decoder targets must be types")
        self._target_type = type
        self._offset = offset
        self._size_prefixed = size_prefixed
        self._check_identifier = check_identifier
        self._dynamic_overrides = dynamic_overrides
        if issubclass(type, msgspec.Struct):
            binding = _model_binding(type)
            self._binding: _ModelBinding | None = binding
            self._model_types: NativeModelTypes | None = resolve_model_types(
                binding.plan,
                binding.generated_type,
                type,
            )
        elif issubclass(type, TableView):
            self._binding = None
            self._model_types = None
            if dynamic_overrides is not None:
                raise TypeError(
                    "dynamic_overrides applies when decoding materialized models; "
                    "pass it to view.to_model() instead"
                )
        else:
            raise TypeError(
                "FlatBuffer decoder targets must be generated models or table views"
            )

    def decode(self, buffer: bytes | bytearray | memoryview) -> _DecodedT:
        """Decode one FlatBuffer."""

        binding = self._binding
        if binding is None:
            target = cast(type[TableView], self._target_type)
            return cast(
                _DecodedT,
                target._from_buffer(
                    buffer,
                    offset=self._offset,
                    size_prefixed=self._size_prefixed,
                    check_identifier=self._check_identifier,
                ),
            )
        return cast(
            _DecodedT,
            binding.plan.unpack(
                binding.object_name,
                buffer,
                identifier=binding.identifier,
                offset=self._offset,
                size_prefixed=self._size_prefixed,
                check_identifier=self._check_identifier,
                model_types=self._model_types,
                dynamic_overrides=self._dynamic_overrides,
            ),
        )


def encode(
    value: msgspec.Struct,
    *,
    size_prefixed: bool = False,
    initial_size: int = 0,
) -> memoryview:
    """Encode one registered model as a FlatBuffer."""

    return Encoder(
        size_prefixed=size_prefixed,
        initial_size=initial_size,
    ).encode(value)


def decode(
    buffer: bytes | bytearray | memoryview,
    *,
    type: type[_DecodedT],
    offset: int = 0,
    size_prefixed: bool = False,
    check_identifier: bool = True,
    dynamic_overrides: DynamicModelOverrides | None = None,
) -> _DecodedT:
    """Decode one FlatBuffer as a materialized model or lazy view."""

    return Decoder(
        type,
        offset=offset,
        size_prefixed=size_prefixed,
        check_identifier=check_identifier,
        dynamic_overrides=dynamic_overrides,
    ).decode(buffer)
