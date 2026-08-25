"""Registry-backed open polymorphism for nested FlatBuffers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import Any, ClassVar, Self, cast

import msgspec

from ._runtime import TableView

_MSGSPEC_TAG_FIELD = "__msgspec_flatbuffers_type__"
_OPAQUE_DATA_FIELD = "__msgspec_flatbuffers_data__"
_DYNAMIC_VALUE_FIELD = "value"
_KNOWN_DYNAMIC_FIELDS = frozenset((_MSGSPEC_TAG_FIELD, _DYNAMIC_VALUE_FIELD))
_OPAQUE_DYNAMIC_FIELDS = frozenset((_MSGSPEC_TAG_FIELD, _OPAQUE_DATA_FIELD))
_ABSENT: Any = object()
_MAX_NEGATIVE_ENTRIES_PER_INDEX = 1024


@dataclass(frozen=True, slots=True)
class DynamicType:
    """One registered dynamic FlatBuffer root table."""

    tag: str
    model_type: type[msgspec.Struct]
    view_type: type[TableView]


class _DynamicTypeRegistry:
    """Thread-safe registry used by dynamic FlatBuffer fields."""

    __slots__ = (
        "_by_model",
        "_by_tag",
        "_lock",
        "_modules",
        "_negative_model_count",
        "_negative_tag_count",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._by_tag: dict[str, DynamicType | None] = {}
        self._by_model: dict[type[object], DynamicType | None] = {}
        self._modules: dict[str, str] = {}
        self._negative_tag_count = 0
        self._negative_model_count = 0

    def register(
        self,
        tag: str,
        model_type: type[msgspec.Struct],
        view_type: type[TableView],
    ) -> DynamicType:
        """Register one dynamic type name, materialized model, and view."""

        entry = _validated_entry(tag, model_type, view_type)
        with self._lock:
            existing = (
                self._by_tag.get(entry.tag),
                self._by_model.get(entry.model_type),
            )
            if any(item is not None and item != entry for item in existing):
                raise ValueError(
                    "dynamic FlatBuffer registration conflicts with an existing "
                    "tag or model"
                )
            for item in existing:
                if item is not None:
                    return item
            if self._by_tag.get(entry.tag, _ABSENT) is None:
                self._negative_tag_count -= 1
            if self._by_model.get(model_type, _ABSENT) is None:
                self._negative_model_count -= 1
            self._by_tag[entry.tag] = entry
            self._by_model[model_type] = entry
        return entry

    def register_module(self, tag: str, module: str) -> str:
        """Associate a trusted type tag with a lazily imported module."""

        _validate_tag(tag)
        if not isinstance(module, str) or not module:
            raise ValueError("dynamic FlatBuffer module names cannot be empty")
        with self._lock:
            existing = self._modules.get(tag)
            if existing is not None and existing != module:
                raise ValueError(
                    f"dynamic FlatBuffer tag {tag!r} already maps to {existing!r}"
                )
            self._modules[tag] = module
            if self._by_tag.get(tag, _ABSENT) is None:
                del self._by_tag[tag]
                self._negative_tag_count -= 1
        return module

    def lookup_tag(self, tag: str) -> DynamicType | None:
        """Return the entry for a tag, importing its trusted module if needed."""

        try:
            return self._by_tag[tag]
        except KeyError:
            return self._load_or_cache_tag(tag)

    def lookup_model(self, model_type: type[object]) -> DynamicType | None:
        """Return the entry for an exact msgspec model type, if registered."""

        try:
            return self._by_model[model_type]
        except KeyError:
            with self._lock:
                entry = self._by_model.get(model_type, _ABSENT)
                if entry is not _ABSENT:
                    return entry
                if self._negative_model_count < _MAX_NEGATIVE_ENTRIES_PER_INDEX:
                    self._by_model[model_type] = None
                    self._negative_model_count += 1
            return None

    def _load_or_cache_tag(self, tag: str) -> DynamicType | None:
        with self._lock:
            entry = self._by_tag.get(tag, _ABSENT)
            if entry is not _ABSENT:
                return entry
            module = self._modules.get(tag)
            if module is None:
                if self._negative_tag_count < _MAX_NEGATIVE_ENTRIES_PER_INDEX:
                    self._by_tag[tag] = None
                    self._negative_tag_count += 1
                return None

        import_module(module)

        with self._lock:
            entry = self._by_tag.get(tag, _ABSENT)
            if entry is _ABSENT or entry is None:
                raise RuntimeError(
                    f"dynamic FlatBuffer module {module!r} did not register {tag!r}"
                )
            return entry


class DynamicValue:
    """A materialized registered model or an opaque dynamic payload."""

    __slots__ = ("_data", "_entry", "_tag", "_value")
    _allowed_prefix: ClassVar[str | None] = None

    def __init__(self, value: msgspec.Struct) -> None:
        entry = dynamic_types.lookup_model(type(value))
        if entry is None:
            raise TypeError(
                f"unregistered dynamic FlatBuffer model {type(value).__qualname__}"
            )
        self._tag = entry.tag
        self._entry: DynamicType | None = entry
        self._value: msgspec.Struct | None = value
        self._data: bytes | None = None

    @classmethod
    def opaque(cls, tag: str, data: bytes | bytearray | memoryview) -> Self:
        """Create a losslessly forwardable value for an unknown type tag."""

        cls._require_allowed(tag)
        value = cls.__new__(cls)
        value._tag = tag
        value._entry = None
        value._value = None
        value._data = bytes(data)
        return value

    @classmethod
    def _known(cls, entry: DynamicType, value: msgspec.Struct) -> Self:
        cls._require_allowed(entry.tag)
        result = cls.__new__(cls)
        result._tag = entry.tag
        result._entry = entry
        result._value = value
        result._data = None
        return result

    @classmethod
    def _require_allowed(cls, tag: str) -> None:
        _require_allowed_prefix(tag, cls._allowed_prefix)

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def value(self) -> msgspec.Struct | None:
        return self._value

    @property
    def data(self) -> bytes | None:
        return self._data

    @property
    def is_known(self) -> bool:
        return self._value is not None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicValue)
            and self._tag == other._tag
            and self._value == other._value
            and self._data == other._data
        )

    def __repr__(self) -> str:
        if self._value is not None:
            return f"DynamicValue(tag={self._tag!r}, value={self._value!r})"
        return f"DynamicValue.opaque(tag={self._tag!r}, data={self._data!r})"


class DynamicView:
    """A zero-copy dynamic payload resolved through the global registry."""

    __slots__ = ("_data", "_entry", "_tag", "_value", "_value_type")

    def __init__(
        self,
        tag: str,
        data: memoryview,
        value_type: type[DynamicValue] = DynamicValue,
    ) -> None:
        if not isinstance(value_type, type) or not issubclass(
            value_type,
            DynamicValue,
        ):
            raise TypeError("dynamic FlatBuffer value types must extend DynamicValue")
        value_type._require_allowed(tag)
        if not isinstance(data, memoryview) or not data.readonly:
            raise TypeError(
                "dynamic FlatBuffer view data must be a read-only memoryview"
            )
        self._tag = tag
        self._data = data
        self._value_type = value_type
        self._entry: DynamicType | None = None
        self._value: object = _ABSENT

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def data(self) -> memoryview:
        return self._data

    @property
    def value(self) -> TableView | None:
        resolved = self._resolve_view()
        if resolved is None:
            return None
        return resolved[1]

    @property
    def is_known(self) -> bool:
        return self._resolve_entry() is not None

    def to_model(self) -> DynamicValue:
        resolved = self._resolve_view()
        if resolved is None:
            return self._value_type.opaque(self._tag, self._data)
        entry, view = resolved
        model = cast(Any, view).to_model()
        if type(model) is not entry.model_type:
            raise TypeError(
                f"{entry.view_type.__qualname__}.to_model() returned "
                f"{type(model).__qualname__}, expected {entry.model_type.__qualname__}"
            )
        return self._value_type._known(entry, model)

    def _resolve_view(self) -> tuple[DynamicType, TableView] | None:
        entry = self._resolve_entry()
        if entry is None:
            return None
        cached = self._value
        if cached is _ABSENT:
            cached = entry.view_type.from_buffer(self._data)
            self._value = cached
        return entry, cast(TableView, cached)

    def _resolve_entry(self) -> DynamicType | None:
        entry = self._entry
        if entry is not None:
            return entry
        entry = dynamic_types.lookup_tag(self._tag)
        if entry is not None:
            self._entry = entry
        return entry


def _validate_tag(tag: object) -> str:
    if not isinstance(tag, str):
        raise TypeError("dynamic FlatBuffer tags must be strings")
    if not tag:
        raise ValueError("dynamic FlatBuffer tags cannot be empty")
    return tag


def _require_allowed_prefix(tag: str, prefix: str | None) -> None:
    _validate_tag(tag)
    if prefix is None or (tag.startswith(prefix) and len(tag) > len(prefix)):
        return
    raise ValueError(
        f"dynamic FlatBuffer tag {tag!r} is outside {prefix + '*'!r}"
    )


def _validated_entry(
    tag: str,
    model_type: type[msgspec.Struct],
    view_type: type[TableView],
) -> DynamicType:
    if not isinstance(model_type, type) or not issubclass(
        model_type,
        msgspec.Struct,
    ):
        raise TypeError("dynamic FlatBuffer models must be msgspec.Struct types")
    if not isinstance(view_type, type) or not issubclass(view_type, TableView):
        raise TypeError("dynamic FlatBuffer views must be TableView types")
    if not callable(getattr(model_type, "to_flatbuffer", None)):
        raise TypeError("dynamic FlatBuffer models must define to_flatbuffer()")
    if not callable(getattr(view_type, "to_model", None)):
        raise TypeError("dynamic FlatBuffer views must define to_model()")

    return DynamicType(_validate_tag(tag), model_type, view_type)


def dynamic_allow_prefix(pattern: str) -> str:
    """Validate a terminal namespace wildcard and return its tag prefix."""

    if (
        not isinstance(pattern, str)
        or not pattern.endswith(".*")
        or pattern.count("*") != 1
        or len(pattern) <= 2
    ):
        raise ValueError(
            "dynamic_allow must be a nonempty namespace followed by '.*'"
        )
    return pattern[:-1]


def register_dynamic_type(
    tag: str,
    model_type: type[msgspec.Struct],
    view_type: type[TableView],
) -> DynamicType:
    """Register a dynamic type in the process-wide registry."""

    return dynamic_types.register(tag, model_type, view_type)


def register_dynamic_module(tag: str, module: str) -> str:
    """Register a trusted module for lazy loading of one type tag."""

    return dynamic_types.register_module(tag, module)


def encode_dynamic(
    value: DynamicValue,
    allowed_prefix: str,
) -> tuple[str, bytes | bytearray | memoryview]:
    """Return the physical type tag and nested FlatBuffer bytes."""

    if not isinstance(value, DynamicValue):
        raise TypeError("dynamic FlatBuffer fields require DynamicValue instances")
    _require_allowed_prefix(value._tag, allowed_prefix)
    model = value._value
    if model is None:
        data = value._data
        if data is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("opaque dynamic FlatBuffer has no data")
        return value._tag, data
    entry = value._entry
    if entry is None or entry.tag != value._tag:
        raise TypeError(
            f"unregistered dynamic FlatBuffer model {type(model).__qualname__}"
        )
    data = cast(Any, model).to_flatbuffer()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("dynamic FlatBuffer serializers must return a buffer")
    return entry.tag, data


def dynamic_to_builtins(
    value: DynamicValue,
    *,
    enc_hook: Callable[[Any], Any],
) -> object:
    """Convert a dynamic value for msgspec's JSON and builtin encoders."""

    model = value._value
    if model is None:
        return {
            _MSGSPEC_TAG_FIELD: value._tag,
            _OPAQUE_DATA_FIELD: value._data,
        }
    return {
        _MSGSPEC_TAG_FIELD: value._tag,
        _DYNAMIC_VALUE_FIELD: msgspec.to_builtins(model, enc_hook=enc_hook),
    }


def dynamic_from_builtins(
    value_type: type[DynamicValue],
    value: object,
    *,
    dec_hook: Callable[[Any, Any], Any],
) -> DynamicValue:
    """Resolve a tagged builtin object through the global registry."""

    if not isinstance(value, dict):
        raise TypeError("dynamic FlatBuffer values must decode from an object")
    tag = value.get(_MSGSPEC_TAG_FIELD)
    if not isinstance(tag, str):
        raise ValueError(
            f"dynamic FlatBuffer object is missing {_MSGSPEC_TAG_FIELD!r}"
    )
    value_type._require_allowed(tag)
    if _OPAQUE_DATA_FIELD in value:
        if value.keys() != _OPAQUE_DYNAMIC_FIELDS:
            raise ValueError(
                "opaque dynamic FlatBuffer objects have unexpected fields"
            )
        data = msgspec.convert(value[_OPAQUE_DATA_FIELD], type=bytes)
        return value_type.opaque(tag, data)

    if value.keys() != _KNOWN_DYNAMIC_FIELDS:
        raise ValueError("dynamic FlatBuffer objects must contain type and value")

    entry = dynamic_types.lookup_tag(tag)
    if entry is None:
        raise ValueError(f"unregistered dynamic FlatBuffer tag {tag!r}")
    model = msgspec.convert(
        value[_DYNAMIC_VALUE_FIELD],
        type=entry.model_type,
        dec_hook=dec_hook,
    )
    return value_type._known(entry, model)


dynamic_types = _DynamicTypeRegistry()


__all__ = [
    "DynamicType",
    "DynamicValue",
    "DynamicView",
    "dynamic_allow_prefix",
    "dynamic_from_builtins",
    "dynamic_to_builtins",
    "dynamic_types",
    "encode_dynamic",
    "register_dynamic_module",
    "register_dynamic_type",
]
