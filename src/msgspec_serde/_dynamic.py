"""Registry-backed open polymorphism for nested FlatBuffers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import msgspec

from ._flatbuffer import _model_binding
from ._flatbuffer import decode as decode_flatbuffer
from ._flatbuffer import encode as encode_flatbuffer
from ._models import validate_model_subclass
from ._runtime import TableView

if TYPE_CHECKING:
    from ._overrides import DynamicModelOverrides

_MSGSPEC_TAG_FIELD = "__msgspec_serde_type__"
_DYNAMIC_VALUE_FIELD = "value"
_OPAQUE_DATA_FIELD = "__msgspec_serde_data__"
_KNOWN_DYNAMIC_FIELDS = frozenset((_MSGSPEC_TAG_FIELD, _DYNAMIC_VALUE_FIELD))
_OPAQUE_DYNAMIC_FIELDS = frozenset((_MSGSPEC_TAG_FIELD, _OPAQUE_DATA_FIELD))
_ABSENT: Any = object()
_MODEL_RESOLUTION_CACHE_LIMIT = 1024


@dataclass(frozen=True, slots=True)
class DynamicType:
    """Describe one registered dynamic FlatBuffer root table.

    Attributes:
        tag: The stable wire discriminator for the type.
        model_type: The generated materialized model type.
        view_type: The generated lazy root view type.

    Example:
        Inspect a registered type:

        >>> entry = register_dynamic_type("example.Monster", Monster, MonsterView)
        >>> entry.model_type is Monster
        True
    """

    tag: str
    model_type: type[msgspec.Struct]
    view_type: type[TableView]


class _DynamicTypeRegistry:
    """Store dynamic FlatBuffer types in the process-wide registry.

    Use the exported :data:`msgspec_serde.dynamic_types` instance to inspect or
    extend the registry.

    Example:
        Look up a previously registered type:

        >>> dynamic_types.register("example.Monster", Monster, MonsterView)
        DynamicType(...)
        >>> dynamic_types.lookup_tag("example.Monster").model_type is Monster
        True
    """

    __slots__ = (
        "_by_model",
        "_by_tag",
        "_lock",
        "_model_resolution_cache",
        "_modules",
        "_registration_version",
    )

    def __init__(self) -> None:
        self._by_model: dict[type[object], DynamicType] = {}
        self._by_tag: dict[str, DynamicType] = {}
        self._lock = Lock()
        self._modules: dict[str, str] = {}
        self._model_resolution_cache: dict[
            type[object],
            tuple[int, DynamicType | None],
        ] = {}
        self._registration_version = 0

    def register(
        self,
        tag: str,
        model_type: type[msgspec.Struct],
        view_type: type[TableView],
    ) -> DynamicType:
        """Register one type tag, materialized model, and lazy view.

        Args:
            tag: The stable wire discriminator for the type.
            model_type: The generated materialized model type.
            view_type: The generated lazy root view type.

        Returns:
            The new or identical existing registry entry.

        Raises:
            TypeError: A model or view has the wrong base type.
            ValueError: The tag is empty or conflicts with an existing entry.

        Example:
            Register a generated root model and view:

            >>> entry = dynamic_types.register(
            ...     "example.Monster", Monster, MonsterView
            ... )
            >>> entry.tag
            'example.Monster'
        """

        entry = _validated_entry(tag, model_type, view_type)
        with self._lock:
            tag_entry = self._by_tag.get(entry.tag)
            model_entry = self._by_model.get(entry.model_type)
            if (tag_entry is not None and tag_entry != entry) or (
                model_entry is not None and model_entry != entry
            ):
                raise ValueError(
                    "dynamic FlatBuffer registration conflicts with an existing "
                    "tag or model"
                )
            existing = tag_entry or model_entry
            if existing is not None:
                return existing
            self._by_tag[entry.tag] = entry
            self._by_model[entry.model_type] = entry
            self._registration_version += 1
        return entry

    def register_module(self, tag: str, module: str) -> str:
        """Associate a trusted type tag with a lazily imported module.

        Args:
            tag: The stable wire discriminator registered by the module.
            module: The importable module name.

        Returns:
            The registered module name.

        Raises:
            ValueError: A value is empty or the tag maps to another module.

        Example:
            Defer importing generated plugin types until they are decoded:

            >>> dynamic_types.register_module(
            ...     "plugins.weather.Report", "plugins.weather.generated"
            ... )
            'plugins.weather.generated'
        """

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
        return module

    def lookup_tag(self, tag: str) -> DynamicType | None:
        """Find a tag, importing its trusted module when necessary.

        Args:
            tag: The stable wire discriminator to find.

        Returns:
            The registered dynamic type, or ``None`` when unknown.

        Raises:
            RuntimeError: A configured module does not register the requested
                tag when imported.

        Example:
            Check whether a dynamic type is available:

            >>> dynamic_types.lookup_tag("example.Missing") is None
            True
        """

        entry = self._by_tag.get(tag)
        if entry is not None:
            return entry
        return self._load_tag(tag)

    def lookup_model(self, model_type: type[object]) -> DynamicType | None:
        """Find the entry for a registered model or unambiguous subclass.

        Args:
            model_type: The materialized model type to find.

        Returns:
            The matching dynamic type, or ``None`` when unknown.

        Raises:
            TypeError: The model inherits from multiple registered bases.

        Example:
            Resolve a registered model type:

            >>> dynamic_types.lookup_model(Monster).tag
            'example.Monster'
        """

        entry = self._by_model.get(model_type)
        if entry is not None:
            return entry

        version = self._registration_version
        cached = self._model_resolution_cache.get(model_type)
        if cached is not None:
            cached_version, cached_resolution = cached
            if cached_version == version:
                return cached_resolution

        resolved: DynamicType | None = None
        for base in getattr(model_type, "__mro__", ())[1:]:
            candidate = self._by_model.get(base)
            if candidate is None:
                continue
            if resolved is not None and candidate != resolved:
                raise TypeError(
                    f"dynamic FlatBuffer model {model_type.__qualname__} has "
                    "multiple registered bases"
                )
            resolved = candidate

        cache = self._model_resolution_cache
        if model_type in cache or len(cache) < _MODEL_RESOLUTION_CACHE_LIMIT:
            cache[model_type] = (version, resolved)
        return resolved

    def _load_tag(self, tag: str) -> DynamicType | None:
        module = self._modules.get(tag)
        if module is None:
            return None

        import_module(module)

        entry = self._by_tag.get(tag)
        if entry is None:
            raise RuntimeError(
                f"dynamic FlatBuffer module {module!r} did not register {tag!r}"
            )
        return entry


class DynamicValue:
    """Hold a registered model or a forwardable opaque dynamic payload.

    Args:
        value: An instance of a registered generated model.

    Raises:
        TypeError: The model type has not been registered.

    Example:
        Wrap a registered model for a dynamic field:

        >>> register_dynamic_type("example.Monster", Monster, MonsterView)
        DynamicType(...)
        >>> dynamic = DynamicValue(Monster(name="Orc"))
        >>> dynamic.tag
        'example.Monster'
    """

    __slots__ = ("_data", "_entry", "_tag", "_value")
    _allowed_prefix: ClassVar[str | None] = None

    def __init__(self, value: msgspec.Struct) -> None:
        model_type = type(value)
        entry = dynamic_types.lookup_model(model_type)
        if entry is None:
            raise TypeError(
                f"unregistered dynamic FlatBuffer model {model_type.__qualname__}"
            )
        if model_type is not entry.model_type:
            validate_model_subclass(entry.model_type, model_type)
        self._tag = entry.tag
        self._entry: DynamicType | None = entry
        self._value: msgspec.Struct | None = value
        self._data: bytes | None = None

    @classmethod
    def opaque(cls, tag: str, data: bytes | bytearray | memoryview) -> Self:
        """Create a forwardable value for an unknown type tag.

        Args:
            tag: The unknown wire discriminator.
            data: The nested FlatBuffer bytes to preserve.

        Returns:
            A dynamic value that retains the tag and bytes without decoding.

        Raises:
            ValueError: The tag is empty or outside the allowed namespace.

        Example:
            Preserve a payload whose schema is not installed:

            >>> value = DynamicValue.opaque("plugins.Future", b"payload")
            >>> value.data
            b'payload'
        """

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
        """The stable wire discriminator.

        Example:
            Inspect a dynamic value's type tag:

            >>> dynamic.tag
            'example.Monster'
        """

        return self._tag

    @property
    def value(self) -> msgspec.Struct | None:
        """The materialized model, or ``None`` for an opaque payload.

        Example:
            Access a known dynamic model:

            >>> dynamic.value.name
            'Orc'
        """

        return self._value

    @property
    def data(self) -> bytes | None:
        """The preserved bytes, or ``None`` for a known model.

        Example:
            Access the bytes of an opaque value:

            >>> DynamicValue.opaque("plugins.Future", b"payload").data
            b'payload'
        """

        return self._data

    @property
    def is_known(self) -> bool:
        """Whether this value contains a registered materialized model.

        Example:
            Distinguish known values from opaque payloads:

            >>> DynamicValue.opaque("plugins.Future", b"payload").is_known
            False
        """

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
    """Resolve a nested FlatBuffer lazily through the dynamic registry.

    Args:
        tag: The nested value's wire discriminator.
        data: A read-only view of the nested FlatBuffer bytes.
        value_type: The dynamic value class produced by :meth:`to_model`.

    Raises:
        TypeError: ``value_type`` is invalid or ``data`` is not read-only.
        ValueError: The tag is empty or outside the value type's namespace.

    Example:
        Create a lazy view over a nested dynamic payload:

        >>> nested = memoryview(payload).toreadonly()
        >>> view = DynamicView("example.Monster", nested)
        >>> view.tag
        'example.Monster'
    """

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
        """The nested value's stable wire discriminator.

        Example:
            Inspect the unresolved type tag:

            >>> view.tag
            'example.Monster'
        """

        return self._tag

    @property
    def data(self) -> memoryview:
        """The read-only nested FlatBuffer bytes.

        Example:
            Forward the nested bytes without decoding them:

            >>> send(view.data)
        """

        return self._data

    @property
    def value(self) -> TableView | None:
        """The resolved lazy table view, or ``None`` for an unknown tag.

        Example:
            Resolve a registered payload on first access:

            >>> isinstance(view.value, MonsterView)
            True
        """

        resolved = self._resolve_view()
        if resolved is None:
            return None
        return resolved[1]

    @property
    def is_known(self) -> bool:
        """Whether the tag resolves to a registered dynamic type.

        Example:
            Check a tag before materializing its payload:

            >>> view.is_known
            True
        """

        return self._resolve_entry() is not None

    def to_model(
        self,
        *,
        dynamic_overrides: DynamicModelOverrides | None = None,
    ) -> DynamicValue:
        """Materialize the nested payload as a dynamic value.

        Args:
            dynamic_overrides: Optional generated-model to application-subclass
                mappings used for nested dynamic values.

        Returns:
            A known dynamic model, or an opaque value when the tag is unknown.

        Raises:
            TypeError: A generated view returns a model of the wrong type.

        Example:
            Convert a lazy dynamic view into an owned model:

            >>> dynamic = view.to_model()
            >>> isinstance(dynamic.value, Monster)
            True
        """

        resolved = self._resolve_view()
        if resolved is None:
            return self._value_type.opaque(self._tag, self._data)
        entry, view = resolved
        target_type = _dynamic_model_type(entry, dynamic_overrides)
        materialize = cast(Any, view).to_model
        if dynamic_overrides is None:
            model = materialize()
        else:
            model = materialize(
                target_type,
                dynamic_overrides=dynamic_overrides,
            )
        model_type = type(model)
        if model_type is not target_type:
            raise TypeError(
                f"{entry.view_type.__qualname__}.to_model() returned "
                f"{model_type.__qualname__}, expected {target_type.__qualname__}"
            )
        return self._value_type._known(entry, model)

    def _resolve_view(self) -> tuple[DynamicType, TableView] | None:
        entry = self._resolve_entry()
        if entry is None:
            return None
        cached = self._value
        if cached is _ABSENT:
            cached = decode_flatbuffer(self._data, type=entry.view_type)
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
    if prefix is None:
        return
    if tag.startswith(prefix) and len(tag) > len(prefix):
        return
    raise ValueError(f"dynamic FlatBuffer tag {tag!r} is outside {prefix + '*'!r}")


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
        raise ValueError("dynamic_allow must be a nonempty namespace followed by '.*'")
    return pattern[:-1]


def register_dynamic_module(tag: str, module: str) -> str:
    """Register a trusted module for lazy loading of one type tag.

    Args:
        tag: The stable wire discriminator registered by the module.
        module: The importable generated module name.

    Returns:
        The registered module name.

    Raises:
        ValueError: A value is empty or the tag maps to another module.

    Example:
        Register a generated plugin module without importing it immediately:

        >>> register_dynamic_module(
        ...     "plugins.weather.Report", "plugins.weather.generated"
        ... )
        'plugins.weather.generated'
    """

    return dynamic_types.register_module(tag, module)


def register_dynamic_type(
    tag: str,
    model_type: type[msgspec.Struct],
    view_type: type[TableView],
) -> DynamicType:
    """Register a generated dynamic model and root view.

    Args:
        tag: The stable wire discriminator for the type.
        model_type: The registered generated model type.
        view_type: The matching generated root view type.

    Returns:
        The new or identical existing registry entry.

    Raises:
        TypeError: The model is not a registered FlatBuffer model or the view
            has the wrong base type.
        ValueError: The tag is empty or conflicts with an existing entry.

    Example:
        Register a generated root type for dynamic fields:

        >>> entry = register_dynamic_type("example.Monster", Monster, MonsterView)
        >>> entry.tag
        'example.Monster'
    """

    _model_binding(model_type)
    return dynamic_types.register(tag, model_type, view_type)


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
    data = encode_flatbuffer(model)
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
    dynamic_overrides: DynamicModelOverrides | None = None,
) -> DynamicValue:
    """Resolve a tagged builtin object through the global registry."""

    if not isinstance(value, dict):
        raise TypeError("dynamic FlatBuffer values must decode from an object")
    tag = value.get(_MSGSPEC_TAG_FIELD)
    if not isinstance(tag, str):
        raise ValueError(f"dynamic FlatBuffer object is missing {_MSGSPEC_TAG_FIELD!r}")
    value_type._require_allowed(tag)
    if _OPAQUE_DATA_FIELD in value:
        if value.keys() != _OPAQUE_DYNAMIC_FIELDS:
            raise ValueError("opaque dynamic FlatBuffer objects have unexpected fields")
        data = msgspec.convert(value[_OPAQUE_DATA_FIELD], type=bytes)
        return value_type.opaque(tag, data)

    if value.keys() != _KNOWN_DYNAMIC_FIELDS:
        raise ValueError("dynamic FlatBuffer objects must contain type and value")

    entry = dynamic_types.lookup_tag(tag)
    if entry is None:
        raise ValueError(f"unregistered dynamic FlatBuffer tag {tag!r}")
    model = msgspec.convert(
        value[_DYNAMIC_VALUE_FIELD],
        type=_dynamic_model_type(entry, dynamic_overrides),
        dec_hook=dec_hook,
    )
    return value_type._known(entry, model)


def _dynamic_model_type(
    entry: DynamicType,
    dynamic_overrides: DynamicModelOverrides | None,
) -> type[msgspec.Struct]:
    if dynamic_overrides is None:
        return entry.model_type
    return dynamic_overrides.get(entry.model_type, entry.model_type)


dynamic_types = _DynamicTypeRegistry()


__all__ = [
    "DynamicType",
    "DynamicValue",
    "DynamicView",
    "dynamic_allow_prefix",
    "dynamic_from_builtins",
    "dynamic_to_builtins",
    "dynamic_types",
    "register_dynamic_module",
    "register_dynamic_type",
]
