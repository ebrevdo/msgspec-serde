"""Scoped application model overrides for dynamic FlatBuffer payloads."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable
from typing import Any, Self, overload

import msgspec

from ._conversion import decode_fallback
from ._dynamic import DynamicValue, dynamic_from_builtins, dynamic_types
from ._models import validate_model_subclass


class DynamicModelOverrides(UserDict[type[msgspec.Struct], type[msgspec.Struct]]):
    """Map generated dynamic models to application subclasses.

    Keys must be registered generated models. Values must be structurally
    compatible subclasses of their keys.

    Example:
        Register an application subclass for dynamic decoding:

        >>> overrides = DynamicModelOverrides()
        >>> overrides[Monster] = AppMonster
        >>> decoded = flatbuffer.decode(
        ...     buffer, type=Envelope, dynamic_overrides=overrides
        ... )
        >>> isinstance(decoded.payload.value, AppMonster)
        True
    """

    def __setitem__(
        self,
        key: type[msgspec.Struct],
        item: type[msgspec.Struct],
    ) -> None:
        _validate_dynamic_override(key, item)
        self.data[key] = item

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        """Validate and add several model overrides atomically.

        Args:
            other: A mapping or iterable of generated-model and subclass pairs.
            **kwargs: Additional pairs accepted for compatibility with ``dict``.

        Raises:
            TypeError: A key is unregistered or a value is incompatible.

        Example:
            Add overrides from a mapping:

            >>> overrides = DynamicModelOverrides()
            >>> overrides.update({Monster: AppMonster})
            >>> overrides[Monster] is AppMonster
            True
        """

        entries: dict[Any, Any] = dict(other)
        entries.update(kwargs)
        for generated, replacement in entries.items():
            _validate_dynamic_override(generated, replacement)
        self.data.update(entries)

    def __ior__(self, other: Any) -> Self:
        self.update(other)
        return self

    @overload
    def override[_ModelT: msgspec.Struct](
        self,
        generated: type[_ModelT],
        replacement: type[_ModelT],
        /,
    ) -> type[_ModelT]: ...

    @overload
    def override[_ModelT: msgspec.Struct](
        self,
        generated: type[_ModelT],
        /,
    ) -> Callable[[type[_ModelT]], type[_ModelT]]: ...

    def override[_ModelT: msgspec.Struct](
        self,
        generated: type[_ModelT],
        replacement: type[_ModelT] | None = None,
        /,
    ) -> type[_ModelT] | Callable[[type[_ModelT]], type[_ModelT]]:
        """Add an override directly or decorate its replacement class.

        Args:
            generated: The registered generated model being replaced.
            replacement: A compatible application subclass. When omitted, the
                returned decorator registers the decorated class.

        Returns:
            The replacement class, or a decorator when ``replacement`` is
            omitted.

        Raises:
            TypeError: The generated model is unregistered or the replacement
                is incompatible.

        Example:
            Register a subclass with the decorator form:

            >>> overrides = DynamicModelOverrides()
            >>> @overrides.override(Monster)
            ... class AppMonster(Monster):
            ...     pass
            >>> overrides[Monster] is AppMonster
            True
        """

        if replacement is not None:
            self[generated] = replacement
            return replacement

        def register(value: type[_ModelT]) -> type[_ModelT]:
            self[generated] = value
            return value

        return register

    def dec_hook(self, annotation: Any, value: Any) -> Any:
        """Decode a value while applying these dynamic model overrides.

        Args:
            annotation: The target type requested by msgspec.
            value: The builtin value to convert.

        Returns:
            The converted package value.

        Example:
            Pass the bound method to a msgspec decoder:

            >>> decoder = msgspec.json.Decoder(
            ...     type=Envelope, dec_hook=overrides.dec_hook
            ... )
            >>> envelope = decoder.decode(data)
        """

        if isinstance(annotation, type) and issubclass(annotation, DynamicValue):
            return dynamic_from_builtins(
                annotation,
                value,
                dec_hook=self.dec_hook,
                dynamic_overrides=self,
            )
        return decode_fallback(annotation, value)


def _validate_dynamic_override(
    generated: type[msgspec.Struct],
    replacement: type[msgspec.Struct],
) -> None:
    validate_model_subclass(generated, replacement)
    entry = dynamic_types.lookup_model(generated)
    if entry is None or entry.model_type is not generated:
        raise TypeError(
            f"{generated.__qualname__} is not a registered dynamic FlatBuffer model"
        )


__all__ = ["DynamicModelOverrides"]
