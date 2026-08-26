"""Scoped application model overrides for dynamic FlatBuffer payloads."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable
from typing import Any, Self, overload

import msgspec

from ._conversion import dec_hook as default_dec_hook
from ._dynamic import DynamicValue, dynamic_from_builtins, dynamic_types
from ._models import validate_model_subclass


class DynamicModelOverrides(UserDict[type[msgspec.Struct], type[msgspec.Struct]]):
    """Validated dynamic generated-model to application-subclass mappings."""

    def __setitem__(
        self,
        key: type[msgspec.Struct],
        item: type[msgspec.Struct],
    ) -> None:
        _validate_dynamic_override(key, item)
        self.data[key] = item

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
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
        """Add an override directly or decorate its replacement class."""

        if replacement is not None:
            self[generated] = replacement
            return replacement

        def register(value: type[_ModelT]) -> type[_ModelT]:
            self[generated] = value
            return value

        return register

    def dec_hook(self, annotation: Any, value: Any) -> Any:
        """Decode package values using these dynamic model overrides."""

        if isinstance(annotation, type) and issubclass(annotation, DynamicValue):
            return dynamic_from_builtins(
                annotation,
                value,
                dec_hook=self.dec_hook,
                dynamic_overrides=self,
            )
        return default_dec_hook(annotation, value)


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
