"""Model subclass resolution shared by generated modules."""

from __future__ import annotations

from functools import cache

import msgspec
from msgspec.inspect import ListType, StructType, Type, UnionType, type_info

from ._native import NativeModelTypes, NativePlan

_Binding = tuple[
    type[msgspec.Struct],
    type[msgspec.Struct],
    str,
    type[msgspec.Struct],
    type[msgspec.Struct],
]


def validate_model_subclass(
    generated: type[msgspec.Struct],
    replacement: type[msgspec.Struct],
) -> None:
    """Validate a schema-compatible msgspec model subclass."""

    if not isinstance(generated, type) or not issubclass(generated, msgspec.Struct):
        raise TypeError("generated model types must extend msgspec.Struct")
    if not isinstance(replacement, type) or not issubclass(replacement, generated):
        raise TypeError(
            f"{getattr(replacement, '__qualname__', replacement)!s} must subclass "
            f"{generated.__qualname__}"
        )
    if set(replacement.__struct_fields__) != set(generated.__struct_fields__):
        raise TypeError(
            f"{replacement.__qualname__} changes the serialized msgspec fields of "
            f"{generated.__qualname__}; use ClassVar or dict=True for "
            "application-only state"
        )


class _ModelResolver:
    __slots__ = ("_visited", "bindings")

    def __init__(self) -> None:
        self.bindings: list[_Binding] = []
        self._visited: set[tuple[type[msgspec.Struct], type[msgspec.Struct]]] = set()

    def collect(self, generated: Type, requested: Type) -> None:
        if not isinstance(generated, StructType) or not isinstance(
            requested, StructType
        ):
            raise TypeError("FlatBuffer model targets must be msgspec Struct types")

        generated_type = generated.cls
        requested_type = requested.cls
        pair = (generated_type, requested_type)
        if pair in self._visited:
            return
        validate_model_subclass(generated_type, requested_type)
        self._visited.add(pair)

        requested_fields = {field.name: field for field in requested.fields}
        for generated_field in generated.fields:
            requested_field = requested_fields[generated_field.name]
            self._collect_field(
                generated_type,
                requested_type,
                generated_field.name,
                generated_field.type,
                requested_field.type,
            )

    def _collect_field(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated: Type,
        requested: Type,
    ) -> None:
        if isinstance(generated, StructType):
            if not isinstance(requested, StructType):
                raise TypeError(
                    f"{requested_parent.__qualname__}.{field_name} must select a "
                    f"subclass of {generated.cls.__qualname__}"
                )
            if requested.cls is not generated.cls:
                self.bindings.append(
                    (
                        generated_parent,
                        requested_parent,
                        field_name,
                        generated.cls,
                        requested.cls,
                    )
                )
            self.collect(generated, requested)
            return

        if isinstance(generated, ListType):
            if not isinstance(requested, ListType):
                _reject_model_shape_change(generated, requested_parent, field_name)
                return
            self._collect_field(
                generated_parent,
                requested_parent,
                field_name,
                generated.item_type,
                requested.item_type,
            )
            return

        if not isinstance(generated, UnionType):
            return

        if not isinstance(requested, UnionType):
            _reject_model_shape_change(generated, requested_parent, field_name)
            return

        requested_structs = tuple(
            item for item in requested.types if isinstance(item, StructType)
        )
        requested_lists = tuple(
            item for item in requested.types if isinstance(item, ListType)
        )
        for generated_item in generated.types:
            if isinstance(generated_item, ListType):
                if len(requested_lists) != 1:
                    raise TypeError(
                        f"{requested_parent.__qualname__}.{field_name} has an "
                        "incompatible model-vector annotation"
                    )
                self._collect_field(
                    generated_parent,
                    requested_parent,
                    field_name,
                    generated_item,
                    requested_lists[0],
                )
                continue
            if not isinstance(generated_item, StructType):
                continue
            matches = [
                item
                for item in requested_structs
                if issubclass(item.cls, generated_item.cls)
            ]
            if len(matches) != 1:
                raise TypeError(
                    f"{requested_parent.__qualname__}.{field_name} must contain "
                    f"exactly one subclass of {generated_item.cls.__qualname__}"
                )
            self._collect_field(
                generated_parent,
                requested_parent,
                field_name,
                generated_item,
                matches[0],
            )


@cache
def resolve_model_types(
    plan: NativePlan,
    generated: type[msgspec.Struct],
    requested: type[msgspec.Struct],
) -> NativeModelTypes | None:
    """Resolve and cache one requested msgspec model graph."""

    if requested is generated:
        return None
    resolver = _ModelResolver()
    resolver.collect(type_info(generated), type_info(requested))
    return plan.model_types(generated, requested, resolver.bindings)


def _reject_model_shape_change(
    generated: Type,
    requested_parent: type[msgspec.Struct],
    field_name: str,
) -> None:
    if _contains_struct(generated):
        raise TypeError(
            f"{requested_parent.__qualname__}.{field_name} has an incompatible "
            "model annotation"
        )


def _contains_struct(value: Type) -> bool:
    if isinstance(value, StructType):
        return True
    if isinstance(value, ListType):
        return _contains_struct(value.item_type)
    if isinstance(value, UnionType):
        return any(_contains_struct(item) for item in value.types)
    return False


__all__ = ["resolve_model_types", "validate_model_subclass"]
