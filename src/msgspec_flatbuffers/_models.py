"""Model subclass resolution shared by generated modules."""

from __future__ import annotations

from functools import cache
from types import UnionType
from typing import Any, TypeGuard, Union, get_args, get_origin

import msgspec

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
    generated_fields = {field.name for field in msgspec.structs.fields(generated)}
    replacement_fields = {field.name for field in msgspec.structs.fields(replacement)}
    if replacement_fields != generated_fields:
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

    def collect(
        self,
        generated: type[msgspec.Struct],
        requested: type[msgspec.Struct],
    ) -> None:
        pair = (generated, requested)
        if pair in self._visited:
            return
        validate_model_subclass(generated, requested)
        self._visited.add(pair)

        requested_fields = {
            field.name: field for field in msgspec.structs.fields(requested)
        }
        for generated_field in msgspec.structs.fields(generated):
            requested_field = requested_fields[generated_field.name]
            self._collect_field(
                generated,
                requested,
                generated_field.name,
                generated_field.type,
                requested_field.type,
            )

    def _collect_field(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated_annotation: Any,
        requested_annotation: Any,
    ) -> None:
        if _is_struct_type(generated_annotation):
            self._collect_struct(
                generated_parent,
                requested_parent,
                field_name,
                generated_annotation,
                requested_annotation,
            )
            return

        if get_origin(generated_annotation) in (list, dict):
            self._collect_container_value(
                generated_parent,
                requested_parent,
                field_name,
                generated_annotation,
                requested_annotation,
            )
            return

        if _is_union(generated_annotation):
            self._collect_union(
                generated_parent,
                requested_parent,
                field_name,
                generated_annotation,
                requested_annotation,
            )

    def _collect_struct(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated: type[msgspec.Struct],
        requested: Any,
    ) -> None:
        if not _is_struct_type(requested):
            raise TypeError(
                f"{requested_parent.__qualname__}.{field_name} must select a "
                f"subclass of {generated.__qualname__}"
            )
        if requested is not generated:
            self.bindings.append(
                (
                    generated_parent,
                    requested_parent,
                    field_name,
                    generated,
                    requested,
                )
            )
        self.collect(generated, requested)

    def _collect_container_value(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated_annotation: Any,
        requested_annotation: Any,
    ) -> None:
        origin = get_origin(generated_annotation)
        generated_args = get_args(generated_annotation)
        requested_args = get_args(requested_annotation)
        if get_origin(requested_annotation) is not origin or len(requested_args) != len(
            generated_args
        ):
            _reject_model_shape_change(
                generated_annotation,
                requested_parent,
                field_name,
            )
            return
        if origin is dict:
            if requested_args[0] != generated_args[0]:
                raise TypeError(
                    f"{requested_parent.__qualname__}.{field_name} has an "
                    "incompatible model-mapping key annotation"
                )
            generated_value = generated_args[1]
            requested_value = requested_args[1]
        else:
            generated_value = generated_args[0]
            requested_value = requested_args[0]
        self._collect_field(
            generated_parent,
            requested_parent,
            field_name,
            generated_value,
            requested_value,
        )

    def _collect_union(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated_annotation: Any,
        requested_annotation: Any,
    ) -> None:
        if not _is_union(requested_annotation):
            _reject_model_shape_change(
                generated_annotation,
                requested_parent,
                field_name,
            )
            return

        requested_args = get_args(requested_annotation)
        requested_structs = tuple(
            item for item in requested_args if _is_struct_type(item)
        )
        for generated_item in get_args(generated_annotation):
            if get_origin(generated_item) in (list, dict):
                self._collect_union_container(
                    generated_parent,
                    requested_parent,
                    field_name,
                    generated_item,
                    requested_args,
                )
                continue
            if not _is_struct_type(generated_item):
                continue
            matches = [
                item for item in requested_structs if issubclass(item, generated_item)
            ]
            if len(matches) != 1:
                raise TypeError(
                    f"{requested_parent.__qualname__}.{field_name} must contain "
                    f"exactly one subclass of {generated_item.__qualname__}"
                )
            self._collect_field(
                generated_parent,
                requested_parent,
                field_name,
                generated_item,
                matches[0],
            )

    def _collect_union_container(
        self,
        generated_parent: type[msgspec.Struct],
        requested_parent: type[msgspec.Struct],
        field_name: str,
        generated_annotation: Any,
        requested_args: tuple[Any, ...],
    ) -> None:
        origin = get_origin(generated_annotation)
        matches = tuple(item for item in requested_args if get_origin(item) is origin)
        if len(matches) != 1:
            kind = "vector" if origin is list else "mapping"
            raise TypeError(
                f"{requested_parent.__qualname__}.{field_name} has an "
                f"incompatible model-{kind} annotation"
            )
        self._collect_field(
            generated_parent,
            requested_parent,
            field_name,
            generated_annotation,
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
    resolver.collect(generated, requested)
    return plan.model_types(generated, requested, resolver.bindings)


def _reject_model_shape_change(
    generated: Any,
    requested_parent: type[msgspec.Struct],
    field_name: str,
) -> None:
    if _contains_struct(generated):
        raise TypeError(
            f"{requested_parent.__qualname__}.{field_name} has an incompatible "
            "model annotation"
        )


def _contains_struct(value: Any) -> bool:
    if _is_struct_type(value):
        return True
    return any(_contains_struct(item) for item in get_args(value))


def _is_struct_type(value: Any) -> TypeGuard[type[msgspec.Struct]]:
    return isinstance(value, type) and issubclass(value, msgspec.Struct)


def _is_union(value: Any) -> bool:
    return get_origin(value) in (Union, UnionType)


__all__ = ["resolve_model_types", "validate_model_subclass"]
