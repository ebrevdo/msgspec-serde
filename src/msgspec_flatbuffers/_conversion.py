"""Msgspec hooks for generated materialized models with NumPy arrays."""

from __future__ import annotations

from typing import Any, get_args, get_origin

import numpy as np

from ._dynamic import DynamicValue, dynamic_from_builtins, dynamic_to_builtins


def _ndarray_dtype(annotation: Any) -> np.dtype[Any] | None:
    if get_origin(annotation) is not np.ndarray:
        return None
    arguments = get_args(annotation)
    if len(arguments) < 2:
        raise TypeError(f"ndarray annotation {annotation!r} has no dtype")
    dtype_arguments = get_args(arguments[-1])
    if len(dtype_arguments) != 1:
        raise TypeError(f"ndarray annotation {annotation!r} has no precise dtype")
    return np.dtype(dtype_arguments[0])


def enc_hook(value: Any) -> Any:
    """Encode package model values as msgspec-compatible values."""

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError("numeric vector model fields must be one-dimensional")
        if value.dtype.kind not in "biuf":
            raise TypeError(f"unsupported numeric vector dtype {value.dtype}")
        return value.tolist()
    if isinstance(value, DynamicValue):
        return dynamic_to_builtins(value, enc_hook=enc_hook)
    raise NotImplementedError


def dec_hook(annotation: Any, value: Any) -> Any:
    """Decode package model values from msgspec-compatible values."""

    if isinstance(annotation, type) and issubclass(annotation, DynamicValue):
        return dynamic_from_builtins(annotation, value, dec_hook=dec_hook)

    dtype = _ndarray_dtype(annotation)
    if dtype is None:
        raise NotImplementedError
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if result.ndim != 1:
        raise TypeError("numeric vector model fields must be one-dimensional")
    return result


__all__ = ["dec_hook", "enc_hook"]
