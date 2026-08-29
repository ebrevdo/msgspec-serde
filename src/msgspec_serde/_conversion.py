"""Private msgspec fallback conversion for generated model values."""

from __future__ import annotations

from functools import cache
from typing import Any, get_args, get_origin

import numpy as np
import numpy.typing as npt

from ._dynamic import DynamicValue, dynamic_from_builtins, dynamic_to_builtins


@cache
def _ndarray_dtype(annotation: Any) -> np.dtype[Any] | None:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is npt.NDArray:
        if len(arguments) != 1:
            raise TypeError(f"ndarray annotation {annotation!r} has no dtype")
        return np.dtype(arguments[0])
    if origin is not np.ndarray:
        return None
    if len(arguments) < 2:
        raise TypeError(f"ndarray annotation {annotation!r} has no dtype")
    dtype_arguments = get_args(arguments[-1])
    if len(dtype_arguments) != 1:
        raise TypeError(f"ndarray annotation {annotation!r} has no precise dtype")
    return np.dtype(dtype_arguments[0])


def encode_fallback(value: Any) -> Any:
    """Convert package model values for a msgspec fallback field."""

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError("numeric vector model fields must be one-dimensional")
        if value.dtype.kind not in "biuf":
            raise TypeError(f"unsupported numeric vector dtype {value.dtype}")
        return value.tolist()
    if isinstance(value, DynamicValue):
        return dynamic_to_builtins(value, enc_hook=encode_fallback)
    raise NotImplementedError


def decode_fallback(annotation: Any, value: Any) -> Any:
    """Restore package model values decoded by a msgspec fallback field."""

    if isinstance(annotation, type) and issubclass(annotation, DynamicValue):
        return dynamic_from_builtins(annotation, value, dec_hook=decode_fallback)

    dtype = _ndarray_dtype(annotation)
    if dtype is None:
        raise NotImplementedError
    result = np.asarray(value, dtype=dtype, order="C")
    if not result.flags.owndata or not result.flags.writeable:
        result = result.copy(order="C")
    if result.ndim != 1:
        raise TypeError("numeric vector model fields must be one-dimensional")
    return result


__all__ = ["decode_fallback", "encode_fallback"]
