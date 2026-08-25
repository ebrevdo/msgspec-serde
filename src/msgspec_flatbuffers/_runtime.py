"""Runtime primitives shared by generated read-only FlatBuffers views."""

from __future__ import annotations

import struct
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from enum import IntEnum
from typing import Any, ClassVar, Generic, Self, TypeVar, overload

import flatbuffers
import numpy as np
import numpy.typing as npt

_BOOL = struct.Struct("<?")
_INT8 = struct.Struct("<b")
_UINT8 = struct.Struct("<B")
_INT16 = struct.Struct("<h")
_UINT16 = struct.Struct("<H")
_INT32 = struct.Struct("<i")
_UINT32 = struct.Struct("<I")
_INT64 = struct.Struct("<q")
_UINT64 = struct.Struct("<Q")
_FLOAT32 = struct.Struct("<f")
_FLOAT64 = struct.Struct("<d")
_VTABLE_HEADER_SIZE = _UINT16.size * 2

_SCALAR_FORMATS: dict[str, struct.Struct] = {
    "bool": _BOOL,
    "int8": _INT8,
    "uint8": _UINT8,
    "int16": _INT16,
    "uint16": _UINT16,
    "int32": _INT32,
    "uint32": _UINT32,
    "int64": _INT64,
    "uint64": _UINT64,
    "float32": _FLOAT32,
    "float64": _FLOAT64,
}

_SCALAR_PREPENDERS: dict[str, str] = {
    "bool": "PrependBool",
    "int8": "PrependInt8",
    "uint8": "PrependUint8",
    "int16": "PrependInt16",
    "uint16": "PrependUint16",
    "int32": "PrependInt32",
    "uint32": "PrependUint32",
    "int64": "PrependInt64",
    "uint64": "PrependUint64",
    "float32": "PrependFloat32",
    "float64": "PrependFloat64",
}

_SCALAR_DTYPES: dict[str, np.dtype[Any]] = {
    "bool": np.dtype("?"),
    "int8": np.dtype("i1"),
    "uint8": np.dtype("u1"),
    "int16": np.dtype("<i2"),
    "uint16": np.dtype("<u2"),
    "int32": np.dtype("<i4"),
    "uint32": np.dtype("<u4"),
    "int64": np.dtype("<i8"),
    "uint64": np.dtype("<u8"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}

_VECTOR_DTYPES: dict[str, np.dtype[Any]] = {
    "?": _SCALAR_DTYPES["bool"],
    "<i1": _SCALAR_DTYPES["int8"],
    "<u1": _SCALAR_DTYPES["uint8"],
    "<i2": _SCALAR_DTYPES["int16"],
    "<u2": _SCALAR_DTYPES["uint16"],
    "<i4": _SCALAR_DTYPES["int32"],
    "<u4": _SCALAR_DTYPES["uint32"],
    "<i8": _SCALAR_DTYPES["int64"],
    "<u8": _SCALAR_DTYPES["uint64"],
    "<f4": _SCALAR_DTYPES["float32"],
    "<f8": _SCALAR_DTYPES["float64"],
}

_MISSING: Any = object()
_CACHE_PROMOTION_MIN = 64
_CACHE_PROMOTION_FACTOR = 8
_DENSE_CACHE_MAX = 1024
_NUMPY_VECTOR_MIN = 32
_INTEGER_UNPACKER_FORMATS = {
    ("i", 1): "<b",
    ("u", 1): "<B",
    ("i", 2): "<h",
    ("u", 2): "<H",
    ("i", 4): "<i",
    ("u", 4): "<I",
    ("i", 8): "<q",
    ("u", 8): "<Q",
}

_T = TypeVar("_T")
_ViewT = TypeVar("_ViewT", bound="TableView | StructView")
_UnionT = TypeVar("_UnionT")


class BufferBoundsError(ValueError):
    """Raised when a FlatBuffers offset points outside the backing buffer."""


class InvalidBufferError(ValueError):
    """Raised when FlatBuffers metadata is structurally invalid."""


class UnionDispatch:
    """Validated table alternatives for one generated FlatBuffers union."""

    __slots__ = ("_has_untrusted_types", "_table_types", "name", "none_tag")

    def __init__(
        self,
        name: str,
        none_tag: int,
        table_types: Mapping[int, type[TableView]],
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("FlatBuffers union names cannot be empty")
        if not isinstance(none_tag, int) or isinstance(none_tag, bool):
            raise TypeError("FlatBuffers union NONE tags must be integers")

        alternatives: dict[int, type[TableView]] = {}
        for tag, view_type in table_types.items():
            if not isinstance(tag, int) or isinstance(tag, bool):
                raise TypeError("FlatBuffers union tags must be integers")
            if tag == none_tag:
                raise ValueError("FlatBuffers union NONE cannot have a table type")
            if not isinstance(view_type, type) or not issubclass(
                view_type,
                TableView,
            ):
                raise TypeError(
                    "FlatBuffers table union alternatives require TableView types"
                )
            alternatives[tag] = view_type

        self.name = name
        self.none_tag = none_tag
        self._table_types = alternatives
        self._has_untrusted_types = any(
            view_type.__init__ is not TableView.__init__
            or view_type.__new__ is not object.__new__
            or type(view_type).__call__ is not type.__call__
            for view_type in alternatives.values()
        )


class OpenIntEnum(IntEnum):
    """An integer enum that preserves values unknown to the current schema."""

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if not isinstance(value, int):
            return None
        member = int.__new__(cls, value)
        member._name_ = f"UNKNOWN_{value}"
        member._value_ = value
        return member


def _readonly_bytes(buffer: Buffer) -> memoryview:
    if (
        isinstance(buffer, memoryview)
        and buffer.c_contiguous
        and buffer.ndim == 1
        and buffer.format == "B"
        and buffer.readonly
    ):
        return buffer

    view = memoryview(buffer)
    if not view.c_contiguous:
        raise TypeError("FlatBuffers input must be C-contiguous")
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return view.toreadonly()


def _require_span(
    buffer: memoryview,
    offset: int,
    size: int,
    *,
    description: str,
) -> None:
    buffer_size = len(buffer)
    if offset < 0 or size < 0 or offset > buffer_size - size:
        raise BufferBoundsError(
            f"{description} at offset {offset} with size {size} "
            f"exceeds a {buffer_size}-byte buffer"
        )


def _string_from_offset(buffer: memoryview, offset: int) -> str:
    buffer_size = len(buffer)
    if offset < 0 or offset > buffer_size - _UINT32.size:
        raise BufferBoundsError(
            f"string offset at offset {offset} with size {_UINT32.size} "
            f"exceeds a {buffer_size}-byte buffer"
        )
    relative = _UINT32.unpack_from(buffer, offset)[0]
    if relative == 0:
        raise InvalidBufferError("string offset contains a null offset")
    length_offset = offset + relative
    if length_offset < 0 or length_offset > buffer_size - _UINT32.size:
        raise BufferBoundsError(
            f"string length at offset {length_offset} with size {_UINT32.size} "
            f"exceeds a {buffer_size}-byte buffer"
        )
    length = _UINT32.unpack_from(buffer, length_offset)[0]
    start = length_offset + _UINT32.size
    data_size = length + 1
    if start < 0 or start > buffer_size - data_size:
        raise BufferBoundsError(
            f"string data at offset {start} with size {data_size} "
            f"exceeds a {buffer_size}-byte buffer"
        )
    if buffer[start + length] != 0:
        raise InvalidBufferError("FlatBuffers string is not null-terminated")
    return bytes(buffer[start : start + length]).decode("utf-8")


def build_scalar_vector(
    builder: flatbuffers.Builder,
    values: Sequence[Any] | npt.NDArray[Any],
    scalar_type: str,
) -> int:
    """Build a vector of fixed-width scalar values."""

    length = len(values)
    if isinstance(values, np.ndarray) or length >= _NUMPY_VECTOR_MIN:
        array = _validated_scalar_array(values, scalar_type)
        if array is not None:
            if scalar_type in ("bool", "int8", "uint8"):
                return builder.CreateByteVector(array.tobytes(order="C"))
            return builder.CreateNumpyVector(array)

    scalar = _SCALAR_FORMATS[scalar_type]
    builder.StartVector(scalar.size, length, scalar.size)
    prepend = getattr(builder, _SCALAR_PREPENDERS[scalar_type])
    for value in reversed(values):
        prepend(value)
    return builder.EndVector()


def _validated_scalar_array(
    values: Sequence[Any] | npt.NDArray[Any],
    scalar_type: str,
) -> npt.NDArray[Any] | None:
    """Return a bulk-build array when NumPy preserves scalar validation."""

    array = np.asarray(values)
    if array.ndim != 1:
        return None

    kind = array.dtype.kind
    dtype = _SCALAR_DTYPES[scalar_type]
    if array.dtype == dtype:
        return array

    if scalar_type == "bool":
        if kind not in "biuf":
            return None
        if array.size:
            minimum = array.min()
            maximum = array.max()
            if not bool(minimum >= 0 and maximum <= 1):
                raise TypeError("bad number for type bool")
    elif kind in "biu" and scalar_type not in ("float32", "float64"):
        target_limits = np.iinfo(dtype)
        if kind == "b":
            source_minimum, source_maximum = 0, 1
        else:
            source_limits = np.iinfo(array.dtype)
            source_minimum = source_limits.min
            source_maximum = source_limits.max
        # Preserve NumPy's bool-to-uint64 comparison overflow.
        force_bool_uint64_check = scalar_type == "uint64" and kind == "b"
        requires_value_check = (
            source_minimum < target_limits.min
            or source_maximum > target_limits.max
            or force_bool_uint64_check
        )
        if requires_value_check and array.size:
            minimum = array.min()
            maximum = array.max()
            if bool(
                minimum < target_limits.min or maximum > target_limits.max
            ):
                raise TypeError(f"bad number for type {scalar_type}")
    elif scalar_type in ("float32", "float64"):
        if kind not in "biuf":
            return None
        if scalar_type == "float32" and kind in "biu":
            if array.dtype.itemsize > _FLOAT32.size:
                # Match struct.pack's integer-to-float64-to-float32 rounding.
                return array.astype(_SCALAR_DTYPES["float64"], copy=False).astype(
                    dtype
                )
            return array.astype(dtype, copy=False)
        if (
            scalar_type == "float32"
            and kind == "f"
            and array.dtype.itemsize > dtype.itemsize
            and array.size
        ):
            finite = np.isfinite(array)
            if bool(np.any(np.abs(array[finite]) > np.finfo(dtype).max)):
                raise OverflowError("float too large to pack with f format")
    else:
        return None

    return array.astype(dtype, copy=False)


def build_offset_vector(
    builder: flatbuffers.Builder,
    offsets: Sequence[int],
) -> int:
    """Build a vector of previously constructed FlatBuffers offsets."""

    builder.StartVector(_UINT32.size, len(offsets), _UINT32.size)
    prepend = builder.PrependUOffsetTRelative
    for offset in reversed(offsets):
        prepend(offset)
    return builder.EndVector()


def build_byte_vector(
    builder: flatbuffers.Builder,
    value: Buffer,
) -> int:
    """Build a byte vector from any contiguous buffer without an input copy."""

    view = _readonly_bytes(value)
    length = len(view)
    builder.StartVector(1, length, 1)
    head = builder.Head() - length
    builder.head = head
    builder.Bytes[head : head + length] = view
    return builder.EndVector()


def build_string_vector(
    builder: flatbuffers.Builder,
    values: Sequence[str],
) -> int:
    """Build a vector of UTF-8 strings."""

    return build_offset_vector(
        builder,
        [builder.CreateString(value) for value in values],
    )


def _estimate_sampled_vector_size(
    values: Sequence[_T],
    item_size: Callable[[_T], int],
) -> int:
    """Estimate offset-vector payloads from at most six fixed samples."""

    length = len(values)
    if length == 0:
        return 8
    if length <= 6:
        payload_size = sum(map(item_size, values))
    else:
        middle = length // 2
        payload_size = (
            item_size(values[0])
            + item_size(values[1])
            + item_size(values[middle - 1])
            + item_size(values[middle])
            + item_size(values[-2])
            + item_size(values[-1])
        )
        payload_size = (payload_size * length + 5) // 6
    return 8 + length * _UINT32.size + payload_size


def _estimate_string_vector_size(values: Sequence[str]) -> int:
    return _estimate_sampled_vector_size(values, len) + len(values) * 8


class CachedVector(Sequence[_T], Generic[_T]):
    """A fixed-length vector that strongly caches every accessed element."""

    __slots__ = ("_cache", "_cached_count", "_length")

    def __init__(self, length: int) -> None:
        if length < 0:
            raise ValueError("FlatBuffers vector length cannot be negative")
        self._length = length
        if length == 0:
            cache: dict[int, _T] | list[Any] | tuple[_T, ...] = ()
        elif length <= _DENSE_CACHE_MAX:
            cache = [_MISSING] * length
        else:
            cache = {}
        self._cache = cache
        self._cached_count = 0

    def __len__(self) -> int:
        return self._length

    def _load(self, index: int) -> _T:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[_T, ...]: ...

    def __getitem__(self, index: int | slice) -> _T | tuple[_T, ...]:
        cache = self._cache
        if self._cached_count == self._length:
            if isinstance(index, slice):
                if isinstance(cache, list):
                    cache = tuple(cache)
                    self._cache = cache
                return cache[index]
            return cache[index]

        mutable_cache: dict[int, _T] | list[Any] = (  # ty: ignore[invalid-assignment]
            cache
        )
        cache = mutable_cache
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(self._length)))

        length = self._length
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("FlatBuffers vector index out of range")

        if isinstance(cache, list):
            value = cache[index]
            if value is not _MISSING:
                return value
        else:
            value = cache.get(index, _MISSING)
            if value is not _MISSING:
                return value

        value = self._load(index)
        cache[index] = value
        cached_count = self._cached_count + 1
        self._cached_count = cached_count
        if cached_count == length:
            if not isinstance(cache, list):
                self._cache = [cache[item] for item in range(length)]
        elif not isinstance(cache, list) and (
            cached_count >= _CACHE_PROMOTION_MIN
            and cached_count * _CACHE_PROMOTION_FACTOR >= length
        ):
            self._promote_cache(cache)
        return value

    def __iter__(self) -> Iterator[_T]:
        cache = self._cache
        if self._cached_count == self._length:
            return iter(cache)
        if isinstance(cache, list):
            return self._iter_dense(cache, 0)
        sparse_cache: dict[int, _T] = cache  # ty: ignore[invalid-assignment]
        return self._iter_sparse(sparse_cache)

    def _promote_cache(self, cache: dict[int, _T]) -> list[Any]:
        dense_cache = [_MISSING] * self._length
        for index, value in cache.items():
            dense_cache[index] = value
        self._cache = dense_cache
        return dense_cache

    def _iter_sparse(self, cache: dict[int, _T]) -> Iterator[_T]:
        load = self._load
        get = cache.get
        cached_count = self._cached_count
        length = self._length
        for index in range(length):
            value = get(index, _MISSING)
            if value is _MISSING:
                value = load(index)
                cache[index] = value
                cached_count += 1
                self._cached_count = cached_count
                if (
                    cached_count == length
                    or (
                        cached_count >= _CACHE_PROMOTION_MIN
                        and cached_count * _CACHE_PROMOTION_FACTOR >= length
                    )
                ):
                    dense_cache = self._promote_cache(cache)
                    yield value
                    yield from self._iter_dense(dense_cache, index + 1)
                    return
            yield value

    def _iter_dense(self, cache: list[Any], start: int) -> Iterator[_T]:
        load = self._load
        cached_count = self._cached_count
        for index in range(start, self._length):
            value = cache[index]
            if value is _MISSING:
                value = load(index)
                cache[index] = value
                cached_count += 1
                self._cached_count = cached_count
            yield value

    @property
    def cached_count(self) -> int:
        """Number of vector elements materialized so far."""

        return self._cached_count


class UnionVector(CachedVector[_UnionT], Generic[_UnionT]):
    """Parallel discriminator/payload vectors with cached typed table views."""

    __slots__ = (
        "_buffer",
        "_dispatch",
        "_tag_bytes",
        "_tag_size",
        "_tag_unpack",
        "_types",
        "_value_start",
        "_vtable_cache",
    )

    def __init__(
        self,
        buffer: Buffer,
        types: npt.NDArray[Any],
        value_start: int,
        value_length: int,
        tag_unpacker: struct.Struct,
        dispatch: UnionDispatch,
    ) -> None:
        super().__init__(value_length)
        if not isinstance(types, np.ndarray):
            raise TypeError("FlatBuffers union types must be a NumPy array")
        if types.ndim != 1 or not types.flags.c_contiguous:
            raise TypeError(
                "FlatBuffers union types must be a C-contiguous 1-D array"
            )
        if types.dtype.kind not in "iu":
            raise TypeError("FlatBuffers union discriminators must be integers")
        if (
            types.dtype.itemsize > 1
            and types.dtype != types.dtype.newbyteorder("<")
        ):
            raise TypeError(
                "FlatBuffers union discriminators must be little-endian"
            )
        if types.flags.writeable:
            raise TypeError(
                "FlatBuffers union discriminators must be read-only"
            )
        if not isinstance(tag_unpacker, struct.Struct):
            raise TypeError("FlatBuffers union tag unpackers must be struct.Struct")
        expected_format = _INTEGER_UNPACKER_FORMATS.get(
            (types.dtype.kind, types.dtype.itemsize)
        )
        if expected_format != tag_unpacker.format:
            raise ValueError(
                "FlatBuffers union discriminator dtype and unpacker formats differ"
            )
        if len(types) != value_length:
            raise InvalidBufferError(
                "FlatBuffers union type and value vector lengths differ"
            )
        if not isinstance(dispatch, UnionDispatch):
            raise TypeError("FlatBuffers unions require a UnionDispatch")

        view = _readonly_bytes(buffer)
        _require_span(
            view,
            value_start,
            value_length * _UINT32.size,
            description="union value vector data",
        )
        tag_bytes = memoryview(types).cast("B").toreadonly()

        self._buffer = view
        self._dispatch = dispatch
        self._tag_bytes = tag_bytes
        self._tag_size = tag_unpacker.size
        self._tag_unpack = tag_unpacker.unpack_from
        self._types = types
        self._value_start = value_start
        self._vtable_cache: dict[int, tuple[int, int]] = {}

    def _load(self, index: int) -> _UnionT:
        tag = self._tag_unpack(
            self._tag_bytes,
            index * self._tag_size,
        )[0]
        dispatch = self._dispatch
        offset = self._value_start + index * _UINT32.size
        relative = _UINT32.unpack_from(self._buffer, offset)[0]
        if tag == dispatch.none_tag:
            raise InvalidBufferError(
                f"{dispatch.name} union vectors cannot contain NONE "
                f"at index {index}"
            )

        view_type = dispatch._table_types.get(tag)
        if view_type is None:
            raise InvalidBufferError(
                f"unknown {dispatch.name} discriminator {tag} at index {index}"
            )
        if relative == 0:
            raise InvalidBufferError(
                f"{dispatch.name} discriminator {tag} at index {index} "
                "contains a null offset"
            )
        target = offset + relative
        buffer = self._buffer
        buffer_size = len(buffer)
        if target >= buffer_size:
            raise BufferBoundsError(
                f"union value target at offset {target} with size 1 "
                f"exceeds a {buffer_size}-byte buffer"
            )
        if dispatch._has_untrusted_types and (
            view_type.__init__ is not TableView.__init__
            or view_type.__new__ is not object.__new__
            or type(view_type).__call__ is not type.__call__
        ):
            return view_type(buffer, target)

        if target > buffer_size - _INT32.size:
            raise BufferBoundsError(
                f"table header at offset {target} with size {_INT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        vtable_distance = _INT32.unpack_from(buffer, target)[0]
        vtable_offset = target - vtable_distance
        metadata = self._vtable_cache.get(vtable_offset)
        if metadata is None:
            if (
                vtable_offset < 0
                or vtable_offset > buffer_size - _VTABLE_HEADER_SIZE
            ):
                raise BufferBoundsError(
                    f"vtable header at offset {vtable_offset} "
                    f"with size {_VTABLE_HEADER_SIZE} exceeds a "
                    f"{buffer_size}-byte buffer"
                )
            vtable_size = _UINT16.unpack_from(buffer, vtable_offset)[0]
            object_size = _UINT16.unpack_from(
                buffer,
                vtable_offset + _UINT16.size,
            )[0]
            if vtable_size < 4 or vtable_size % 2 or object_size < 4:
                raise InvalidBufferError("invalid FlatBuffers table metadata")
            if vtable_offset > buffer_size - vtable_size:
                raise BufferBoundsError(
                    f"vtable at offset {vtable_offset} with size {vtable_size} "
                    f"exceeds a {buffer_size}-byte buffer"
                )
            self._vtable_cache[vtable_offset] = (vtable_size, object_size)
        else:
            vtable_size, object_size = metadata
        if target > buffer_size - object_size:
            raise BufferBoundsError(
                f"table at offset {target} with size {object_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )

        view: Any = view_type.__new__(view_type)
        view._cache_storage = None
        view._buffer = buffer
        view._table_offset = target
        view._vtable_offset = vtable_offset
        view._vtable_size = vtable_size
        view._object_size = object_size
        return view

    @property
    def types(self) -> npt.NDArray[Any]:
        """The parallel read-only discriminator array."""

        return self._types


class TableVector(CachedVector[_ViewT], Generic[_ViewT]):
    """A vector of cached, read-only table views."""

    __slots__ = (
        "_buffer",
        "_last_object_size",
        "_last_vtable_offset",
        "_last_vtable_size",
        "_start",
        "_trusted_construction",
        "_view_type",
    )

    def __init__(
        self,
        buffer: memoryview,
        start: int,
        length: int,
        view_type: type[_ViewT],
    ) -> None:
        super().__init__(length)
        if not issubclass(view_type, TableView):
            raise TypeError("FlatBuffers table vectors require a TableView type")
        view = _readonly_bytes(buffer)
        _require_span(
            view,
            start,
            length * _UINT32.size,
            description="table vector data",
        )
        self._buffer = view
        self._start = start
        self._view_type = view_type
        self._trusted_construction = (
            view_type.__init__ is TableView.__init__
            and view_type.__new__ is object.__new__
            and type(view_type).__call__ is type.__call__
            and getattr(view_type._from_validated_vector, "__func__", None)
            is TableView._from_validated_vector.__func__
        )
        self._last_vtable_offset: int | None = None
        self._last_vtable_size = 0
        self._last_object_size = 0

    def _load(self, index: int) -> _ViewT:
        offset = self._start + index * _UINT32.size
        buffer = self._buffer
        relative = _UINT32.unpack_from(buffer, offset)[0]
        if relative == 0:
            raise InvalidBufferError("table vector element contains a null offset")
        target = offset + relative
        buffer_size = len(buffer)
        if target >= buffer_size:
            raise BufferBoundsError(
                f"table vector element target at offset {target} with size 1 "
                f"exceeds a {buffer_size}-byte buffer"
            )
        if not self._trusted_construction:
            return self._view_type(buffer, target)

        if target > buffer_size - _INT32.size:
            raise BufferBoundsError(
                f"table header at offset {target} with size {_INT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        vtable_distance = _INT32.unpack_from(buffer, target)[0]
        vtable_offset = target - vtable_distance
        if vtable_offset == self._last_vtable_offset:
            vtable_size = self._last_vtable_size
            object_size = self._last_object_size
        else:
            if (
                vtable_offset < 0
                or vtable_offset > buffer_size - _VTABLE_HEADER_SIZE
            ):
                raise BufferBoundsError(
                    f"vtable header at offset {vtable_offset} "
                    f"with size {_VTABLE_HEADER_SIZE} exceeds a "
                    f"{buffer_size}-byte buffer"
                )
            vtable_size = _UINT16.unpack_from(buffer, vtable_offset)[0]
            object_size = _UINT16.unpack_from(
                buffer,
                vtable_offset + _UINT16.size,
            )[0]
            if vtable_size < 4 or vtable_size % 2 or object_size < 4:
                raise InvalidBufferError("invalid FlatBuffers table metadata")
            if vtable_offset > buffer_size - vtable_size:
                raise BufferBoundsError(
                    f"vtable at offset {vtable_offset} with size {vtable_size} "
                    f"exceeds a {buffer_size}-byte buffer"
                )
            self._last_vtable_offset = vtable_offset
            self._last_vtable_size = vtable_size
            self._last_object_size = object_size

        if target > buffer_size - object_size:
            raise BufferBoundsError(
                f"table at offset {target} with size {object_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        view_type = self._view_type
        view: Any = view_type.__new__(view_type)
        view._cache_storage = None
        view._buffer = buffer
        view._table_offset = target
        view._vtable_offset = vtable_offset
        view._vtable_size = vtable_size
        view._object_size = object_size
        return view

    def _iter_dense(self, cache: list[Any], start: int) -> Iterator[_ViewT]:
        if not self._trusted_construction:
            yield from super()._iter_dense(cache, start)
            return

        buffer = self._buffer
        buffer_size = len(buffer)
        vector_start = self._start
        view_type = self._view_type
        uint16_unpack = _UINT16.unpack_from
        uint32_unpack = _UINT32.unpack_from
        int32_unpack = _INT32.unpack_from
        uint16_size = _UINT16.size
        uint32_size = _UINT32.size
        int32_size = _INT32.size
        last_vtable_offset = self._last_vtable_offset
        last_vtable_size = self._last_vtable_size
        last_object_size = self._last_object_size
        cached_count = self._cached_count
        for index in range(start, self._length):
            value = cache[index]
            if value is _MISSING:
                offset = vector_start + index * uint32_size
                relative = uint32_unpack(buffer, offset)[0]
                if relative == 0:
                    raise InvalidBufferError(
                        "table vector element contains a null offset"
                    )
                target = offset + relative
                if target >= buffer_size:
                    raise BufferBoundsError(
                        f"table vector element target at offset {target} "
                        f"with size 1 exceeds a {buffer_size}-byte buffer"
                    )
                if target > buffer_size - int32_size:
                    raise BufferBoundsError(
                        f"table header at offset {target} with size "
                        f"{int32_size} exceeds a {buffer_size}-byte buffer"
                    )
                vtable_distance = int32_unpack(buffer, target)[0]
                vtable_offset = target - vtable_distance
                if vtable_offset == last_vtable_offset:
                    vtable_size = last_vtable_size
                    object_size = last_object_size
                else:
                    if (
                        vtable_offset < 0
                        or vtable_offset > buffer_size - _VTABLE_HEADER_SIZE
                    ):
                        raise BufferBoundsError(
                            f"vtable header at offset {vtable_offset} with size "
                            f"{_VTABLE_HEADER_SIZE} exceeds a "
                            f"{buffer_size}-byte buffer"
                        )
                    vtable_size = uint16_unpack(buffer, vtable_offset)[0]
                    object_size = uint16_unpack(
                        buffer,
                        vtable_offset + uint16_size,
                    )[0]
                    if vtable_size < 4 or vtable_size % 2 or object_size < 4:
                        raise InvalidBufferError(
                            "invalid FlatBuffers table metadata"
                        )
                    if vtable_offset > buffer_size - vtable_size:
                        raise BufferBoundsError(
                            f"vtable at offset {vtable_offset} with size "
                            f"{vtable_size} exceeds a {buffer_size}-byte buffer"
                        )
                    last_vtable_offset = vtable_offset
                    last_vtable_size = vtable_size
                    last_object_size = object_size
                    self._last_vtable_offset = vtable_offset
                    self._last_vtable_size = vtable_size
                    self._last_object_size = object_size
                if target > buffer_size - object_size:
                    raise BufferBoundsError(
                        f"table at offset {target} with size {object_size} "
                        f"exceeds a {buffer_size}-byte buffer"
                    )
                view: Any = view_type.__new__(view_type)
                view._cache_storage = None
                view._buffer = buffer
                view._table_offset = target
                view._vtable_offset = vtable_offset
                view._vtable_size = vtable_size
                view._object_size = object_size
                value = view
                cache[index] = value
                cached_count += 1
                self._cached_count = cached_count
            yield value


class StructVector(CachedVector[_ViewT], Generic[_ViewT]):
    """A vector of cached, read-only inline struct views."""

    __slots__ = (
        "_buffer",
        "_start",
        "_stride",
        "_trusted_construction",
        "_view_type",
    )

    def __init__(
        self,
        buffer: memoryview,
        start: int,
        length: int,
        stride: int,
        view_type: type[_ViewT],
    ) -> None:
        super().__init__(length)
        if not issubclass(view_type, StructView):
            raise TypeError("FlatBuffers struct vectors require a StructView type")
        if stride <= 0:
            raise ValueError("FlatBuffers struct vector stride must be positive")
        struct_size = view_type.__flatbuffer_size__
        if struct_size <= 0 or stride < struct_size:
            raise ValueError(
                "FlatBuffers struct vector stride is smaller than its struct"
            )
        view = _readonly_bytes(buffer)
        _require_span(
            view,
            start,
            length * stride,
            description="struct vector data",
        )
        self._buffer = view
        self._start = start
        self._stride = stride
        self._view_type = view_type
        self._trusted_construction = (
            view_type.__init__ is StructView.__init__
            and view_type.__new__ is object.__new__
            and type(view_type).__call__ is type.__call__
            and getattr(view_type._from_validated_vector, "__func__", None)
            is StructView._from_validated_vector.__func__
        )

    def _load(self, index: int) -> _ViewT:
        offset = self._start + index * self._stride
        view_type = self._view_type
        if not self._trusted_construction:
            return view_type(self._buffer, offset)
        view: Any = view_type.__new__(view_type)
        view._cache_storage = None
        view._buffer = self._buffer
        view._struct_offset = offset
        return view

    def _iter_dense(self, cache: list[Any], start: int) -> Iterator[_ViewT]:
        if not self._trusted_construction:
            yield from super()._iter_dense(cache, start)
            return

        view_type = self._view_type
        buffer = self._buffer
        vector_start = self._start
        stride = self._stride
        cached_count = self._cached_count
        for index in range(start, self._length):
            value = cache[index]
            if value is _MISSING:
                view: Any = view_type.__new__(view_type)
                view._cache_storage = None
                view._buffer = buffer
                view._struct_offset = vector_start + index * stride
                value = view
                cache[index] = value
                cached_count += 1
                self._cached_count = cached_count
            yield value


class StringVector(CachedVector[str]):
    """A vector that decodes and strongly caches UTF-8 strings by index."""

    __slots__ = ("_buffer", "_start")

    def __init__(self, buffer: memoryview, start: int, length: int) -> None:
        super().__init__(length)
        view = _readonly_bytes(buffer)
        _require_span(
            view,
            start,
            length * _UINT32.size,
            description="string vector data",
        )
        self._buffer = view
        self._start = start

    def _load(self, index: int) -> str:
        buffer = self._buffer
        offset = self._start + index * _UINT32.size
        relative = _UINT32.unpack_from(buffer, offset)[0]
        if relative == 0:
            raise InvalidBufferError("string offset contains a null offset")
        length_offset = offset + relative
        buffer_size = len(buffer)
        if length_offset > buffer_size - _UINT32.size:
            raise BufferBoundsError(
                f"string length at offset {length_offset} with size {_UINT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        length = _UINT32.unpack_from(buffer, length_offset)[0]
        start = length_offset + _UINT32.size
        data_size = length + 1
        if start > buffer_size - data_size:
            raise BufferBoundsError(
                f"string data at offset {start} with size {data_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        if buffer[start + length] != 0:
            raise InvalidBufferError("FlatBuffers string is not null-terminated")
        return bytes(buffer[start : start + length]).decode("utf-8")

class _CachedView:
    __slots__ = ("_cache_storage",)

    def __init__(self) -> None:
        self._cache_storage: dict[str, Any] | None = None

    @property
    def _cache(self) -> dict[str, Any]:
        """Lazily allocated compatibility cache for custom view subclasses."""

        cache = self._cache_storage
        if cache is None:
            cache = {}
            self._cache_storage = cache
        return cache

    @_cache.setter
    def _cache(self, cache: dict[str, Any]) -> None:
        self._cache_storage = cache

    def _cached(self, key: str, loader: Any) -> Any:
        cache = self._cache_storage
        if cache is None:
            value = loader()
            self._cache_storage = {key: value}
            return value
        try:
            return cache[key]
        except KeyError:
            value = loader()
            cache[key] = value
            return value


class StructView(_CachedView):
    """Read-only location of an inline FlatBuffers struct."""

    __slots__ = ("_buffer", "_struct_offset")
    __flatbuffer_size__: ClassVar[int] = 0

    def __init__(self, buffer: Buffer, struct_offset: int) -> None:
        self._cache_storage = None
        view = _readonly_bytes(buffer)
        size = self.__flatbuffer_size__
        buffer_size = len(view)
        if size < 0 or struct_offset < 0 or struct_offset > buffer_size - size:
            raise BufferBoundsError(
                f"struct data at offset {struct_offset} with size {size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        self._buffer = view
        self._struct_offset = struct_offset

    @classmethod
    def _from_validated_vector(
        cls,
        buffer: memoryview,
        struct_offset: int,
    ) -> Self:
        """Create a view after its containing vector span has been validated."""

        view = cls.__new__(cls)
        view._cache_storage = None
        view._buffer = buffer
        view._struct_offset = struct_offset
        return view

    def _read_scalar(self, relative_offset: int, scalar_type: str) -> Any:
        scalar = _SCALAR_FORMATS[scalar_type]
        offset = self._struct_offset + relative_offset
        buffer = self._buffer
        buffer_size = len(buffer)
        if offset < 0 or offset > buffer_size - scalar.size:
            raise BufferBoundsError(
                f"struct scalar at offset {offset} with size {scalar.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        return scalar.unpack_from(buffer, offset)[0]

    def _read_struct(
        self,
        relative_offset: int,
        view_type: type[_ViewT],
    ) -> _ViewT:
        return view_type(self._buffer, self._struct_offset + relative_offset)

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    @property
    def struct_offset(self) -> int:
        return self._struct_offset


class TableView(_CachedView):
    """Read-only, cached view of a FlatBuffers table."""

    __slots__ = (
        "_buffer",
        "_object_size",
        "_table_offset",
        "_vtable_offset",
        "_vtable_size",
    )
    __flatbuffer_identifier__: ClassVar[bytes | None] = None

    def __init__(self, buffer: Buffer, table_offset: int) -> None:
        self._cache_storage = None
        view = _readonly_bytes(buffer)
        buffer_size = len(view)
        if table_offset < 0 or table_offset > buffer_size - _INT32.size:
            raise BufferBoundsError(
                f"table header at offset {table_offset} with size {_INT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )

        vtable_distance = _INT32.unpack_from(view, table_offset)[0]
        vtable_offset = table_offset - vtable_distance
        if (
            vtable_offset < 0
            or vtable_offset > buffer_size - _VTABLE_HEADER_SIZE
        ):
            raise BufferBoundsError(
                f"vtable header at offset {vtable_offset} "
                f"with size {_VTABLE_HEADER_SIZE} exceeds a "
                f"{buffer_size}-byte buffer"
            )
        vtable_size = _UINT16.unpack_from(view, vtable_offset)[0]
        object_size = _UINT16.unpack_from(view, vtable_offset + _UINT16.size)[0]
        if vtable_size < 4 or vtable_size % 2 or object_size < 4:
            raise InvalidBufferError("invalid FlatBuffers table metadata")
        if vtable_offset > buffer_size - vtable_size:
            raise BufferBoundsError(
                f"vtable at offset {vtable_offset} with size {vtable_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        if table_offset > buffer_size - object_size:
            raise BufferBoundsError(
                f"table at offset {table_offset} with size {object_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )

        self._buffer = view
        self._table_offset = table_offset
        self._vtable_offset = vtable_offset
        self._vtable_size = vtable_size
        self._object_size = object_size

    @classmethod
    def _from_validated_vector(
        cls,
        buffer: memoryview,
        table_offset: int,
        vtable_offset: int,
        vtable_size: int,
        object_size: int,
    ) -> Self:
        """Create a child after its table and vtable spans have been validated."""

        view = cls.__new__(cls)
        view._cache_storage = None
        view._buffer = buffer
        view._table_offset = table_offset
        view._vtable_offset = vtable_offset
        view._vtable_size = vtable_size
        view._object_size = object_size
        return view

    @classmethod
    def from_buffer(
        cls,
        buffer: Buffer,
        *,
        offset: int = 0,
        size_prefixed: bool = False,
        check_identifier: bool = True,
    ) -> Self:
        """Create a root view over a standard or size-prefixed FlatBuffer.

        A size-prefixed view retains only its declared payload.
        """

        view = _readonly_bytes(buffer)
        root_offset = offset
        if size_prefixed:
            _require_span(view, offset, _UINT32.size, description="size prefix")
            size = _UINT32.unpack_from(view, offset)[0]
            payload_start = offset + _UINT32.size
            _require_span(
                view,
                payload_start,
                size,
                description="size-prefixed buffer",
            )
            payload_end = payload_start + size
            view = view[payload_start:payload_end]
            root_offset = 0

        _require_span(view, root_offset, _UINT32.size, description="root offset")
        if check_identifier and cls.__flatbuffer_identifier__ is not None:
            identifier_offset = root_offset + _UINT32.size
            _require_span(view, identifier_offset, 4, description="file identifier")
            identifier = bytes(view[identifier_offset : identifier_offset + 4])
            if identifier != cls.__flatbuffer_identifier__:
                raise InvalidBufferError(
                    f"expected file identifier {cls.__flatbuffer_identifier__!r}, "
                    f"got {identifier!r}"
                )

        relative_offset = _UINT32.unpack_from(view, root_offset)[0]
        if relative_offset == 0:
            raise InvalidBufferError("root table offset is null")
        return cls(view, root_offset + relative_offset)

    @classmethod
    def from_root(
        cls,
        buffer: Buffer,
        *,
        offset: int = 0,
        size_prefixed: bool = False,
        check_identifier: bool = True,
    ) -> Self:
        """Compatibility alias for :meth:`from_buffer`."""

        return cls.from_buffer(
            buffer,
            offset=offset,
            size_prefixed=size_prefixed,
            check_identifier=check_identifier,
        )

    def _field_position(self, vtable_field: int, size: int) -> int | None:
        if vtable_field < 4 or vtable_field % 2:
            raise ValueError("vtable field offsets must be even and at least 4")
        if size < 0:
            raise BufferBoundsError(f"table field size cannot be negative: {size}")
        if vtable_field >= self._vtable_size:
            return None
        relative = _UINT16.unpack_from(
            self._buffer,
            self._vtable_offset + vtable_field,
        )[0]
        if relative == 0:
            return None
        if relative < 4 or relative > self._object_size - size:
            raise InvalidBufferError("field lies outside its FlatBuffers table")
        return self._table_offset + relative

    def _read_scalar(
        self,
        vtable_field: int,
        scalar_type: str,
        default: Any,
    ) -> Any:
        scalar = _SCALAR_FORMATS[scalar_type]
        if vtable_field < 4 or vtable_field % 2:
            raise ValueError("vtable field offsets must be even and at least 4")
        if vtable_field >= self._vtable_size:
            return default
        relative = _UINT16.unpack_from(
            self._buffer,
            self._vtable_offset + vtable_field,
        )[0]
        if relative == 0:
            return default
        if relative < 4 or relative > self._object_size - scalar.size:
            raise InvalidBufferError("field lies outside its FlatBuffers table")
        return scalar.unpack_from(self._buffer, self._table_offset + relative)[0]

    def _read_table(
        self,
        vtable_field: int,
        view_type: type[_ViewT],
    ) -> _ViewT | None:
        position = self._field_position(vtable_field, _UINT32.size)
        if position is None:
            return None
        buffer = self._buffer
        relative = _UINT32.unpack_from(buffer, position)[0]
        if relative == 0:
            raise InvalidBufferError("table field offset contains a null offset")
        target = position + relative
        buffer_size = len(buffer)
        if target >= buffer_size:
            raise BufferBoundsError(
                f"table field offset target at offset {target} with size 1 "
                f"exceeds a {buffer_size}-byte buffer"
            )
        return view_type(buffer, target)

    def _read_union(
        self,
        vtable_field: int,
        discriminator: int,
        dispatch: UnionDispatch,
    ) -> TableView | None:
        """Read one checked table-valued union payload."""

        if not isinstance(discriminator, int):
            raise TypeError("FlatBuffers union discriminators must be integers")
        if not isinstance(dispatch, UnionDispatch):
            raise TypeError("FlatBuffers unions require a UnionDispatch")
        if discriminator == dispatch.none_tag:
            if self._field_position(vtable_field, _UINT32.size) is not None:
                raise InvalidBufferError(
                    f"{dispatch.name} NONE discriminator has a payload"
                )
            return None

        view_type = dispatch._table_types.get(discriminator)
        if view_type is None:
            raise InvalidBufferError(
                f"unknown {dispatch.name} discriminator {discriminator}"
            )
        position = self._field_position(vtable_field, _UINT32.size)
        if position is None:
            raise InvalidBufferError(
                f"{dispatch.name} discriminator {discriminator} has no payload"
            )
        buffer = self._buffer
        relative = _UINT32.unpack_from(buffer, position)[0]
        if relative == 0:
            raise InvalidBufferError(
                f"{dispatch.name} discriminator {discriminator} "
                "has a null payload"
            )
        target = position + relative
        buffer_size = len(buffer)
        if target >= buffer_size:
            raise BufferBoundsError(
                f"union field target at offset {target} with size 1 "
                f"exceeds a {buffer_size}-byte buffer"
            )
        return view_type(buffer, target)

    def _read_struct(
        self,
        vtable_field: int,
        view_type: type[_ViewT],
    ) -> _ViewT | None:
        position = self._field_position(
            vtable_field,
            view_type.__flatbuffer_size__,
        )
        if position is None:
            return None
        return view_type(self._buffer, position)

    def _read_string(self, vtable_field: int) -> str | None:
        position = self._field_position(vtable_field, _UINT32.size)
        if position is None:
            return None
        return _string_from_offset(self._buffer, position)

    def _vector_info(
        self,
        vtable_field: int,
        element_size: int,
    ) -> tuple[int, int] | None:
        if element_size <= 0:
            raise BufferBoundsError(
                f"vector element size must be positive: {element_size}"
            )
        position = self._field_position(vtable_field, _UINT32.size)
        if position is None:
            return None
        buffer = self._buffer
        relative = _UINT32.unpack_from(buffer, position)[0]
        if relative == 0:
            raise InvalidBufferError("vector field offset contains a null offset")
        length_offset = position + relative
        buffer_size = len(buffer)
        if length_offset > buffer_size - _UINT32.size:
            raise BufferBoundsError(
                f"vector length at offset {length_offset} with size {_UINT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        length = _UINT32.unpack_from(buffer, length_offset)[0]
        start = length_offset + _UINT32.size
        data_size = length * element_size
        if start > buffer_size - data_size:
            raise BufferBoundsError(
                f"vector data at offset {start} with size {data_size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        return start, length

    def _read_byte_vector(self, vtable_field: int) -> memoryview | None:
        info = self._vector_info(vtable_field, 1)
        if info is None:
            return None
        start, length = info
        return self._buffer[start : start + length]

    def _read_numpy_vector(
        self,
        vtable_field: int,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[Any] | None:
        resolved_dtype = (
            _VECTOR_DTYPES.get(dtype) if isinstance(dtype, str) else None
        )
        if resolved_dtype is None:
            resolved_dtype = np.dtype(dtype)
        info = self._vector_info(vtable_field, resolved_dtype.itemsize)
        if info is None:
            return None
        start, length = info
        return np.frombuffer(
            self._buffer,
            dtype=resolved_dtype,
            count=length,
            offset=start,
        )

    def _read_table_vector(
        self,
        vtable_field: int,
        view_type: type[_ViewT],
    ) -> TableVector[_ViewT] | None:
        info = self._vector_info(vtable_field, _UINT32.size)
        if info is None:
            return None
        start, length = info
        return TableVector(self._buffer, start, length, view_type)

    def _read_struct_vector(
        self,
        vtable_field: int,
        view_type: type[_ViewT],
    ) -> StructVector[_ViewT] | None:
        size = view_type.__flatbuffer_size__
        info = self._vector_info(vtable_field, size)
        if info is None:
            return None
        start, length = info
        return StructVector(self._buffer, start, length, size, view_type)

    def _read_string_vector(self, vtable_field: int) -> StringVector | None:
        info = self._vector_info(vtable_field, _UINT32.size)
        if info is None:
            return None
        start, length = info
        return StringVector(self._buffer, start, length)

    @property
    def buffer(self) -> memoryview:
        """The read-only bytes used by this view."""

        return self._buffer

    @property
    def table_offset(self) -> int:
        return self._table_offset


__all__ = [
    "BufferBoundsError",
    "CachedVector",
    "InvalidBufferError",
    "OpenIntEnum",
    "StringVector",
    "StructVector",
    "StructView",
    "TableVector",
    "TableView",
    "UnionDispatch",
    "UnionVector",
    "build_offset_vector",
    "build_scalar_vector",
    "build_string_vector",
]
