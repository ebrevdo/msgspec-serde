"""Runtime primitives shared by generated read-only FlatBuffers views."""

from __future__ import annotations

import struct
from collections.abc import Buffer, Iterator, Mapping, Sequence
from enum import IntEnum
from typing import Any, ClassVar, Generic, Self, TypeVar, overload

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
_DENSE_CACHE_LIMIT = 8
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
_KeyT = TypeVar("_KeyT")
_ViewT = TypeVar("_ViewT", bound="TableView | StructView")
_UnionT = TypeVar("_UnionT")


def _has_default_view_construction(
    view_type: type[Any],
    base_type: type[Any],
) -> bool:
    return (
        view_type.__init__ is base_type.__init__
        and view_type.__new__ is object.__new__
        and type(view_type).__call__ is type.__call__
    )


def _supports_fast_vector_construction(
    view_type: type[Any],
    base_type: type[Any],
) -> bool:
    return (
        _has_default_view_construction(view_type, base_type)
        and getattr(view_type._from_validated_vector, "__func__", None)
        is base_type._from_validated_vector.__func__
    )


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
            not _has_default_view_construction(view_type, TableView)
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


def _cache_materialized_prefix(
    cache: dict[int, _T],
    values: list[_T],
) -> None:
    for index, value in enumerate(values):
        cache.setdefault(index, value)


class CachedVector(Sequence[_T], Generic[_T]):
    """A fixed-length vector that strongly caches every accessed element."""

    __slots__ = ("_cache", "_length")

    def __init__(self, length: int) -> None:
        if length < 0:
            raise ValueError("FlatBuffers vector length cannot be negative")
        self._length = length
        if length == 0:
            cache: dict[int, _T] | list[Any] | tuple[_T, ...] = ()
        elif length <= _DENSE_CACHE_LIMIT:
            cache = [_MISSING] * length
        else:
            cache = {}
        self._cache = cache

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
        if type(cache) is tuple:
            if isinstance(index, slice):
                return cache[index]
            return cache[index]

        if isinstance(index, slice):
            if index == slice(None):
                return tuple(self)
            return tuple(self[item] for item in range(*index.indices(self._length)))

        length = self._length
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("FlatBuffers vector index out of range")

        if type(cache) is list:
            value = cache[index]
            if value is not _MISSING:
                return value
        else:
            sparse_cache: dict[int, _T] = cache  # ty: ignore[invalid-assignment]
            value = sparse_cache.get(index, _MISSING)
            if value is not _MISSING:
                return value

        value = self._load(index)
        if type(cache) is list:
            cache[index] = value
            return value
        return sparse_cache.setdefault(index, value)

    def __iter__(self) -> Iterator[_T]:
        cache = self._cache
        if type(cache) is tuple:
            return iter(cache)
        mutable_cache: dict[int, _T] | list[Any] = (  # ty: ignore[invalid-assignment]
            cache
        )
        return self._iter_materializing(mutable_cache)

    def _iter_materializing(
        self,
        cache: dict[int, _T] | list[Any],
    ) -> Iterator[_T]:
        load = self._load
        materialized: list[_T] = []
        if type(cache) is list:
            for index in range(self._length):
                value = cache[index]
                if value is _MISSING:
                    value = load(index)
                    cache[index] = value
                materialized.append(value)
                yield value
            self._cache = tuple(materialized)
            return

        sparse_cache: dict[int, _T] = cache  # ty: ignore[invalid-assignment]
        if not sparse_cache:
            published = False
            try:
                for index in range(self._length):
                    value = load(index)
                    materialized.append(value)
                    yield value
                self._cache = tuple(materialized)
                published = True
            finally:
                if not published:
                    _cache_materialized_prefix(sparse_cache, materialized)
            return

        get = sparse_cache.get
        for index in range(self._length):
            value = get(index, _MISSING)
            if value is _MISSING:
                value = load(index)
                sparse_cache[index] = value
            materialized.append(value)
            yield value
        self._cache = tuple(materialized)

    @property
    def cached_count(self) -> int:
        """Number of vector elements materialized so far."""

        cache = self._cache
        if type(cache) is list:
            return len(cache) - cache.count(_MISSING)
        return len(cache)


class UnionVector(CachedVector[_UnionT], Generic[_UnionT]):
    """Parallel discriminator/payload vectors with cached typed views."""

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

        buffer_view = _readonly_bytes(buffer)
        _require_span(
            buffer_view,
            value_start,
            value_length * _UINT32.size,
            description="union value vector data",
        )
        tag_bytes = memoryview(types).cast("B").toreadonly()

        self._buffer = buffer_view
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
            if relative != 0:
                raise InvalidBufferError(
                    f"{dispatch.name} NONE at index {index} has a payload"
                )
            return None  # ty: ignore[invalid-return-type]

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
        if (
            dispatch._has_untrusted_types
            and not _has_default_view_construction(view_type, TableView)
        ):
            view: Any = view_type(buffer, target)
            return view

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
            vtable_size, object_size = self._vtable_cache.setdefault(
                vtable_offset,
                (vtable_size, object_size),
            )
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
        "_last_vtable",
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
        buffer_view = _readonly_bytes(buffer)
        _require_span(
            buffer_view,
            start,
            length * _UINT32.size,
            description="table vector data",
        )
        self._buffer = buffer_view
        self._start = start
        self._view_type = view_type
        self._trusted_construction = _supports_fast_vector_construction(
            view_type,
            TableView,
        )
        self._last_vtable: tuple[int, int, int] | None = None

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
        last_vtable = self._last_vtable
        if last_vtable is not None and vtable_offset == last_vtable[0]:
            _, vtable_size, object_size = last_vtable
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
            self._last_vtable = (vtable_offset, vtable_size, object_size)

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

    def _iter_materializing(
        self,
        cache: dict[int, _ViewT] | list[Any],
    ) -> Iterator[_ViewT]:
        if not self._trusted_construction:
            yield from super()._iter_materializing(cache)
            return

        sparse = type(cache) is dict
        cold_sparse = sparse and not cache
        published = False
        materialized: list[_ViewT] = []
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
        last_vtable = self._last_vtable
        try:
            for index in range(self._length):
                if cold_sparse:
                    value = _MISSING
                elif sparse:
                    value = cache.get(index, _MISSING)
                else:
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
                    if last_vtable is not None and vtable_offset == last_vtable[0]:
                        _, vtable_size, object_size = last_vtable
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
                        last_vtable = (vtable_offset, vtable_size, object_size)
                        self._last_vtable = last_vtable
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
                    if not cold_sparse:
                        cache[index] = value
                materialized.append(value)
                yield value
            self._cache = tuple(materialized)
            published = True
        finally:
            if cold_sparse and not published:
                _cache_materialized_prefix(cache, materialized)


class TableMap(Mapping[_KeyT, _ViewT], Generic[_KeyT, _ViewT]):
    """A read-only keyed table vector with lazy binary-search lookup."""

    __slots__ = ("_key_name", "_round_float32_keys", "_values")

    def __init__(
        self,
        buffer: memoryview,
        start: int,
        length: int,
        view_type: type[_ViewT],
        key_name: str,
        key_type: str,
    ) -> None:
        self._values = TableVector(buffer, start, length, view_type)
        self._key_name = key_name
        self._round_float32_keys = key_type == "float32"

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[_KeyT]:
        for value in self._values:
            yield getattr(value, self._key_name)

    def __getitem__(self, key: _KeyT) -> _ViewT:
        lookup_key: Any = key
        if self._round_float32_keys:
            lookup_key = _FLOAT32.unpack(_FLOAT32.pack(key))[0]
        values = self._values
        key_name = self._key_name
        start = 0
        stop = len(values)
        while start < stop:
            middle = (start + stop) // 2
            value = values[middle]
            candidate: Any = getattr(value, key_name)
            if candidate < lookup_key:
                start = middle + 1
            elif lookup_key < candidate:
                stop = middle
            else:
                return value
        raise KeyError(key)


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
        buffer_view = _readonly_bytes(buffer)
        _require_span(
            buffer_view,
            start,
            length * stride,
            description="struct vector data",
        )
        self._buffer = buffer_view
        self._start = start
        self._stride = stride
        self._view_type = view_type
        self._trusted_construction = _supports_fast_vector_construction(
            view_type,
            StructView,
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

    def _iter_materializing(
        self,
        cache: dict[int, _ViewT] | list[Any],
    ) -> Iterator[_ViewT]:
        if not self._trusted_construction:
            yield from super()._iter_materializing(cache)
            return

        sparse = type(cache) is dict
        cold_sparse = sparse and not cache
        published = False
        materialized: list[_ViewT] = []
        view_type = self._view_type
        buffer = self._buffer
        vector_start = self._start
        stride = self._stride
        try:
            for index in range(self._length):
                if cold_sparse:
                    value = _MISSING
                elif sparse:
                    value = cache.get(index, _MISSING)
                else:
                    value = cache[index]
                if value is _MISSING:
                    view: Any = view_type.__new__(view_type)
                    view._cache_storage = None
                    view._buffer = buffer
                    view._struct_offset = vector_start + index * stride
                    value = view
                    if not cold_sparse:
                        cache[index] = value
                materialized.append(value)
                yield value
            self._cache = tuple(materialized)
            published = True
        finally:
            if cold_sparse and not published:
                _cache_materialized_prefix(cache, materialized)


class StringVector(CachedVector[str]):
    """A vector that decodes and strongly caches UTF-8 strings by index."""

    __slots__ = ("_buffer", "_start")

    def __init__(self, buffer: memoryview, start: int, length: int) -> None:
        super().__init__(length)
        buffer_view = _readonly_bytes(buffer)
        _require_span(
            buffer_view,
            start,
            length * _UINT32.size,
            description="string vector data",
        )
        self._buffer = buffer_view
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
        """Return the lazily allocated cache used by typed byte-vector helpers."""

        cache = self._cache_storage
        if cache is None:
            cache = {}
            self._cache_storage = cache
        return cache


class StructView(_CachedView):
    """Read-only location of an inline FlatBuffers struct."""

    __slots__ = ("_buffer", "_struct_offset")
    __flatbuffer_size__: ClassVar[int] = 0

    def __init__(self, buffer: Buffer, struct_offset: int) -> None:
        self._cache_storage = None
        buffer_view = _readonly_bytes(buffer)
        size = self.__flatbuffer_size__
        buffer_size = len(buffer_view)
        if size < 0 or struct_offset < 0 or struct_offset > buffer_size - size:
            raise BufferBoundsError(
                f"struct data at offset {struct_offset} with size {size} "
                f"exceeds a {buffer_size}-byte buffer"
            )
        self._buffer = buffer_view
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

    def _read_numpy_array(
        self,
        relative_offset: int,
        length: int,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[Any]:
        resolved_dtype = (
            _VECTOR_DTYPES.get(dtype) if isinstance(dtype, str) else None
        )
        if resolved_dtype is None:
            resolved_dtype = np.dtype(dtype)
        start = self._struct_offset + relative_offset
        _require_span(
            self._buffer,
            start,
            length * resolved_dtype.itemsize,
            description="struct array data",
        )
        return np.frombuffer(
            self._buffer,
            dtype=resolved_dtype,
            count=length,
            offset=start,
        )

    def _read_struct_array(
        self,
        relative_offset: int,
        length: int,
        view_type: type[_ViewT],
    ) -> StructVector[_ViewT]:
        return StructVector(
            self._buffer,
            self._struct_offset + relative_offset,
            length,
            view_type.__flatbuffer_size__,
            view_type,
        )

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
        buffer_view = _readonly_bytes(buffer)
        buffer_size = len(buffer_view)
        if table_offset < 0 or table_offset > buffer_size - _INT32.size:
            raise BufferBoundsError(
                f"table header at offset {table_offset} with size {_INT32.size} "
                f"exceeds a {buffer_size}-byte buffer"
            )

        vtable_distance = _INT32.unpack_from(buffer_view, table_offset)[0]
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
        vtable_size = _UINT16.unpack_from(buffer_view, vtable_offset)[0]
        object_size = _UINT16.unpack_from(
            buffer_view, vtable_offset + _UINT16.size
        )[0]
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

        self._buffer = buffer_view
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
    def _from_buffer(
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

        buffer_view = _readonly_bytes(buffer)
        root_offset = offset
        if size_prefixed:
            _require_span(
                buffer_view, offset, _UINT32.size, description="size prefix"
            )
            size = _UINT32.unpack_from(buffer_view, offset)[0]
            payload_start = offset + _UINT32.size
            _require_span(
                buffer_view,
                payload_start,
                size,
                description="size-prefixed buffer",
            )
            payload_end = payload_start + size
            buffer_view = buffer_view[payload_start:payload_end]
            root_offset = 0

        _require_span(
            buffer_view, root_offset, _UINT32.size, description="root offset"
        )
        expected_identifier = cls.__flatbuffer_identifier__
        if check_identifier and expected_identifier is not None:
            identifier_offset = root_offset + _UINT32.size
            _require_span(
                buffer_view,
                identifier_offset,
                4,
                description="file identifier",
            )
            actual_identifier = bytes(
                buffer_view[identifier_offset : identifier_offset + 4]
            )
            if actual_identifier != expected_identifier:
                raise InvalidBufferError(
                    f"expected file identifier {expected_identifier!r}, "
                    f"got {actual_identifier!r}"
                )

        relative_offset = _UINT32.unpack_from(buffer_view, root_offset)[0]
        if relative_offset == 0:
            raise InvalidBufferError("root table offset is null")
        return cls(buffer_view, root_offset + relative_offset)

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

    def _read_table_map(
        self,
        vtable_field: int,
        view_type: type[_ViewT],
        key_name: str,
        key_type: str,
    ) -> TableMap[Any, _ViewT] | None:
        info = self._vector_info(vtable_field, _UINT32.size)
        if info is None:
            return None
        start, length = info
        return TableMap(
            self._buffer,
            start,
            length,
            view_type,
            key_name,
            key_type,
        )

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
    "TableMap",
    "TableVector",
    "TableView",
    "UnionDispatch",
    "UnionVector",
]
