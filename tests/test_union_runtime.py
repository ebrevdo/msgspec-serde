from __future__ import annotations

import struct
from collections.abc import Buffer, Sequence

import flatbuffers
import numpy as np
import pytest

from msgspec_flatbuffers import (
    BufferBoundsError,
    InvalidBufferError,
    TableView,
    UnionDispatch,
    UnionVector,
)

_INT32 = struct.Struct("<i")
_UINT32 = struct.Struct("<I")


class ChildView(TableView):
    __slots__ = ()


class HolderView(TableView):
    __slots__ = ()


class CountingChildView(ChildView):
    __slots__ = ()
    init_calls = 0

    def __init__(self, buffer: Buffer, table_offset: int) -> None:
        type(self).init_calls += 1
        super().__init__(buffer, table_offset)


_DISPATCH = UnionDispatch(
    "Example.Any",
    0,
    {555: ChildView, 777: ChildView},
)


def _build_child(builder: flatbuffers.Builder) -> int:
    builder.StartObject(0)
    return builder.EndObject()


def _build_scalar_union(*, tag: int, include_payload: bool) -> bytes:
    builder = flatbuffers.Builder(0)
    payload = _build_child(builder) if include_payload else None
    builder.StartObject(2)
    if payload is not None:
        builder.PrependUOffsetTRelativeSlot(1, payload, 0)
    builder.PrependInt32Slot(0, tag, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def _build_union_vector(
    tags: Sequence[int],
    present: Sequence[bool],
) -> bytes:
    if len(tags) != len(present):
        raise ValueError("tag and presence lengths differ")
    builder = flatbuffers.Builder(0)
    offsets = [
        _build_child(builder) if include_payload else None
        for include_payload in present
    ]
    builder.StartVector(_UINT32.size, len(offsets), _UINT32.size)
    for offset in reversed(offsets):
        if offset is None:
            builder.PrependUint32(0)
        else:
            builder.PrependUOffsetTRelative(offset)
    values = builder.EndVector()
    builder.StartVector(_INT32.size, len(tags), _INT32.size)
    for tag in reversed(tags):
        builder.PrependInt32(tag)
    types = builder.EndVector()
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(1, values, 0)
    builder.PrependUOffsetTRelativeSlot(0, types, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def _union_vector(buffer: bytes | bytearray) -> UnionVector[ChildView]:
    holder = HolderView.from_buffer(buffer)
    types = holder._read_numpy_vector(4, "<i4")
    values = holder._vector_info(6, _UINT32.size)
    assert types is not None and values is not None
    start, length = values
    return UnionVector(
        holder.buffer,
        types,
        start,
        length,
        _INT32,
        _DISPATCH,
    )


def test_scalar_union_reads_known_table_and_none() -> None:
    populated = HolderView.from_buffer(
        _build_scalar_union(tag=555, include_payload=True)
    )
    empty = HolderView.from_buffer(
        _build_scalar_union(tag=0, include_payload=False)
    )

    value = populated._read_union(6, 555, _DISPATCH)

    assert isinstance(value, ChildView)
    assert value.buffer.readonly
    assert empty._read_union(6, 0, _DISPATCH) is None


@pytest.mark.parametrize(
    ("tag", "include_payload", "message"),
    [
        (0, True, "NONE discriminator has a payload"),
        (555, False, "has no payload"),
        (999, False, "unknown Example.Any discriminator"),
    ],
)
def test_scalar_union_rejects_inconsistent_or_unknown_values(
    tag: int,
    include_payload: bool,
    message: str,
) -> None:
    view = HolderView.from_buffer(
        _build_scalar_union(tag=tag, include_payload=include_payload)
    )

    with pytest.raises(InvalidBufferError, match=message):
        view._read_union(6, tag, _DISPATCH)


def test_scalar_union_rejects_an_out_of_bounds_target() -> None:
    buffer = bytearray(_build_scalar_union(tag=555, include_payload=True))
    view = HolderView.from_buffer(buffer)
    position = view._field_position(6, _UINT32.size)
    assert position is not None
    _UINT32.pack_into(buffer, position, 2**32 - 1)

    with pytest.raises(BufferBoundsError, match="union field target"):
        HolderView.from_buffer(buffer)._read_union(6, 555, _DISPATCH)


def test_union_vector_caches_checked_table_views_with_wide_tags() -> None:
    vector = _union_vector(_build_union_vector((555, 777), (True, True)))

    first = vector[0]
    values = tuple(vector)

    assert first is values[0]
    assert vector[-1] is values[-1]
    assert vector[:] == values
    assert vector.cached_count == 2
    assert values[0].table_offset != values[1].table_offset
    assert vector.types.dtype == np.dtype("<i4")
    assert not vector.types.flags.writeable
    assert values[0].buffer.readonly


def test_union_vector_preserves_custom_view_construction() -> None:
    CountingChildView.init_calls = 0
    base = _union_vector(_build_union_vector((555,), (True,)))
    vector = UnionVector[CountingChildView](
        base._buffer,
        base.types,
        base._value_start,
        1,
        _INT32,
        UnionDispatch("Custom", 0, {555: CountingChildView}),
    )

    value = vector[0]

    assert isinstance(value, CountingChildView)
    assert CountingChildView.init_calls == 1
    assert vector[0] is value
    assert CountingChildView.init_calls == 1


@pytest.mark.parametrize(
    ("tag", "dtype", "unpacker"),
    [
        (1, "<i1", struct.Struct("<b")),
        (1, "<u1", struct.Struct("<B")),
        (257, "<i2", struct.Struct("<h")),
        (257, "<u2", struct.Struct("<H")),
        (65_537, "<i4", struct.Struct("<i")),
        (65_537, "<u4", struct.Struct("<I")),
        (1 << 40, "<i8", struct.Struct("<q")),
        (1 << 40, "<u8", struct.Struct("<Q")),
    ],
)
def test_union_vector_supports_all_integral_discriminator_widths(
    tag: int,
    dtype: str,
    unpacker: struct.Struct,
) -> None:
    base = _union_vector(_build_union_vector((555,), (True,)))
    dispatch = UnionDispatch("Wide", 0, {tag: ChildView})
    type_values = np.array([tag], dtype=dtype)
    type_values.setflags(write=False)
    vector = UnionVector[ChildView](
        base._buffer,
        type_values,
        base._value_start,
        1,
        unpacker,
        dispatch,
    )

    assert isinstance(vector[0], ChildView)
    assert not vector.types.flags.writeable


def test_union_vector_accepts_none_and_rejects_invalid_payloads() -> None:
    unknown = _union_vector(_build_union_vector((999,), (True,)))
    none = _union_vector(_build_union_vector((0,), (False,)))
    missing = _union_vector(_build_union_vector((555,), (False,)))

    with pytest.raises(InvalidBufferError, match="unknown Example.Any"):
        _ = unknown[0]
    assert none[0] is None
    with pytest.raises(InvalidBufferError, match="null offset"):
        _ = missing[0]
    assert none.cached_count == 1
    assert unknown.cached_count == missing.cached_count == 0


def test_union_vector_rejects_mismatched_lengths_and_widths() -> None:
    buffer = _build_union_vector((555,), (True,))
    vector = _union_vector(buffer)
    types = vector.types
    repeated_types = np.repeat(types, 2)
    repeated_types.setflags(write=False)

    with pytest.raises(InvalidBufferError, match="lengths differ"):
        UnionVector(
            buffer,
            repeated_types,
            vector._value_start,
            1,
            _INT32,
            _DISPATCH,
        )
    with pytest.raises(ValueError, match="formats differ"):
        UnionVector(
            buffer,
            types,
            vector._value_start,
            1,
            struct.Struct("<q"),
            _DISPATCH,
        )


def test_union_vector_rejects_an_out_of_bounds_target() -> None:
    buffer = bytearray(_build_union_vector((555,), (True,)))
    vector = _union_vector(buffer)
    _UINT32.pack_into(buffer, vector._value_start, 2**32 - 1)

    with pytest.raises(BufferBoundsError, match="union value target"):
        _ = _union_vector(buffer)[0]


def test_union_vector_rejects_writable_type_arrays() -> None:
    buffer = _build_union_vector((555,), (True,))
    vector = _union_vector(buffer)
    writable_types = np.array([555], dtype="<i4")

    with pytest.raises(TypeError, match="read-only"):
        UnionVector(
            buffer,
            writable_types,
            vector._value_start,
            1,
            _INT32,
            _DISPATCH,
        )


def test_union_vector_rejects_big_endian_discriminator_arrays() -> None:
    buffer = _build_union_vector((555,), (True,))
    vector = _union_vector(buffer)
    big_endian_types = np.array([555], dtype=">i4")
    big_endian_types.setflags(write=False)

    with pytest.raises(TypeError, match="little-endian"):
        UnionVector(
            buffer,
            big_endian_types,
            vector._value_start,
            1,
            _INT32,
            _DISPATCH,
        )


def test_union_dispatch_rejects_invalid_alternatives() -> None:
    with pytest.raises(ValueError, match="NONE"):
        UnionDispatch("Bad", 0, {0: ChildView})
    with pytest.raises(TypeError, match="TableView"):
        UnionDispatch(
            "Bad",
            0,
            {1: object},  # ty: ignore[invalid-argument-type]
        )
