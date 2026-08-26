from __future__ import annotations

import struct
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from threading import Barrier
from typing import Any

import flatbuffers
import numpy as np
import pytest

from msgspec_flatbuffers import (
    BufferBoundsError,
    CachedVector,
    TableView,
    build_scalar_vector,
)


class ExampleView(TableView):
    pass


class StringExampleView(TableView):
    @property
    def name(self) -> str | None:
        return self._read_string(4)


class ExampleEnum(IntEnum):
    ZERO = 0
    ONE = 1


class ObjectVector(CachedVector[object]):
    def __init__(self, length: int) -> None:
        super().__init__(length)
        self.loaded: list[int] = []

    def _load(self, index: int) -> object:
        self.loaded.append(index)
        return object()


class FailingVector(CachedVector[int]):
    def __init__(self) -> None:
        super().__init__(1)
        self.attempts = 0

    def _load(self, index: int) -> int:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("load failed")
        return index


class ConcurrentLoadVector(CachedVector[object]):
    def __init__(self) -> None:
        super().__init__(1)
        self.barrier = Barrier(2)

    def _load(self, index: int) -> object:
        self.barrier.wait()
        return object()


_SCALAR_BUILDERS = {
    "bool": ("PrependBool", 1),
    "int8": ("PrependInt8", 1),
    "uint8": ("PrependUint8", 1),
    "int16": ("PrependInt16", 2),
    "uint16": ("PrependUint16", 2),
    "int32": ("PrependInt32", 4),
    "uint32": ("PrependUint32", 4),
    "int64": ("PrependInt64", 8),
    "uint64": ("PrependUint64", 8),
    "float32": ("PrependFloat32", 4),
    "float64": ("PrependFloat64", 8),
}


def _reference_scalar_vector(
    builder: flatbuffers.Builder,
    values: Sequence[Any],
    scalar_type: str,
) -> int:
    prepender_name, size = _SCALAR_BUILDERS[scalar_type]
    builder.StartVector(size, len(values), size)
    prepend = getattr(builder, prepender_name)
    for value in reversed(values):
        prepend(value)
    return builder.EndVector()


def _finished_vector(
    values: Sequence[Any],
    scalar_type: str,
    *,
    reference: bool,
) -> bytes:
    builder = flatbuffers.Builder(0)
    if reference:
        offset = _reference_scalar_vector(builder, values, scalar_type)
    else:
        offset = build_scalar_vector(builder, values, scalar_type)
    builder.Finish(offset)
    return bytes(builder.Output())


def _minimal_table_buffer() -> bytearray:
    buffer = bytearray(12)
    struct.pack_into("<I", buffer, 0, 8)
    struct.pack_into("<HH", buffer, 4, 4, 4)
    struct.pack_into("<i", buffer, 8, 4)
    return buffer


def _size_prefixed(payload: bytes | bytearray) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def test_from_root_keeps_a_read_only_view_of_the_original_buffer() -> None:
    buffer = _minimal_table_buffer()

    view = ExampleView.from_root(buffer)

    assert view.table_offset == 8
    assert view.buffer.readonly
    assert view.buffer.obj is buffer


def test_size_prefixed_root_is_bounded_to_its_declared_payload() -> None:
    payload = _minimal_table_buffer()
    leading = b"lead"
    trailing = b"trailing frame"
    framed = bytearray(leading + _size_prefixed(payload) + trailing)

    view = ExampleView.from_root(
        framed,
        offset=len(leading),
        size_prefixed=True,
    )

    assert view.table_offset == 8
    assert len(view.buffer) == len(payload)
    assert view.buffer.obj is framed
    assert bytes(view.buffer) == payload


@pytest.mark.parametrize("declared_size", [3, 4])
def test_size_prefixed_root_rejects_a_payload_too_small_for_its_root(
    declared_size: int,
) -> None:
    payload = bytearray(12)
    struct.pack_into("<I", payload, 0, 8)
    framed = bytearray(struct.pack("<I", declared_size) + payload)

    with pytest.raises(BufferBoundsError):
        ExampleView.from_root(framed, size_prefixed=True)


def test_size_prefixed_root_rejects_a_declared_payload_past_the_input() -> None:
    framed = struct.pack("<I", 12) + bytes(11)

    with pytest.raises(BufferBoundsError, match="size-prefixed buffer"):
        ExampleView.from_root(framed, size_prefixed=True)


def test_concatenated_size_prefixed_roots_keep_separate_buffers() -> None:
    payload = _minimal_table_buffer()
    frame = _size_prefixed(payload)
    combined = bytearray(frame + frame)

    first = ExampleView.from_root(combined, size_prefixed=True)
    second = ExampleView.from_root(
        combined,
        offset=len(frame),
        size_prefixed=True,
    )

    assert bytes(first.buffer) == payload
    assert bytes(second.buffer) == payload
    assert first.buffer.obj is combined
    assert second.buffer.obj is combined


def test_size_prefixed_root_cannot_use_a_vtable_before_its_payload() -> None:
    leading_vtable = struct.pack("<HH", 4, 4)
    payload = bytearray(8)
    struct.pack_into("<I", payload, 0, 4)
    struct.pack_into("<i", payload, 4, 12)
    framed = leading_vtable + _size_prefixed(payload)

    with pytest.raises(BufferBoundsError, match="vtable header"):
        ExampleView.from_root(
            framed,
            offset=len(leading_vtable),
            size_prefixed=True,
        )


def test_size_prefixed_field_cannot_use_trailing_string_bytes() -> None:
    payload = bytearray(26)
    struct.pack_into("<I", payload, 0, 12)
    struct.pack_into("<HHH", payload, 6, 6, 8, 4)
    struct.pack_into("<iI", payload, 12, 6, 4)
    struct.pack_into("<I", payload, 20, 1)
    payload[24:26] = b"x\x00"
    declared_size = 20
    framed = struct.pack("<I", declared_size) + payload

    view = StringExampleView.from_root(framed, size_prefixed=True)

    with pytest.raises(BufferBoundsError, match="string length"):
        _ = view.name


@pytest.mark.parametrize(
    ("buffer", "offset"),
    [
        (b"\x00\x00\x00", 0),
        (b"\x04\x00\x00\x00", 0),
        (b"\x00\x00\x00\x00", -1),
    ],
)
def test_from_root_rejects_out_of_bounds_offsets(
    buffer: bytes,
    offset: int,
) -> None:
    with pytest.raises(BufferBoundsError):
        ExampleView.from_root(buffer, offset=offset)


def test_view_rejects_noncontiguous_input() -> None:
    buffer = memoryview(bytearray(16))[::2]

    with pytest.raises(TypeError, match="C-contiguous"):
        ExampleView(buffer, 0)


def test_full_iteration_materializes_without_changing_identity_or_order() -> None:
    vector = ObjectVector(128)
    assert isinstance(vector._cache, dict)
    accessed = list(range(127, 63, -1))
    first_values = {index: vector[index] for index in accessed}

    assert vector.cached_count == 64
    assert vector[-1] is first_values[127]
    assert vector[64:68] == tuple(first_values[index] for index in range(64, 68))

    all_values = tuple(vector)
    assert isinstance(vector._cache, tuple)
    assert vector.cached_count == 128
    assert tuple(vector) == all_values
    assert vector[:] == all_values
    assert all(all_values[index] is value for index, value in first_values.items())
    assert vector.loaded.count(127) == 1


def test_cached_vector_uses_dense_storage_only_through_eight_items() -> None:
    assert isinstance(ObjectVector(8)._cache, list)
    assert isinstance(ObjectVector(9)._cache, dict)


def test_concurrent_dense_cache_misses_leave_one_cached_value() -> None:
    vector = ConcurrentLoadVector()

    with ThreadPoolExecutor(max_workers=2) as executor:
        values = list(executor.map(vector.__getitem__, (0, 0)))

    cached = vector[0]
    assert vector.cached_count == 1
    assert cached in values
    assert vector[0] is cached


def test_cached_vector_does_not_cache_loader_exceptions() -> None:
    vector = FailingVector()

    with pytest.raises(RuntimeError, match="load failed"):
        vector[0]

    assert vector.cached_count == 0
    assert vector[0] == 0
    assert vector.cached_count == 1
    assert vector.attempts == 2


@pytest.mark.parametrize(
    ("scalar_type", "values"),
    [
        ("bool", [False, True] * 16),
        ("int8", [-128, -1, 0, 127] * 8),
        ("int8", [ExampleEnum.ZERO, ExampleEnum.ONE] * 16),
        ("uint8", [0, 1, 127, 255] * 8),
        ("int16", [-32768, -1, 0, 32767] * 8),
        ("uint16", [0, 1, 32768, 65535] * 8),
        ("int32", [-(2**31), -1, 0, 2**31 - 1] * 8),
        ("uint32", [0, 1, 2**31, 2**32 - 1] * 8),
        ("int64", [-(2**63), -1, 0, 2**63 - 1] * 8),
        ("uint64", [0, 1, 2**63, 2**64 - 1] * 8),
        ("float32", [-1.25, 0.0, 1.5, 3.25] * 8),
        ("float64", [-1.25, 0.0, 1.5, 3.25] * 8),
    ],
)
def test_bulk_scalar_vectors_match_element_builds(
    scalar_type: str,
    values: list[bool | int | float],
) -> None:
    assert _finished_vector(values, scalar_type, reference=False) == (
        _finished_vector(values, scalar_type, reference=True)
    )


@pytest.mark.parametrize(
    ("scalar_type", "values"),
    [
        ("bool", [False] * 31 + [2]),
        ("bool", [False] * 31 + [float("nan")]),
        ("int8", [0] * 31 + [128]),
        ("uint8", [0] * 31 + [-1]),
        ("int32", [0] * 31 + [1.5]),
        ("float32", [0.0] * 31 + ["1.0"]),
        ("float32", [0.0] * 31 + [float(np.finfo(np.float32).max) * 2]),
        ("int32", np.zeros((8, 4), dtype=np.int32)),
    ],
)
def test_bulk_scalar_vectors_preserve_element_build_errors(
    scalar_type: str,
    values: Sequence[Any],
) -> None:
    with pytest.raises(Exception) as reference_error:
        _finished_vector(values, scalar_type, reference=True)

    with pytest.raises(type(reference_error.value)):
        _finished_vector(values, scalar_type, reference=False)
