from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from msgspec_serde import BufferBoundsError, CachedVector, TableView, flatbuffer


class ExampleView(TableView):
    pass


class StringExampleView(TableView):
    @property
    def name(self) -> str | None:
        return self._read_string(4)


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


def _minimal_table_buffer() -> bytearray:
    buffer = bytearray(12)
    struct.pack_into("<I", buffer, 0, 8)
    struct.pack_into("<HH", buffer, 4, 4, 4)
    struct.pack_into("<i", buffer, 8, 4)
    return buffer


def _size_prefixed(payload: bytes | bytearray) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def test_decode_keeps_a_read_only_view_of_the_original_buffer() -> None:
    buffer = _minimal_table_buffer()

    view = flatbuffer.decode(buffer, type=ExampleView)

    assert view.table_offset == 8
    assert view.buffer.readonly
    assert view.buffer.obj is buffer


def test_size_prefixed_root_is_bounded_to_its_declared_payload() -> None:
    payload = _minimal_table_buffer()
    leading = b"lead"
    trailing = b"trailing frame"
    framed = bytearray(leading + _size_prefixed(payload) + trailing)

    view = flatbuffer.decode(
        framed,
        type=ExampleView,
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

    with pytest.raises(BufferBoundsError, match="exceeds a .*byte buffer"):
        flatbuffer.decode(framed, type=ExampleView, size_prefixed=True)


def test_size_prefixed_root_rejects_a_declared_payload_past_the_input() -> None:
    framed = struct.pack("<I", 12) + bytes(11)

    with pytest.raises(BufferBoundsError, match="size-prefixed buffer"):
        flatbuffer.decode(framed, type=ExampleView, size_prefixed=True)


def test_concatenated_size_prefixed_roots_keep_separate_buffers() -> None:
    payload = _minimal_table_buffer()
    frame = _size_prefixed(payload)
    combined = bytearray(frame + frame)

    first = flatbuffer.decode(combined, type=ExampleView, size_prefixed=True)
    second = flatbuffer.decode(
        combined,
        type=ExampleView,
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
        flatbuffer.decode(
            framed,
            type=ExampleView,
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

    view = flatbuffer.decode(framed, type=StringExampleView, size_prefixed=True)

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
def test_decode_rejects_out_of_bounds_offsets(
    buffer: bytes,
    offset: int,
) -> None:
    with pytest.raises(BufferBoundsError, match="exceeds a .*byte buffer"):
        flatbuffer.decode(buffer, type=ExampleView, offset=offset)


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
