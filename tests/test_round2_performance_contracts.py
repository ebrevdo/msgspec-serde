from __future__ import annotations

import struct
from collections.abc import Buffer
from itertools import islice
from typing import Any, ClassVar, Self, cast

import pytest

from msgspec_flatbuffers import (
    BaseType,
    BufferBoundsError,
    CachedVector,
    EnumDefinition,
    EnumValue,
    FieldDefinition,
    GenerationError,
    InvalidBufferError,
    ObjectDefinition,
    Schema,
    StructVector,
    StructView,
    TableVector,
    TableView,
    TypeReference,
    flatbuffer,
    render_module,
)


class IndexedObject:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


class IndexedVector(CachedVector[IndexedObject]):
    def __init__(self, length: int) -> None:
        super().__init__(length)
        self.loaded: list[int] = []

    def _load(self, index: int) -> IndexedObject:
        self.loaded.append(index)
        return IndexedObject(index)


class PairView(StructView):
    __slots__ = ()
    __flatbuffer_size__ = 8

    @property
    def left(self) -> int:
        return cast(int, self._read_scalar(0, "int32"))

    @property
    def right(self) -> int:
        return cast(int, self._read_scalar(4, "int32"))


class CountingPairView(PairView):
    __slots__ = ()
    init_calls: ClassVar[int] = 0

    def __init__(self, buffer: Buffer, struct_offset: int) -> None:
        type(self).init_calls += 1
        super().__init__(buffer, struct_offset)


class CountingViewMeta(type):
    calls: int = 0

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        CountingViewMeta.calls += 1
        return super().__call__(*args, **kwargs)


class MetaclassPairView(PairView, metaclass=CountingViewMeta):
    __slots__ = ()


class OverriddenFactoryTableView(TableView):
    __slots__ = ()
    validated_factory_calls: ClassVar[int] = 0

    @classmethod
    def _from_validated_vector(
        cls,
        buffer: memoryview,
        table_offset: int,
        vtable_offset: int,
        vtable_size: int,
        object_size: int,
    ) -> Self:
        cls.validated_factory_calls += 1
        return super()._from_validated_vector(
            buffer,
            table_offset,
            vtable_offset,
            vtable_size,
            object_size,
        )


def _single_table_vector_buffer(
    *,
    size: int = 16,
    target: int = 8,
    vtable_distance: int = 4,
    vtable_size: int = 4,
    object_size: int = 4,
) -> bytearray:
    buffer = bytearray(size)
    struct.pack_into("<I", buffer, 0, target)
    if target > size - 4:
        return buffer

    struct.pack_into("<i", buffer, target, vtable_distance)
    vtable_offset = target - vtable_distance
    if 0 <= vtable_offset <= size - 4:
        struct.pack_into(
            "<HH",
            buffer,
            vtable_offset,
            vtable_size,
            object_size,
        )
    return buffer


def _rendered_namespace(schema: Schema, declaration_file: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(render_module(schema, declaration_file), namespace)
    return namespace


def _scalar_table_schema(offset: int = 4) -> Schema:
    declaration_file = "//scalar.fbs"
    root = ObjectDefinition(
        name="Round2.ScalarRoot",
        fields=(
            FieldDefinition(
                name="value",
                type=TypeReference(base_type=BaseType.INT),
                id=0,
                offset=offset,
            ),
        ),
        declaration_file=declaration_file,
    )
    return Schema(objects=(root,), enums=(), root_table=root.name)


def _padded_struct_schema() -> Schema:
    declaration_file = "//padded.fbs"
    mode = EnumDefinition(
        name="Round2.Mode",
        values=(
            EnumValue(name="Zero", value=0),
            EnumValue(name="Seven", value=7),
        ),
        underlying_type=TypeReference(base_type=BaseType.UBYTE),
        declaration_file=declaration_file,
    )
    padded = ObjectDefinition(
        name="Round2.Padded",
        fields=(
            FieldDefinition(
                name="flag",
                type=TypeReference(base_type=BaseType.UBYTE),
                id=0,
                offset=0,
            ),
            FieldDefinition(
                name="mode",
                type=TypeReference(base_type=BaseType.UBYTE, index=0),
                id=1,
                offset=3,
            ),
            FieldDefinition(
                name="count",
                type=TypeReference(base_type=BaseType.INT),
                id=2,
                offset=4,
            ),
            FieldDefinition(
                name="delta",
                type=TypeReference(base_type=BaseType.SHORT),
                id=3,
                offset=8,
            ),
        ),
        is_struct=True,
        min_alignment=4,
        byte_size=12,
        declaration_file=declaration_file,
    )
    return Schema(objects=(padded,), enums=(mode,))


def test_huge_partial_iteration_stays_sparse_and_loads_only_the_prefix() -> None:
    vector = IndexedVector(1_000_000)

    prefix = tuple(islice(vector, 96))

    assert tuple(value.index for value in prefix) == tuple(range(96))
    assert vector.loaded == list(range(96))
    assert vector.cached_count == 96
    assert vector[95] is prefix[-1]
    # A short scan of a huge vector must not retain a million-element list.
    assert not isinstance(vector._cache, list)


def test_fully_cached_vector_preserves_dense_index_and_slice_semantics() -> None:
    vector = IndexedVector(128)
    seeded = {index: vector[index] for index in range(0, 128, 2)}

    values = tuple(vector)

    assert vector.cached_count == len(vector)
    assert isinstance(vector._cache, tuple)
    assert tuple(value.index for value in values) == tuple(range(128))
    assert all(values[index] is value for index, value in seeded.items())
    assert vector[-1] is values[-1]
    assert vector[5:31:4] == values[5:31:4]
    assert vector[::-11] == values[::-11]
    with pytest.raises(IndexError, match="tuple index out of range"):
        _ = vector[len(vector)]
    with pytest.raises(IndexError, match="tuple index out of range"):
        _ = vector[-len(vector) - 1]


def test_struct_vector_reuses_a_validated_read_only_span() -> None:
    buffer = bytearray(16)
    struct.pack_into("<iiii", buffer, 0, 1, 2, 3, 4)
    vector = StructVector(memoryview(buffer), 0, 2, 8, PairView)

    first = vector[0]

    assert (first.left, first.right) == (1, 2)
    assert (vector[1].left, vector[1].right) == (3, 4)
    assert vector[0] is first
    assert first.buffer.readonly
    assert first.buffer.obj is buffer
    with pytest.raises(TypeError, match="read-only"):
        first.buffer[0] = 0


def test_struct_vector_calls_custom_view_constructor() -> None:
    CountingPairView.init_calls = 0
    vector = StructVector(memoryview(bytes(8)), 0, 1, 8, CountingPairView)

    assert vector[0] is vector[0]
    assert CountingPairView.init_calls == 1


def test_struct_vector_calls_custom_metaclass() -> None:
    CountingViewMeta.calls = 0
    vector = StructVector(memoryview(bytes(8)), 0, 1, 8, MetaclassPairView)

    assert vector[0] is vector[0]
    assert CountingViewMeta.calls == 1


@pytest.mark.parametrize(("start", "length"), [(-1, 1), (9, 1), (0, 3)])
def test_struct_vector_rejects_corrupt_spans(start: int, length: int) -> None:
    with pytest.raises(BufferBoundsError, match="struct vector data"):
        StructVector(memoryview(bytes(16)), start, length, 8, PairView)


def test_table_vector_accepts_a_valid_negative_vtable_distance() -> None:
    buffer = _single_table_vector_buffer(target=4, vtable_distance=-8)
    vector = TableVector(memoryview(buffer), 0, 1, TableView)

    view = vector[0]

    assert view.table_offset == 4
    assert view._vtable_offset == 12
    assert view.buffer.readonly
    assert vector[0] is view


def test_table_vector_rejects_a_non_table_view_type() -> None:
    with pytest.raises(TypeError, match="TableView"):
        TableVector(
            memoryview(_single_table_vector_buffer()),
            0,
            1,
            PairView,
        )


def test_table_vector_rejects_a_truncated_table_header() -> None:
    buffer = _single_table_vector_buffer(size=8, target=6)

    with pytest.raises(BufferBoundsError, match="table header"):
        _ = TableVector(memoryview(buffer), 0, 1, TableView)[0]


@pytest.mark.parametrize("vtable_distance", [20, -10])
def test_table_vector_rejects_an_out_of_bounds_vtable_header(
    vtable_distance: int,
) -> None:
    buffer = _single_table_vector_buffer(
        size=12,
        target=4,
        vtable_distance=vtable_distance,
    )

    with pytest.raises(BufferBoundsError, match="vtable header"):
        _ = TableVector(memoryview(buffer), 0, 1, TableView)[0]


@pytest.mark.parametrize("vtable_size", [2, 5])
def test_table_vector_rejects_invalid_vtable_metadata(vtable_size: int) -> None:
    buffer = _single_table_vector_buffer(vtable_size=vtable_size)

    with pytest.raises(InvalidBufferError, match="table metadata"):
        _ = TableVector(memoryview(buffer), 0, 1, TableView)[0]


def test_table_vector_rejects_a_vtable_span_overflow() -> None:
    buffer = _single_table_vector_buffer(vtable_size=14)

    with pytest.raises(BufferBoundsError, match="vtable at offset"):
        _ = TableVector(memoryview(buffer), 0, 1, TableView)[0]


def test_table_vector_rejects_an_object_span_overflow() -> None:
    buffer = _single_table_vector_buffer(size=12, object_size=8)

    with pytest.raises(BufferBoundsError, match="table at offset"):
        _ = TableVector(memoryview(buffer), 0, 1, TableView)[0]


def test_table_vector_rejects_a_null_element_offset() -> None:
    with pytest.raises(InvalidBufferError, match="null offset"):
        _ = TableVector(memoryview(bytes(4)), 0, 1, TableView)[0]


def test_table_vector_does_not_call_an_overridden_validated_factory() -> None:
    OverriddenFactoryTableView.validated_factory_calls = 0
    vector = TableVector(
        memoryview(_single_table_vector_buffer()),
        0,
        1,
        OverriddenFactoryTableView,
    )

    assert isinstance(vector[0], OverriddenFactoryTableView)
    assert OverriddenFactoryTableView.validated_factory_calls == 0


@pytest.mark.parametrize("offset", [-2, 2, 5])
def test_generator_rejects_invalid_scalar_vtable_offsets(offset: int) -> None:
    with pytest.raises(GenerationError, match="invalid vtable offset"):
        render_module(_scalar_table_schema(offset), "//scalar.fbs")


def test_generator_rejects_a_byte_vector_helper_field_collision() -> None:
    declaration_file = "//collision.fbs"
    root = ObjectDefinition(
        name="Round2.Collision",
        fields=(
            FieldDefinition(
                name="data",
                type=TypeReference(
                    base_type=BaseType.VECTOR,
                    element=BaseType.UBYTE,
                ),
                id=0,
                offset=4,
            ),
            FieldDefinition(
                name="data_as",
                type=TypeReference(base_type=BaseType.INT),
                id=1,
                offset=6,
            ),
        ),
        declaration_file=declaration_file,
    )
    schema = Schema(objects=(root,), enums=(), root_table=root.name)

    with pytest.raises(GenerationError, match="conflicts with a field"):
        render_module(schema, declaration_file)


@pytest.mark.parametrize("relative", [1, 2, 3, 5])
def test_generated_scalar_reads_reject_positions_outside_the_table(
    relative: int,
) -> None:
    namespace = _rendered_namespace(_scalar_table_schema(), "//scalar.fbs")
    root_type = namespace["ScalarRoot"]
    view_type = namespace["ScalarRootView"]
    buffer = bytearray(flatbuffer.encode(root_type(value=123)))
    valid_view = flatbuffer.decode(buffer, type=view_type)
    struct.pack_into("<H", buffer, valid_view._vtable_offset + 4, relative)
    corrupt_view = flatbuffer.decode(buffer, type=view_type)

    with pytest.raises(InvalidBufferError, match="outside"):
        _ = corrupt_view.value
    with pytest.raises(InvalidBufferError, match="outside"):
        corrupt_view.to_model()


def test_generated_padded_scalar_struct_converts_enums_and_padding() -> None:
    namespace = _rendered_namespace(_padded_struct_schema(), "//padded.fbs")
    model_type = namespace["Padded"]
    view_type = namespace["PaddedView"]
    mode_type = namespace["Mode"]
    buffer = struct.pack("<B2xBIh2x", 9, 7, 123_456, -321)

    view = view_type(buffer, 0)

    assert view.to_model() == model_type(
        flag=9,
        mode=mode_type.Seven,
        count=123_456,
        delta=-321,
    )
    assert view.mode is mode_type.Seven
    assert view.buffer.readonly
