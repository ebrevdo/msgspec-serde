from __future__ import annotations

import importlib.util
import shutil
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import flatbuffers
import msgspec
import numpy as np
import pytest

from msgspec_flatbuffers import (
    BufferBoundsError,
    flatbuffer,
    generate,
    json,
    msgpack,
)

HAS_FLATC = shutil.which("flatc") is not None
pytestmark = pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")


def _load_module(name: str, path: Path) -> ModuleType:
    generated_root = path.parent
    while (generated_root / "__init__.py").is_file():
        generated_root = generated_root.parent
    root_string = str(generated_root)
    generated_packages = {
        child.name
        for child in generated_root.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }
    generated_modules = {child.stem for child in generated_root.glob("*.py")}
    for imported_name in tuple(sys.modules):
        if (
            imported_name.split(".", 1)[0] in generated_packages
            or imported_name in generated_modules
            or imported_name == name
        ):
            sys.modules.pop(imported_name, None)
    sys.path.insert(0, root_string)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root_string)
    return module


def _write_schema(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _generate_module(
    tmp_path: Path,
    name: str,
    *schema_lines: str,
) -> ModuleType:
    schema = tmp_path / f"{name}.fbs"
    _write_schema(schema, *schema_lines)
    return _load_module(
        f"native.{name}",
        generate(schema, tmp_path / "generated", project_root=tmp_path),
    )


def _assert_writable_owned_arrays(model: object, names: Iterable[str]) -> None:
    for name in names:
        array = getattr(model, name)
        assert array.flags.owndata
        assert array.flags.writeable


def _assert_serde_round_trip(model: msgspec.Struct) -> None:
    for codec in (json, msgpack):
        assert codec.decode(codec.encode(model), type=type(model)) == model


def test_native_plan_builds_compatible_flatbuffers(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "basic",
        "namespace Native;",
        "table Basic {",
        "  flag:bool = true;",
        "  count:int = 7;",
        "  ratio:double = 1.5;",
        "  label:string (required);",
        "  optional_count:int = null;",
        "}",
        "root_type Basic;",
        'file_identifier "BASC";',
    )
    model = generated.Basic(
        flag=False,
        count=42,
        ratio=3.25,
        label="built in Rust",
        optional_count=9,
    )

    model_buffer = flatbuffer.encode(model)
    assert model_buffer.readonly
    assert type(model_buffer.obj).__name__ == "NativeBuffer"
    assert type(model_buffer.obj).__module__ == "msgspec_flatbuffers._native"
    assert flatbuffer.decode(model_buffer, type=generated.BasicView).to_model() == model

    packed_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
    )
    assert (
        flatbuffer.decode(packed_buffer, type=generated.BasicView).to_model() == model
    )

    presized_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        initial_size=128,
    )
    view = flatbuffer.decode(presized_buffer, type=generated.BasicView)

    assert view.to_model() == model
    assert presized_buffer[4:8] == b"BASC"

    prefixed_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        size_prefixed=True,
        initial_size=128,
    )
    assert (
        flatbuffer.decode(
            prefixed_buffer,
            type=generated.BasicView,
            size_prefixed=True,
        ).to_model()
        == model
    )
    assert (
        flatbuffer.decode(
            prefixed_buffer,
            type=generated.Basic,
            size_prefixed=True,
        )
        == model
    )

    undersized_frame = bytearray(prefixed_buffer)
    struct.pack_into("<I", undersized_frame, 0, 8)
    with pytest.raises(BufferBoundsError, match="int32 at offset"):
        flatbuffer.decode(undersized_frame, type=generated.Basic, size_prefixed=True)

    invalid_utf8 = bytearray(model_buffer)
    label_start = invalid_utf8.find(b"built in Rust")
    assert label_start >= 0
    invalid_utf8[label_start] = 0xFF
    with pytest.raises(UnicodeDecodeError, match="invalid utf-8 sequence"):
        flatbuffer.decode(invalid_utf8, type=generated.Basic)

    defaults = generated.Basic(label="defaults")
    defaults_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        defaults,
        identifier="BASC",
    )
    assert (
        flatbuffer.decode(defaults_buffer, type=generated.BasicView).to_model()
        == defaults
    )

    class MissingLabel:
        flag = True
        count = 7
        ratio = 1.5
        label = None
        optional_count = None

    with pytest.raises(TypeError, match="generated FlatBuffer model base"):
        generated._FB_NATIVE_MODULE.pack("Native.Basic", MissingLabel())
    with pytest.raises(ValueError, match="initial_size"):
        generated._FB_NATIVE_MODULE.pack(
            "Native.Basic",
            MissingLabel(),
            initial_size=-1,
        )


def test_native_module_plan_builds_nested_tables(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "nested",
        "namespace Native;",
        "table Child { value:int; }",
        "table Parent { child:Child; label:string; }",
        "root_type Parent;",
        'file_identifier "NEST";',
    )
    model = generated.Parent(
        child=generated.Child(value=42),
        label="parent",
    )

    buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Parent",
        model,
        identifier="NEST",
    )

    assert flatbuffer.decode(buffer, type=generated.ParentView).to_model() == model


def test_native_materialization_handles_deep_recursive_tables(
    tmp_path: Path,
) -> None:
    generated = _generate_module(
        tmp_path,
        "recursive",
        "namespace Native;",
        "table Node { next:Node; value:int; }",
        "root_type Node;",
    )
    depth = 20_000
    builder = flatbuffers.Builder(300_000)
    node = 0
    for value in range(depth):
        builder.StartObject(2)
        builder.PrependInt32Slot(1, value, 0)
        if node:
            builder.PrependUOffsetTRelativeSlot(0, node, 0)
        node = builder.EndObject()
    builder.Finish(node)

    current = flatbuffer.decode(builder.Output(), type=generated.Node)
    count = 0
    while current is not None:
        following = current.next
        current.next = None
        current = following
        count += 1

    assert count == depth


def test_nested_structs_and_fixed_arrays_round_trip(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "fixed_arrays",
        "namespace Native;",
        "enum Mode : byte { Off = 0, On = 1 }",
        "struct Point { x:int; y:int; }",
        "struct Pair { first:Point; second:Point; }",
        "struct FixedData {",
        "  values:[int:4];",
        "  modes:[Mode:2];",
        "  points:[Point:2];",
        "  pair:Pair;",
        "}",
        "table Root { data:FixedData; }",
        "root_type Root;",
    )
    point_a = generated.Point(x=1, y=2)
    point_b = generated.Point(x=3, y=4)
    data = generated.FixedData(
        values=np.array([10, 20, 30, 40], dtype=np.int32),
        modes=[generated.Mode.Off, generated.Mode.On],
        points=[point_a, point_b],
        pair=generated.Pair(first=point_b, second=point_a),
    )
    model = generated.Root(data=data)

    buffer = flatbuffer.encode(model)
    restored = flatbuffer.decode(buffer, type=generated.Root)
    view = flatbuffer.decode(buffer, type=generated.RootView)

    assert restored == model
    _assert_serde_round_trip(model)
    assert restored.data is not None
    _assert_writable_owned_arrays(restored.data, ["values"])
    assert view.data is not None
    np.testing.assert_array_equal(view.data.values, [10, 20, 30, 40])
    assert not view.data.values.flags.writeable
    np.testing.assert_array_equal(view.data.modes, [0, 1])
    assert view.data.points[0].x == 1
    assert view.data.points[0] is view.data.points[0]
    assert view.data.pair.first.y == 4
    assert generated.FixedData.__annotations__["values"] == ("npt.NDArray[np.int32]")
    assert generated.FixedData.__annotations__["modes"].endswith(".Mode]")

    data.values = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="fixed array requires 4 items"):
        flatbuffer.encode(model)


def test_union_vectors_round_trip_none_elements(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "nullable_union_vector",
        "namespace Native;",
        "table Cat { lives:int = 9; }",
        "table Dog { good:bool = true; }",
        "union Pet { Cat, Dog }",
        "table Root { pets:[Pet]; }",
        "root_type Root;",
    )
    model = generated.Root(
        pets=[generated.Cat(lives=7), None, generated.Dog(good=False)]
    )

    buffer = flatbuffer.encode(model)
    restored = flatbuffer.decode(buffer, type=generated.Root)
    pets = flatbuffer.decode(buffer, type=generated.RootView).pets

    assert restored == model
    _assert_serde_round_trip(model)
    assert pets is not None
    assert pets[0].lives == 7
    assert pets[1] is None
    assert not pets[2].good
    model_annotation = generated.Root.__annotations__["pets"]
    view_annotation = generated.RootView.pets.fget.__annotations__["return"]
    assert model_annotation.startswith("list[")
    assert ".Cat" in model_annotation and ".Dog" in model_annotation
    assert model_annotation.endswith(" | None] | None")
    assert view_annotation.startswith("UnionVector[")
    assert ".CatView" in view_annotation and ".DogView" in view_annotation
    assert view_annotation.endswith(" | None] | None")


def test_fixed_arrays_round_trip_every_numpy_dtype(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "fixed_array_dtypes",
        "namespace Native;",
        "struct FixedArrays {",
        "bools:[bool:3]; i8s:[byte:3]; u8s:[ubyte:3];",
        "i16s:[short:3]; u16s:[ushort:3];",
        "i32s:[int:3]; u32s:[uint:3];",
        "i64s:[long:3]; u64s:[ulong:3];",
        "f32s:[float:3]; f64s:[double:3];",
        "}",
        "table Root { values:FixedArrays; }",
        "root_type Root;",
    )
    arrays = {
        "bools": np.array([True, False, True], dtype=np.bool_),
        "i8s": np.array([-2, 0, 3], dtype=np.int8),
        "u8s": np.array([0, 3, 255], dtype=np.uint8),
        "i16s": np.array([-300, 0, 400], dtype=np.int16),
        "u16s": np.array([0, 400, 65_535], dtype=np.uint16),
        "i32s": np.array([-100_000, 0, 200_000], dtype=np.int32),
        "u32s": np.array([0, 200_000, 2**32 - 1], dtype=np.uint32),
        "i64s": np.array([-(2**50), 0, 2**50], dtype=np.int64),
        "u64s": np.array([0, 2**50, 2**63], dtype=np.uint64),
        "f32s": np.array([-1.25, 0.0, 2.5], dtype=np.float32),
        "f64s": np.array([-1.25, 0.0, 2.5], dtype=np.float64),
    }
    model = generated.Root(values=generated.FixedArrays(**arrays))

    restored = flatbuffer.decode(flatbuffer.encode(model), type=generated.Root)

    assert restored == model
    assert restored.values is not None
    _assert_writable_owned_arrays(restored.values, arrays)


def test_native_plan_round_trips_every_numpy_vector_dtype(tmp_path: Path) -> None:
    generated = _generate_module(
        tmp_path,
        "vectors",
        "namespace Native;",
        "table Vectors {",
        "bools:[bool]; i8s:[byte]; i16s:[short]; u16s:[ushort];",
        "i32s:[int]; u32s:[uint]; i64s:[long]; u64s:[ulong];",
        "f32s:[float]; f64s:[double]; blob:[ubyte]; }",
        "root_type Vectors;",
        'file_identifier "NVEC";',
    )
    arrays = {
        "bools": np.array([True, False, True], dtype=np.bool_),
        "i8s": np.array([-2, 0, 3], dtype=np.int8),
        "i16s": np.array([-300, 0, 400], dtype=np.int16),
        "u16s": np.array([0, 400, 65_535], dtype=np.uint16),
        "i32s": np.array([-100_000, 0, 200_000], dtype=np.int32),
        "u32s": np.array([0, 200_000, 2**32 - 1], dtype=np.uint32),
        "i64s": np.array([-(2**50), 0, 2**50], dtype=np.int64),
        "u64s": np.array([0, 2**50, 2**63], dtype=np.uint64),
        "f32s": np.array([-1.25, 0.0, 2.5], dtype=np.float32),
        "f64s": np.array([-1.25, 0.0, 2.5], dtype=np.float64),
    }
    model = generated.Vectors(**arrays, blob=b"\x00\x01\xff")

    buffer = flatbuffer.encode(model)
    view_restored = flatbuffer.decode(buffer, type=generated.VectorsView).to_model()
    model_restored = flatbuffer.decode(buffer, type=generated.Vectors)

    assert view_restored == model
    assert model_restored == model
    _assert_writable_owned_arrays(model_restored, arrays)
    assert generated.Vectors.__annotations__["bools"] == (
        "npt.NDArray[np.bool_] | None"
    )
    assert generated.Vectors.__annotations__["i64s"] == ("npt.NDArray[np.int64] | None")
    assert generated.Vectors.__annotations__["blob"] == "bytes | None"

    assert model_restored.f32s is not None
    model_restored.f32s[0] = 99.0
    source = flatbuffer.decode(buffer, type=generated.VectorsView).f32s
    assert source is not None
    assert source[0] == arrays["f32s"][0]

    empty_model = generated.Vectors(
        **{name: np.empty(0, dtype=array.dtype) for name, array in arrays.items()},
        blob=b"",
    )
    empty_restored = flatbuffer.decode(
        flatbuffer.encode(empty_model), type=generated.Vectors
    )
    assert empty_restored == empty_model
    _assert_writable_owned_arrays(empty_restored, arrays)

    strided = {name: np.repeat(array, 2)[::2] for name, array in arrays.items()}
    strided_model = generated.Vectors(**strided, blob=memoryview(b"strided"))
    assert (
        flatbuffer.decode(
            flatbuffer.encode(strided_model), type=generated.VectorsView
        ).to_model()
        == strided_model
    )
