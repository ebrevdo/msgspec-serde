from __future__ import annotations

import importlib.util
import shutil
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from msgspec_flatbuffers import BufferBoundsError, generate

HAS_FLATC = shutil.which("flatc") is not None
pytestmark = pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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

    model_buffer = model.to_flatbuffer()
    assert model_buffer.readonly
    assert type(model_buffer.obj).__name__ == "NativeBuffer"
    assert type(model_buffer.obj).__module__ == "msgspec_flatbuffers._native"
    assert generated.BasicView.from_buffer(model_buffer).to_model() == model

    packed_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
    )
    assert generated.BasicView.from_buffer(packed_buffer).to_model() == model

    presized_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        initial_size=128,
    )
    view = generated.BasicView.from_buffer(presized_buffer)

    assert view.to_model() == model
    assert presized_buffer[4:8] == b"BASC"

    prefixed_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        size_prefixed=True,
        initial_size=128,
    )
    assert generated.BasicView.from_buffer(
        prefixed_buffer,
        size_prefixed=True,
    ).to_model() == model
    assert generated.Basic.from_flatbuffer(
        prefixed_buffer,
        size_prefixed=True,
    ) == model

    undersized_frame = bytearray(prefixed_buffer)
    struct.pack_into("<I", undersized_frame, 0, 8)
    with pytest.raises(BufferBoundsError):
        generated.Basic.from_flatbuffer(undersized_frame, size_prefixed=True)

    invalid_utf8 = bytearray(model_buffer)
    label_start = invalid_utf8.find(b"built in Rust")
    assert label_start >= 0
    invalid_utf8[label_start] = 0xFF
    with pytest.raises(UnicodeDecodeError):
        generated.Basic.from_flatbuffer(invalid_utf8)

    defaults = generated.Basic(label="defaults")
    defaults_buffer = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        defaults,
        identifier="BASC",
    )
    assert generated.BasicView.from_buffer(defaults_buffer).to_model() == defaults

    class MissingLabel:
        flag = True
        count = 7
        ratio = 1.5
        label = None
        optional_count = None

    with pytest.raises(TypeError, match="required field.*label"):
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

    assert generated.ParentView.from_buffer(buffer).to_model() == model


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

    buffer = model.to_flatbuffer()
    view_restored = generated.VectorsView.from_buffer(buffer).to_model()
    model_restored = generated.Vectors.from_flatbuffer(buffer)

    assert view_restored == model
    assert model_restored == model
    _assert_writable_owned_arrays(model_restored, arrays)
    assert generated.Vectors.__annotations__["bools"] == (
        "npt.NDArray[np.bool_] | None"
    )
    assert generated.Vectors.__annotations__["i64s"] == (
        "npt.NDArray[np.int64] | None"
    )
    assert generated.Vectors.__annotations__["blob"] == "bytes | None"

    assert model_restored.f32s is not None
    model_restored.f32s[0] = 99.0
    source = generated.VectorsView.from_buffer(buffer).f32s
    assert source is not None
    assert source[0] == arrays["f32s"][0]

    empty_model = generated.Vectors(
        **{name: np.empty(0, dtype=array.dtype) for name, array in arrays.items()},
        blob=b"",
    )
    empty_restored = generated.Vectors.from_flatbuffer(empty_model.to_flatbuffer())
    assert empty_restored == empty_model
    _assert_writable_owned_arrays(empty_restored, arrays)

    strided = {
        name: np.repeat(array, 2)[::2] for name, array in arrays.items()
    }
    strided_model = generated.Vectors(**strided, blob=memoryview(b"strided"))
    assert generated.VectorsView.from_buffer(
        strided_model.to_flatbuffer()
    ).to_model() == strided_model
