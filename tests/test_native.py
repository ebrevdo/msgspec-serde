from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from msgspec_flatbuffers import generate


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


def test_native_plan_builds_compatible_flatbuffers(tmp_path: Path) -> None:
    schema = tmp_path / "basic.fbs"
    schema.write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
    )
    generated = _load_module(
        "native.basic",
        generate(schema, tmp_path / "generated", project_root=tmp_path),
    )
    model = generated.Basic(
        flag=False,
        count=42,
        ratio=3.25,
        label="built in Rust",
        optional_count=9,
    )

    generated_buffer = model.to_flatbuffer()
    assert generated_buffer.readonly
    assert type(generated_buffer.obj).__name__ == "NativeBuffer"
    assert type(generated_buffer.obj).__module__ == "msgspec_flatbuffers._native"
    assert generated.BasicView.from_buffer(generated_buffer).to_model() == model

    module_native = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
    )
    assert generated.BasicView.from_buffer(module_native).to_model() == model

    native = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        initial_size=128,
    )
    view = generated.BasicView.from_buffer(native)

    assert view.to_model() == model
    assert native[4:8] == b"BASC"

    prefixed = generated._FB_NATIVE_MODULE.pack(
        "Native.Basic",
        model,
        identifier="BASC",
        size_prefixed=True,
        initial_size=128,
    )
    assert generated.BasicView.from_buffer(
        prefixed,
        size_prefixed=True,
    ).to_model() == model

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
    schema = tmp_path / "nested.fbs"
    schema.write_text(
        "\n".join(
            [
                "namespace Native;",
                "table Child { value:int; }",
                "table Parent { child:Child; label:string; }",
                "root_type Parent;",
                'file_identifier "NEST";',
            ]
        ),
        encoding="utf-8",
    )
    generated = _load_module(
        "native.nested",
        generate(schema, tmp_path / "generated", project_root=tmp_path),
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
    schema = tmp_path / "vectors.fbs"
    schema.write_text(
        " ".join(
            (
                "namespace Native;",
                "table Vectors {",
                "bools:[bool]; i8s:[byte]; i16s:[short]; u16s:[ushort];",
                "i32s:[int]; u32s:[uint]; i64s:[long]; u64s:[ulong];",
                "f32s:[float]; f64s:[double]; blob:[ubyte]; }",
                "root_type Vectors;",
                'file_identifier "NVEC";',
            )
        ),
        encoding="utf-8",
    )
    generated = _load_module(
        "native.vectors",
        generate(schema, tmp_path / "generated", project_root=tmp_path),
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

    restored = generated.VectorsView.from_buffer(model.to_flatbuffer()).to_model()

    assert restored == model
    assert generated.Vectors.__annotations__["bools"] == (
        "npt.NDArray[np.bool_] | None"
    )
    assert generated.Vectors.__annotations__["i64s"] == (
        "npt.NDArray[np.int64] | None"
    )
    assert generated.Vectors.__annotations__["blob"] == "bytes | None"

    strided = {
        name: np.repeat(array, 2)[::2]
        for name, array in arrays.items()
    }
    strided_model = generated.Vectors(**strided, blob=memoryview(b"strided"))
    assert generated.VectorsView.from_buffer(
        strided_model.to_flatbuffer()
    ).to_model() == strided_model
