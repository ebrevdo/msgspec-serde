"""Quick comparison of native plans and the Python FlatBuffers builder."""

from __future__ import annotations

import gc
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import flatbuffers

from msgspec_flatbuffers import generate


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _measure(function: Callable[[], object], number: int) -> float:
    samples: list[float] = []
    function()
    function()
    gc.collect()
    gc.disable()
    try:
        for _ in range(5):
            start = time.process_time_ns()
            for _ in range(number):
                function()
            samples.append((time.process_time_ns() - start) / number)
    finally:
        gc.enable()
    return statistics.median(samples)


def _finish(builder: flatbuffers.Builder, root: int, identifier: bytes | None) -> memoryview:
    builder.Finish(root, file_identifier=identifier)
    return memoryview(builder.Bytes)[builder.Head() :].toreadonly()


def _python_basic_pack(model: Any) -> memoryview:
    builder = flatbuffers.Builder(128)
    label = builder.CreateString(model.label)
    builder.StartObject(5)
    if model.optional_count is not None:
        builder.PrependInt32Slot(4, model.optional_count, 0)
    builder.PrependUOffsetTRelativeSlot(3, label, 0)
    builder.PrependFloat64Slot(2, model.ratio, 1.5)
    builder.PrependInt32Slot(1, model.count, 7)
    builder.PrependBoolSlot(0, model.flag, True)
    return _finish(builder, builder.EndObject(), b"BASC")


def _python_wide_pack(model: Any) -> memoryview:
    builder = flatbuffers.Builder(512)
    strings = [
        builder.CreateString(getattr(model, f"text_{index}"))
        for index in range(8)
    ]
    builder.StartObject(40)
    for index, offset in enumerate(strings, 32):
        builder.PrependUOffsetTRelativeSlot(index, offset, 0)
    for index in range(32):
        builder.PrependInt32Slot(index, getattr(model, f"value_{index}"), 0)
    return _finish(builder, builder.EndObject(), None)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="msgspec-flatbuffers-native-") as tmp:
        root = Path(tmp)
        schema = root / "basic.fbs"
        scalar_fields = [f"  value_{index}:int;" for index in range(32)]
        string_fields = [f"  text_{index}:string;" for index in range(8)]
        schema.write_text(
            "\n".join(
                [
                    "namespace NativeBench;",
                    "table Basic {",
                    "  flag:bool = true;",
                    "  count:int = 7;",
                    "  ratio:double = 1.5;",
                    "  label:string (required);",
                    "  optional_count:int = null;",
                    "}",
                    "table Wide {",
                    *scalar_fields,
                    *string_fields,
                    "}",
                    "root_type Basic;",
                    'file_identifier "BASC";',
                ]
            ),
            encoding="utf-8",
        )
        generated = _load_module(
            "native_benchmark.basic",
            generate(schema, root / "generated", project_root=root),
        )

        basic = generated.Basic(
            flag=False,
            count=42,
            ratio=3.25,
            label="built in Rust",
            optional_count=9,
        )
        wide_values: dict[str, int | str] = {
            f"value_{index}": index + 1 for index in range(32)
        }
        wide_values.update(
            {f"text_{index}": f"value-{index}" for index in range(8)}
        )
        wide = generated.Wide(**wide_values)

        cases = {
            "basic": (
                basic,
                lambda: _python_basic_pack(basic),
                basic.to_flatbuffer,
                5_000,
            ),
            "wide": (
                wide,
                lambda: _python_wide_pack(wide),
                wide.to_flatbuffer,
                500,
            ),
        }
        result: dict[str, dict[str, float | int]] = {}
        for name, (model, python_pack, native_pack, number) in cases.items():
            python_buffer = python_pack()
            native_buffer = native_pack()
            view_type = getattr(generated, f"{type(model).__name__}View")
            assert view_type.from_buffer(native_buffer).to_model() == model
            python_ns = _measure(python_pack, number)
            native_ns = _measure(native_pack, number)
            result[name] = {
                "python_us": round(python_ns / 1_000, 3),
                "rust_us": round(native_ns / 1_000, 3),
                "rust_over_python": round(native_ns / python_ns, 3),
                "python_bytes": len(python_buffer),
                "rust_bytes": len(native_buffer),
            }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
