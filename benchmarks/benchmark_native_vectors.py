"""CPU-time benchmarks for native scalar-vector and byte-buffer inputs."""

from __future__ import annotations

import gc
import importlib.util
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from msgspec_flatbuffers import generate

VECTOR_SIZE = 65_536
BYTE_SIZE = 1 << 20
REPEAT = 9


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("native_vector_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _measure(function: Callable[[], object], number: int) -> tuple[float, float]:
    for _ in range(3):
        function()
    gc.collect()
    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(REPEAT):
            start = time.process_time_ns()
            for _ in range(number):
                function()
            samples.append((time.process_time_ns() - start) / number / 1_000)
    finally:
        gc.enable()
    return min(samples), statistics.median(samples)


def _schema(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "namespace NativeVectors;",
                "table Vectors {",
                "  bools:[bool];",
                "  i8s:[byte];",
                "  i16s:[short];",
                "  u16s:[ushort];",
                "  i32s:[int];",
                "  u32s:[uint];",
                "  i64s:[long];",
                "  u64s:[ulong];",
                "  f32s:[float];",
                "  f64s:[double];",
                "  blob:[ubyte];",
                "}",
                "root_type Vectors;",
                'file_identifier "NVEC";',
            )
        ),
        encoding="utf-8",
    )


def _exact_arrays() -> dict[str, np.ndarray[Any, Any]]:
    indices = np.arange(VECTOR_SIZE)
    return {
        "bools": (indices % 2).astype(np.bool_),
        "i8s": ((indices % 255) - 127).astype(np.int8),
        "i16s": ((indices % 65_535) - 32_767).astype(np.int16),
        "u16s": (indices % 65_536).astype(np.uint16),
        "i32s": (indices - VECTOR_SIZE // 2).astype(np.int32),
        "u32s": indices.astype(np.uint32),
        "i64s": (indices - VECTOR_SIZE // 2).astype(np.int64),
        "u64s": indices.astype(np.uint64),
        "f32s": (indices / 16).astype(np.float32),
        "f64s": (indices / 16).astype(np.float64),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="msgspec-flatbuffers-vectors-") as tmp:
        root = Path(tmp)
        schema = root / "vectors.fbs"
        _schema(schema)
        generated = _load_module(
            generate(schema, root / "generated", project_root=root)
        )

        cases: dict[str, tuple[object, int]] = {}
        arrays = _exact_arrays()
        for field, array in arrays.items():
            cases[f"numpy_{field}"] = (generated.Vectors(**{field: array}), 10)
            base = np.empty(array.size * 2, dtype=array.dtype)
            base[::2] = array
            cases[f"numpy_{field}_strided"] = (
                generated.Vectors(**{field: base[::2]}),
                5,
            )

        blob = bytes(range(256)) * (BYTE_SIZE // 256)
        native = generated.Vectors(blob=blob).to_flatbuffer()
        cases.update(
            {
                "bytes": (generated.Vectors(blob=blob), 5),
                "bytearray": (generated.Vectors(blob=bytearray(blob)), 5),
                "memoryview_bytes": (generated.Vectors(blob=memoryview(blob)), 5),
                "memoryview_native": (generated.Vectors(blob=native), 5),
            }
        )

        print(f"vector length: {VECTOR_SIZE:,}; byte payload: {BYTE_SIZE:,}")
        for name, (model, number) in cases.items():
            function = model.to_flatbuffer
            buffer = function()
            generated.VectorsView.from_buffer(buffer).to_model()
            best, median = _measure(function, number)
            print(f"{name:<28} best={best:>10.3f} us  median={median:>10.3f} us")


if __name__ == "__main__":
    main()
