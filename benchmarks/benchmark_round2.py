"""Second-round benchmarks for cold views, dense caches, and builders.

Run from the repository root with::

    uv run --no-sync python benchmarks/benchmark_round2.py --json results.json

The workloads, iteration counts, and seven-sample repetition are fixed so that
source-pinned baseline and optimized runs are directly comparable. ``timeit``
disables cyclic garbage collection during each timed sample.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import timeit
from collections.abc import Buffer, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

REPEAT = 7
TABLE_SCAN_COUNT = 256
STRUCT_SCAN_COUNT = 256
MODEL_SCORE_COUNT = 4096
MODEL_TAG_COUNT = 64
MODEL_INVENTORY_BYTES = 4096

LARGE_BYTE_COUNT = 1 << 20
LARGE_NUMERIC_COUNT = 65_536
LARGE_STRUCT_COUNT = 4096
LARGE_TABLE_COUNT = 1024
LARGE_STRING_COUNT = 1024

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "monster.fbs"


class BenchmarkSpec(NamedTuple):
    """A fixed timeit workload and its reporting scale."""

    label: str
    statement: str
    number: int
    warmup_number: int
    unit: str
    scale: float
    note: str


class PreparedFixture(NamedTuple):
    """Generated objects and metadata kept alive for the benchmark run."""

    temporary_directory: tempfile.TemporaryDirectory[str]
    generated: ModuleType
    main_buffer: memoryview
    reused_view: Any
    build_models: dict[str, Any]
    build_metadata: dict[str, dict[str, Any]]
    generated_sha256: str


BENCHMARKS = {
    "cold_table_vector_first_scan": BenchmarkSpec(
        label="cold table-vector first scan",
        statement="cold_table_scan(view_type, main_buffer)",
        number=500,
        warmup_number=50,
        unit="us/scan",
        scale=1e6,
        note=(
            "Creates a root view, loads the vector, and materializes all "
            f"{TABLE_SCAN_COUNT} table child views."
        ),
    ),
    "cold_struct_vector_first_scan": BenchmarkSpec(
        label="cold struct-vector first scan",
        statement="cold_struct_scan(view_type, main_buffer)",
        number=500,
        warmup_number=50,
        unit="us/scan",
        scale=1e6,
        note=(
            "Creates a root view, loads the vector, and materializes all "
            f"{STRUCT_SCAN_COUNT} inline struct views."
        ),
    ),
    "fully_cached_indexed_scan": BenchmarkSpec(
        label="fully cached indexed scan",
        statement="indexed_scan(cached_tables)",
        number=5_000,
        warmup_number=500,
        unit="us/scan",
        scale=1e6,
        note=f"Reads all {TABLE_SCAN_COUNT} cached table views by integer index.",
    ),
    "fully_cached_slice": BenchmarkSpec(
        label="fully cached full slice",
        statement="slice_tables[:]",
        number=50_000,
        warmup_number=5_000,
        unit="us/slice",
        scale=1e6,
        note=f"Returns a tuple containing all {TABLE_SCAN_COUNT} cached views.",
    ),
    "cold_to_model": BenchmarkSpec(
        label="new view to_model",
        statement="cold_to_model(view_type, main_buffer)",
        number=100,
        warmup_number=10,
        unit="us/call",
        scale=1e6,
        note="Creates a new root view and materializes the entire object graph.",
    ),
    "warm_to_model": BenchmarkSpec(
        label="reused view to_model",
        statement="reused_view.to_model()",
        number=300,
        warmup_number=30,
        unit="us/call",
        scale=1e6,
        note="Converts the same existing view without constructing another root view.",
    ),
    "small_default_build": BenchmarkSpec(
        label="small default model build",
        statement="default_model.to_flatbuffer()",
        number=5_000,
        warmup_number=500,
        unit="us/build",
        scale=1e6,
        note="Builds a Monster whose fields all use generated defaults.",
    ),
    "large_byte_vector_build": BenchmarkSpec(
        label="large byte-vector build",
        statement="byte_model.to_flatbuffer()",
        number=100,
        warmup_number=10,
        unit="us/build",
        scale=1e6,
        note=f"Builds only a {LARGE_BYTE_COUNT}-byte inventory vector.",
    ),
    "large_numeric_vector_build": BenchmarkSpec(
        label="large numeric-vector build",
        statement="numeric_model.to_flatbuffer()",
        number=100,
        warmup_number=10,
        unit="us/build",
        scale=1e6,
        note=f"Builds only a {LARGE_NUMERIC_COUNT}-element float32 vector.",
    ),
    "large_struct_vector_build": BenchmarkSpec(
        label="large struct-vector build",
        statement="struct_model.to_flatbuffer()",
        number=10,
        warmup_number=1,
        unit="us/build",
        scale=1e6,
        note=f"Builds only a {LARGE_STRUCT_COUNT}-element Vec3 vector.",
    ),
    "large_table_vector_build": BenchmarkSpec(
        label="large table-vector build",
        statement="table_model.to_flatbuffer()",
        number=25,
        warmup_number=3,
        unit="us/build",
        scale=1e6,
        note=f"Builds only a {LARGE_TABLE_COUNT}-element Weapon vector.",
    ),
    "large_string_vector_build": BenchmarkSpec(
        label="large string-vector build",
        statement="string_model.to_flatbuffer()",
        number=50,
        warmup_number=5,
        unit="us/build",
        scale=1e6,
        note=f"Builds only a {LARGE_STRING_COUNT}-element string vector.",
    ),
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _flatc_version() -> str:
    executable = shutil.which("flatc")
    if executable is None:
        return "not-found"
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def _cold_table_scan(view_type: type[Any], buffer: Buffer) -> tuple[Any, ...]:
    vector = view_type.from_buffer(buffer).weapons
    if vector is None:
        raise AssertionError("benchmark table vector is absent")
    return tuple(vector)


def _cold_struct_scan(view_type: type[Any], buffer: Buffer) -> tuple[Any, ...]:
    vector = view_type.from_buffer(buffer).path
    if vector is None:
        raise AssertionError("benchmark struct vector is absent")
    return tuple(vector)


def _indexed_scan(vector: Sequence[Any]) -> Any:
    last = None
    for index in range(len(vector)):
        last = vector[index]
    return last


def _same_identities(first: Sequence[Any], second: Sequence[Any]) -> bool:
    return all(left is right for left, right in zip(first, second))


def _cold_to_model(view_type: type[Any], buffer: Buffer) -> Any:
    return view_type.from_buffer(buffer).to_model()


def _fully_scanned_tables(
    generated: ModuleType,
    buffer: Buffer,
) -> tuple[Any, Any]:
    """Return a root and dense table vector that have never been sliced."""

    view = generated.MonsterView.from_buffer(buffer)
    tables = view.weapons
    if tables is None:
        raise AssertionError("benchmark table vector is absent")
    first_scan = tuple(tables)
    second_scan = tuple(tables)
    if not _same_identities(first_scan, second_scan):
        raise AssertionError("table-vector identities are not strongly cached")
    if tables.cached_count != len(tables):
        raise AssertionError("full scan did not populate every table cache entry")
    return view, tables


def _make_main_model(generated: ModuleType) -> Any:
    weapons = [
        generated.Weapon(
            name=f"scan-weapon-{index:04d}",
            damage=(index * 17) % 30_000,
        )
        for index in range(TABLE_SCAN_COUNT)
    ]
    scores = np.array(
        [
            (index - MODEL_SCORE_COUNT // 2) / 8.0
            for index in range(MODEL_SCORE_COUNT)
        ],
        dtype=np.float32,
    )
    tags = [f"model-tag-{index:04d}" for index in range(MODEL_TAG_COUNT)]
    path = [
        generated.Vec3(
            x=float(index),
            y=float(index) + 0.25,
            z=-float(index),
        )
        for index in range(STRUCT_SCAN_COUNT)
    ]
    inventory = (bytes(range(256)) * (MODEL_INVENTORY_BYTES // 256 + 1))[
        :MODEL_INVENTORY_BYTES
    ]
    return generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        mana=175,
        hp=80,
        name="round-two-benchmark",
        inventory=inventory,
        color=generated.Color.Green,
        weapons=weapons,
        scores=scores,
        tags=tags,
        path=path,
        optional_energy=123,
    )


def _make_build_models(generated: ModuleType) -> dict[str, Any]:
    byte_values = (bytes(range(256)) * (LARGE_BYTE_COUNT // 256 + 1))[
        :LARGE_BYTE_COUNT
    ]
    numeric_values = np.array(
        [
            (index - LARGE_NUMERIC_COUNT // 2) / 16.0
            for index in range(LARGE_NUMERIC_COUNT)
        ],
        dtype=np.float32,
    )
    struct_values = [
        generated.Vec3(
            x=float(index),
            y=float(index) + 0.5,
            z=-float(index),
        )
        for index in range(LARGE_STRUCT_COUNT)
    ]
    table_values = [
        generated.Weapon(
            name=f"build-weapon-{index:04d}",
            damage=(index * 31) % 30_000,
        )
        for index in range(LARGE_TABLE_COUNT)
    ]
    string_values = [
        f"build-string-{index:04d}" for index in range(LARGE_STRING_COUNT)
    ]
    return {
        "default": generated.Monster(),
        "byte": generated.Monster(inventory=byte_values),
        "numeric": generated.Monster(scores=numeric_values),
        "struct": generated.Monster(path=struct_values),
        "table": generated.Monster(weapons=table_values),
        "string": generated.Monster(tags=string_values),
    }


def _assert_main_semantics(
    generated: ModuleType,
    model: Any,
    buffer: Buffer,
) -> Any:
    view = generated.MonsterView.from_buffer(buffer)
    if not view.buffer.readonly:
        raise AssertionError("root view exposes a writable buffer")
    inventory = view.inventory
    if inventory is None or not inventory.readonly:
        raise AssertionError("byte vector is absent or writable")
    scores = view.scores
    if scores is None or scores.flags.writeable:
        raise AssertionError("numeric vector is absent or writable")

    tables = view.weapons
    structs = view.path
    if tables is None or structs is None:
        raise AssertionError("benchmark cached vector is absent")
    first_tables = tuple(tables)
    second_tables = tuple(tables)
    first_structs = tuple(structs)
    second_structs = tuple(structs)
    if not _same_identities(first_tables, second_tables):
        raise AssertionError("table-vector identities are not strongly cached")
    if not _same_identities(first_structs, second_structs):
        raise AssertionError("struct-vector identities are not strongly cached")
    if tables.cached_count != len(tables) or structs.cached_count != len(structs):
        raise AssertionError("full scans did not populate every vector cache entry")

    reused_model = view.to_model()
    if reused_model != model or view.to_model() != model:
        raise AssertionError("reused view failed its semantic model conversion")
    if _cold_to_model(generated.MonsterView, buffer) != model:
        raise AssertionError("new view failed its semantic model conversion")
    return view


def _assert_build_semantics(
    generated: ModuleType,
    models: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    buffers: dict[str, memoryview] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        buffer = model.to_flatbuffer()
        if model.to_flatbuffer() != buffer:
            raise AssertionError(f"{name} builder is not byte deterministic")
        buffers[name] = buffer
        metadata[name] = {
            "bytes": len(buffer),
            "sha256": hashlib.sha256(buffer).hexdigest(),
        }

    default_view = generated.MonsterView.from_buffer(buffers["default"])
    if default_view.to_model() != models["default"]:
        raise AssertionError("default build failed semantic validation")

    byte_view = generated.MonsterView.from_buffer(buffers["byte"])
    byte_values = byte_view.inventory
    if byte_values is None or not byte_values.readonly:
        raise AssertionError("built byte vector is absent or writable")
    if (
        len(byte_values) != LARGE_BYTE_COUNT
        or bytes(byte_values[:4]) != b"\x00\x01\x02\x03"
    ):
        raise AssertionError("built byte vector contents are incorrect")

    numeric_view = generated.MonsterView.from_buffer(buffers["numeric"])
    numeric_values = numeric_view.scores
    if numeric_values is None or numeric_values.flags.writeable:
        raise AssertionError("built numeric vector is absent or writable")
    if len(numeric_values) != LARGE_NUMERIC_COUNT:
        raise AssertionError("built numeric vector length is incorrect")
    if (
        float(numeric_values[0]) != -2048.0
        or float(numeric_values[-1]) != 2047.9375
    ):
        raise AssertionError("built numeric vector contents are incorrect")

    struct_view = generated.MonsterView.from_buffer(buffers["struct"])
    struct_values = struct_view.path
    if struct_values is None or len(struct_values) != LARGE_STRUCT_COUNT:
        raise AssertionError("built struct vector length is incorrect")
    if struct_values[0].x != 0.0 or struct_values[-1].z != -4095.0:
        raise AssertionError("built struct vector contents are incorrect")
    if struct_values[-1] is not struct_values[-1]:
        raise AssertionError("built struct vector does not cache child identity")

    table_view = generated.MonsterView.from_buffer(buffers["table"])
    table_values = table_view.weapons
    if table_values is None or len(table_values) != LARGE_TABLE_COUNT:
        raise AssertionError("built table vector length is incorrect")
    if table_values[0].name != "build-weapon-0000":
        raise AssertionError("built table vector first element is incorrect")
    if table_values[-1].name != "build-weapon-1023":
        raise AssertionError("built table vector last element is incorrect")
    if table_values[-1] is not table_values[-1]:
        raise AssertionError("built table vector does not cache child identity")

    string_view = generated.MonsterView.from_buffer(buffers["string"])
    string_values = string_view.tags
    if string_values is None or len(string_values) != LARGE_STRING_COUNT:
        raise AssertionError("built string vector length is incorrect")
    if string_values[0] != "build-string-0000":
        raise AssertionError("built string vector first element is incorrect")
    if string_values[-1] != "build-string-1023":
        raise AssertionError("built string vector last element is incorrect")
    if string_values[-1] is not string_values[-1]:
        raise AssertionError("built string vector does not cache decoded strings")

    return metadata


def _prepare_fixture() -> PreparedFixture:
    from msgspec_flatbuffers import generate

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="msgspec-flatbuffers-round2-"
    )
    generated_root = Path(temporary_directory.name) / "generated"
    module_path = generate(FIXTURE, generated_root)
    generated_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    generated = _load_module("_msgspec_flatbuffers_round2_fixture", module_path)

    main_model = _make_main_model(generated)
    main_buffer = main_model.to_flatbuffer()
    reused_view = _assert_main_semantics(generated, main_model, main_buffer)
    build_models = _make_build_models(generated)
    build_metadata = _assert_build_semantics(generated, build_models)
    return PreparedFixture(
        temporary_directory=temporary_directory,
        generated=generated,
        main_buffer=main_buffer,
        reused_view=reused_view,
        build_models=build_models,
        build_metadata=build_metadata,
        generated_sha256=generated_sha256,
    )


def _run_one(spec: BenchmarkSpec, timer_globals: dict[str, Any]) -> dict[str, Any]:
    timer = timeit.Timer(
        spec.statement,
        globals=timer_globals,
        timer=time.process_time,
    )
    warmup_seconds = timer.timeit(number=spec.warmup_number)
    raw_seconds = timer.repeat(repeat=REPEAT, number=spec.number)
    samples = [seconds / spec.number * spec.scale for seconds in raw_seconds]
    return {
        "label": spec.label,
        "unit": spec.unit,
        "median": median(samples),
        "samples": samples,
        "raw_seconds": raw_seconds,
        "iterations_per_sample": spec.number,
        "warmup_iterations": spec.warmup_number,
        "warmup_seconds": warmup_seconds,
        "note": spec.note,
    }


def _environment() -> dict[str, Any]:
    import msgspec_flatbuffers

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "flatc": _flatc_version(),
        "msgspec_flatbuffers_source": str(
            Path(msgspec_flatbuffers.__file__).resolve()
        ),
        "packages": {
            "flatbuffers": _package_version("flatbuffers"),
            "msgspec": _package_version("msgspec"),
            "msgspec-flatbuffers": _package_version("msgspec-flatbuffers"),
            "numpy": _package_version("numpy"),
        },
    }


def _run_benchmarks() -> dict[str, Any]:
    fixture = _prepare_fixture()
    try:
        indexed_view, cached_tables = _fully_scanned_tables(
            fixture.generated, fixture.main_buffer
        )
        slice_view, slice_tables = _fully_scanned_tables(
            fixture.generated, fixture.main_buffer
        )
        if cached_tables is slice_tables or indexed_view is slice_view:
            raise AssertionError("indexed and slice workloads unexpectedly share state")
        build_models = fixture.build_models
        timer_globals = {
            "byte_model": build_models["byte"],
            "cached_tables": cached_tables,
            "cold_struct_scan": _cold_struct_scan,
            "cold_table_scan": _cold_table_scan,
            "cold_to_model": _cold_to_model,
            "default_model": build_models["default"],
            "indexed_scan": _indexed_scan,
            "main_buffer": fixture.main_buffer,
            "numeric_model": build_models["numeric"],
            "slice_tables": slice_tables,
            "string_model": build_models["string"],
            "struct_model": build_models["struct"],
            "table_model": build_models["table"],
            "view_type": fixture.generated.MonsterView,
            "reused_view": fixture.reused_view,
        }
        results = {
            name: _run_one(spec, timer_globals)
            for name, spec in BENCHMARKS.items()
        }
        sliced = slice_tables[:]
        if len(sliced) != len(slice_tables) or not all(
            item is slice_tables[index] for index, item in enumerate(sliced)
        ):
            raise AssertionError("cached slice did not preserve child identities")
    finally:
        fixture.temporary_directory.cleanup()

    return {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "fixture": {
            "schema": str(FIXTURE.relative_to(REPOSITORY_ROOT)),
            "schema_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "generated_module_sha256": fixture.generated_sha256,
            "main_buffer_bytes": len(fixture.main_buffer),
            "main_model": {
                "inventory_bytes": MODEL_INVENTORY_BYTES,
                "score_count": MODEL_SCORE_COUNT,
                "struct_count": STRUCT_SCAN_COUNT,
                "table_count": TABLE_SCAN_COUNT,
                "tag_count": MODEL_TAG_COUNT,
            },
            "build_models": fixture.build_metadata,
        },
        "timing": {
            "clock": "process_time",
            "repeat": REPEAT,
            "fixed_iteration_counts": True,
            "cyclic_gc_during_samples": "disabled by timeit",
        },
        "results": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    environment = report["environment"]
    fixture = report["fixture"]
    print(
        "msgspec-flatbuffers round-two benchmark "
        f"(Python {environment['python']}, {environment['flatc']})"
    )
    print(
        f"main fixture: {fixture['main_buffer_bytes']:,} bytes, "
        f"{fixture['main_model']['table_count']} tables, "
        f"{fixture['main_model']['struct_count']} structs"
    )
    print()
    label_width = max(len(result["label"]) for result in report["results"].values())
    for result in report["results"].values():
        print(
            f"{result['label']:<{label_width}}  "
            f"{result['median']:>11.3f} {result['unit']}  "
            f"(median of {len(result['samples'])})"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write environment metadata and every timing sample to PATH",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _run_benchmarks()
    _print_summary(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
