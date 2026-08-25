"""Reproducible microbenchmarks for generated FlatBuffers hot paths.

Run from the repository root with::

    uv run python benchmarks/benchmark_hotpaths.py --json results.json

The benchmark deliberately uses fixed workloads and iteration counts so two
runs are directly comparable.  ``timeit`` disables cyclic garbage collection
during each timed sample; normal reference-count cleanup still occurs.
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
import timeit
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

REPEAT = 7
WEAPON_COUNT = 256
SCORE_COUNT = 4096
TAG_COUNT = 64
PATH_COUNT = 256
INVENTORY_BYTES = 4096

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "monster.fbs"


class BenchmarkSpec(NamedTuple):
    """One fixed benchmark workload."""

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
    model: Any
    view: Any
    generated_sha256: str


BENCHMARKS = {
    "scalar_access": BenchmarkSpec(
        label="scalar property access",
        statement="view.hp",
        number=250_000,
        warmup_number=50_000,
        unit="ns/access",
        scale=1e9,
        note="Present int16 table field; scalar fields are not cached.",
    ),
    "cached_table_vector_scan": BenchmarkSpec(
        label="cached table-vector scan",
        statement="tuple(weapons)",
        number=2_000,
        warmup_number=200,
        unit="us/scan",
        scale=1e6,
        note=(
            f"Scans {WEAPON_COUNT} already materialized child views and creates "
            "one result tuple per scan."
        ),
    ),
    "cached_table_field_scan": BenchmarkSpec(
        label="cached table-vector field scan",
        statement="sum_damage(weapons)",
        number=1_000,
        warmup_number=100,
        unit="us/scan",
        scale=1e6,
        note=(
            f"Scans {WEAPON_COUNT} cached child views and reads one scalar field "
            "from each child."
        ),
    ),
    "cached_numeric_vector_access": BenchmarkSpec(
        label="cached numeric-vector access",
        statement="view.scores",
        number=250_000,
        warmup_number=50_000,
        unit="ns/access",
        scale=1e9,
        note=f"Returns the same cached, read-only {SCORE_COUNT}-element ndarray.",
    ),
    "uncached_numeric_vector_materialization": BenchmarkSpec(
        label="uncached numeric-vector materialization",
        statement="fresh_scores(view_type, view.buffer, view.table_offset)",
        number=20_000,
        warmup_number=2_000,
        unit="us/access",
        scale=1e6,
        note=(
            f"Constructs a table view and materializes a read-only {SCORE_COUNT}-"
            "element ndarray; includes table metadata validation."
        ),
    ),
    "to_model": BenchmarkSpec(
        label="existing view to_model",
        statement="view.to_model()",
        number=50,
        warmup_number=5,
        unit="us/call",
        scale=1e6,
        note="Materializes a model from the same existing root view.",
    ),
    "to_flatbuffer": BenchmarkSpec(
        label="model to_flatbuffer",
        statement="model.to_flatbuffer()",
        number=25,
        warmup_number=3,
        unit="us/call",
        scale=1e6,
        note="Builds and returns a new read-only FlatBuffer memoryview.",
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


def _sum_damage(items: Sequence[Any]) -> int:
    total = 0
    for item in items:
        total += item.damage
    return total


def _fresh_scores(view_type: type[Any], buffer: Any, table_offset: int) -> Any:
    return view_type(buffer, table_offset).scores


def _make_model(generated: ModuleType) -> Any:
    weapons = [
        generated.Weapon(
            name=f"weapon-{index:04d}",
            damage=(index * 17) % 30_000,
        )
        for index in range(WEAPON_COUNT)
    ]
    scores = np.array(
        [(index - SCORE_COUNT // 2) / 8.0 for index in range(SCORE_COUNT)],
        dtype=np.float32,
    )
    tags = [f"tag-{index:04d}" for index in range(TAG_COUNT)]
    path = [
        generated.Vec3(
            x=float(index),
            y=float(index) + 0.25,
            z=-float(index),
        )
        for index in range(PATH_COUNT)
    ]
    inventory = (bytes(range(256)) * (INVENTORY_BYTES // 256 + 1))[
        :INVENTORY_BYTES
    ]
    return generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        mana=175,
        hp=80,
        name="benchmark-monster",
        inventory=inventory,
        color=generated.Color.Green,
        weapons=weapons,
        scores=scores,
        tags=tags,
        path=path,
        optional_energy=123,
    )


def _prepare_fixture() -> PreparedFixture:
    from msgspec_flatbuffers import generate

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="msgspec-flatbuffers-benchmark-"
    )
    generated_root = Path(temporary_directory.name) / "generated"
    module_path = generate(FIXTURE, generated_root)
    generated_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    generated = _load_module("_msgspec_flatbuffers_benchmark_fixture", module_path)
    model = _make_model(generated)
    buffer = model.to_flatbuffer()
    view = generated.MonsterView.from_buffer(buffer)

    if not view.buffer.readonly:
        raise AssertionError("benchmark view unexpectedly exposes a writable buffer")
    weapons = view.weapons
    if weapons is None or len(weapons) != WEAPON_COUNT:
        raise AssertionError("benchmark table-vector fixture has the wrong size")
    first_scan = tuple(weapons)
    second_scan = tuple(weapons)
    if not all(first is second for first, second in zip(first_scan, second_scan)):
        raise AssertionError("table-vector child identities are not strongly cached")
    scores = view.scores
    if scores is None or len(scores) != SCORE_COUNT:
        raise AssertionError("benchmark numeric-vector fixture has the wrong size")
    if scores.flags.writeable:
        raise AssertionError("benchmark numeric vector is unexpectedly writable")

    if view.to_model() != model:
        raise AssertionError("generated fixture failed its semantic round trip")

    return PreparedFixture(
        temporary_directory=temporary_directory,
        generated=generated,
        model=model,
        view=view,
        generated_sha256=generated_sha256,
    )


def _run_one(
    spec: BenchmarkSpec,
    timer_globals: dict[str, Any],
) -> dict[str, Any]:
    timer = timeit.Timer(spec.statement, globals=timer_globals)
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
        weapons = fixture.view.weapons
        if weapons is None:
            raise AssertionError("benchmark weapons vector is absent")
        timer_globals = {
            "fresh_scores": _fresh_scores,
            "model": fixture.model,
            "sum_damage": _sum_damage,
            "view": fixture.view,
            "view_type": fixture.generated.MonsterView,
            "weapons": weapons,
        }
        results = {
            name: _run_one(spec, timer_globals)
            for name, spec in BENCHMARKS.items()
        }
        buffer_size = len(fixture.view.buffer)
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
            "buffer_bytes": buffer_size,
            "inventory_bytes": INVENTORY_BYTES,
            "weapon_count": WEAPON_COUNT,
            "score_count": SCORE_COUNT,
            "tag_count": TAG_COUNT,
            "path_count": PATH_COUNT,
        },
        "timing": {
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
        "msgspec-flatbuffers hot-path benchmark "
        f"(Python {environment['python']}, {environment['flatc']})"
    )
    print(
        f"fixture: {fixture['buffer_bytes']:,} bytes, "
        f"{fixture['weapon_count']} weapons, {fixture['score_count']} scores"
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
        help="write environment metadata and all timing samples to PATH",
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
