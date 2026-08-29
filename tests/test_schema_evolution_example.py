from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from msgspec_flatbuffers import compile_schema, flatbuffer, generate, load_bfbs

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "schema_evolution"
SCHEMAS = EXAMPLE / "schemas"
FIXTURES = EXAMPLE / "fixtures"
GENERATED = EXAMPLE / "schema_evolution_generated" / "example" / "evolution"
HAS_FLATC = shutil.which("flatc") is not None


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def versioned_modules() -> tuple[ModuleType, ModuleType]:
    v1 = _load_module("schema_evolution_v1", GENERATED / "reading_v1.py")
    v2 = _load_module("schema_evolution_v2", GENERATED / "reading_v2.py")
    return v1, v2


def test_new_reader_reads_frozen_old_buffer(
    versioned_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, v2 = versioned_modules

    reading = flatbuffer.decode(
        (FIXTURES / "reading_v1.bin").read_bytes(), type=v2.ReadingView
    )

    assert reading.id == 7
    assert reading.label == "legacy sensor"
    np.testing.assert_array_equal(reading.samples, [1.25, 2.5])
    assert reading.quality == 100
    assert reading.note is None


def test_old_reader_reads_frozen_new_buffer(
    versioned_modules: tuple[ModuleType, ModuleType],
) -> None:
    v1, _ = versioned_modules

    reading = flatbuffer.decode(
        (FIXTURES / "reading_v2.bin").read_bytes(), type=v1.ReadingView
    )

    assert reading.id == 9
    assert reading.label == "current sensor"
    np.testing.assert_array_equal(reading.samples, [3.5])


def test_generated_models_interoperate_across_schema_versions(
    versioned_modules: tuple[ModuleType, ModuleType],
) -> None:
    v1, v2 = versioned_modules

    old_buffer = flatbuffer.encode(
        v1.Reading(
            id=17,
            label="old writer",
            samples=np.array([1.0, 2.0], dtype=np.float32),
        )
    )
    upgraded = flatbuffer.decode(old_buffer, type=v2.ReadingView)
    assert upgraded.id == 17
    assert upgraded.quality == 100
    assert upgraded.note is None

    new_buffer = flatbuffer.encode(
        v2.Reading(
            id=23,
            label="new writer",
            samples=np.array([4.0], dtype=np.float32),
            quality=91,
            note="ignored by v1",
        )
    )
    legacy = flatbuffer.decode(new_buffer, type=v1.ReadingView)
    assert legacy.id == 23
    assert legacy.label == "new writer"
    np.testing.assert_array_equal(legacy.samples, [4.0])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_frozen_binary_schema_remains_readable(version: str) -> None:
    schema = load_bfbs(FIXTURES / f"reading_{version}.bfbs")

    assert schema.root_table == "Example.Evolution.Reading"
    assert schema.file_identifier == "EVOL"


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_installed_flatc_can_drive_current_generator(
    version: str,
    tmp_path: Path,
    versioned_modules: tuple[ModuleType, ModuleType],
) -> None:
    source = SCHEMAS / f"reading_{version}.fbs"
    v1, v2 = versioned_modules

    schema = compile_schema(source, project_root=EXAMPLE)
    module_path = generate(
        source,
        tmp_path,
        project_root=EXAMPLE,
        package="current_flatc_generated",
    )

    assert schema.root_table == "Example.Evolution.Reading"
    assert schema.file_identifier == "EVOL"

    generated = _load_module(f"current_flatc_{version}", module_path)
    frozen = flatbuffer.decode(
        (FIXTURES / f"reading_{version}.bin").read_bytes(),
        type=generated.ReadingView,
    )
    assert frozen.id == (7 if version == "v1" else 9)

    rebuilt = flatbuffer.encode(
        generated.Reading(
            id=31,
            label=f"generated from {version}",
            samples=np.array([6.25], dtype=np.float32),
        )
    )
    fixed = v1 if version == "v1" else v2
    checked_in = flatbuffer.decode(rebuilt, type=fixed.ReadingView)
    assert checked_in.id == 31
    assert checked_in.label == f"generated from {version}"
    np.testing.assert_array_equal(checked_in.samples, [6.25])
