from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TUTORIAL = ROOT / "examples" / "tutorial"
SCHEMAS = TUTORIAL / "schemas"
SCHEMA_PATHS = tuple(sorted(SCHEMAS.glob("*.fbs")))
HAS_FLATC = shutil.which("flatc") is not None


def test_tutorial_uses_the_checked_in_schemas() -> None:
    tutorial = (ROOT / "tutorial.md").read_text(encoding="utf-8")
    for schema in SCHEMA_PATHS:
        assert f"```fbs\n{schema.read_text(encoding='utf-8')}```" in tutorial


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_tutorial_runs_from_generated_modules(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "msgspec_serde.cli",
            "generate",
            *(str(schema) for schema in SCHEMA_PATHS),
            "-I",
            str(SCHEMAS),
            "-o",
            str(tmp_path),
            "--project-root",
            str(SCHEMAS),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((str(tmp_path), existing_pythonpath))
        if existing_pythonpath
        else str(tmp_path)
    )
    completed = subprocess.run(
        [sys.executable, str(TUTORIAL / "demo.py")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.splitlines()[:3] == [
        "Orc Axe",
        "Cat nightly snapshot",
        "Tutorial.Metric",
    ]
