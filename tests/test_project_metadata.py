from __future__ import annotations

import tomllib
from pathlib import Path


def test_license_metadata_includes_project_and_vendored_licenses() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == [
        "LICENSE",
        "src/msgspec_flatbuffers/_reflection/LICENSE.txt",
    ]
    assert all((root / path).is_file() for path in project["license-files"])
