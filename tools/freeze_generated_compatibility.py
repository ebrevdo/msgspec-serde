"""Freeze or validate generated-code compatibility fixtures for a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tests" / "generated_compatibility" / "core.fbs"
RELEASES = ROOT / "tests" / "generated_compatibility" / "releases"
METADATA_NAME = "metadata.json"
VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


class SnapshotError(RuntimeError):
    """Raised when a release snapshot is missing or invalid."""


def _configured_version(path: Path, section: str) -> object:
    with path.open("rb") as file:
        return tomllib.load(file)[section]["version"]


def _release_version() -> str:
    python_version = _configured_version(ROOT / "pyproject.toml", "project")
    rust_version = _configured_version(ROOT / "rust" / "Cargo.toml", "package")
    if python_version != rust_version:
        raise SnapshotError(
            f"version mismatch: pyproject.toml={python_version}, "
            f"rust/Cargo.toml={rust_version}"
        )
    if (
        not isinstance(python_version, str)
        or VERSION_PATTERN.fullmatch(python_version) is None
    ):
        raise SnapshotError(f"release version is not SemVer: {python_version!r}")
    return python_version


def _snapshot_path(version: str) -> Path:
    return RELEASES / f"v{version.replace('.', '_')}"


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _generated_files(snapshot: Path) -> dict[str, str]:
    generated: dict[str, str] = {}
    for path in sorted(snapshot.rglob("*")):
        relative = path.relative_to(snapshot)
        if (
            not path.is_file()
            or relative.name == METADATA_NAME
            or "__pycache__" in relative.parts
        ):
            continue
        generated[relative.as_posix()] = _sha256(path)
    return generated


def _metadata(version: str, snapshot: Path, flatc_version: str) -> dict[str, object]:
    return {
        "flatc_version": flatc_version,
        "generated_files": _generated_files(snapshot),
        "msgspec_flatbuffers_version": version,
        "schema": SCHEMA.relative_to(ROOT).as_posix(),
        "schema_sha256": _sha256(SCHEMA),
    }


def _freeze(version: str, flatc: str) -> Path:
    target = _snapshot_path(version)
    if target.exists():
        raise SnapshotError(f"release snapshot already exists: {target}")

    # Release CI runs --check before installing the package.
    import msgspec_flatbuffers
    from msgspec_flatbuffers import generate

    if msgspec_flatbuffers.__version__ != version:
        raise SnapshotError(
            "installed msgspec-flatbuffers version does not match the release: "
            f"installed={msgspec_flatbuffers.__version__}, release={version}; "
            "run uv sync"
        )

    flatc_version = subprocess.run(
        [flatc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    RELEASES.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".snapshot-", dir=RELEASES) as temporary:
        stage = Path(temporary) / "snapshot"
        generate(
            SCHEMA,
            stage,
            flatc=flatc,
            project_root=ROOT,
            gen_onefile=True,
        )
        metadata = _metadata(version, stage, flatc_version)
        (stage / METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.rename(target)
    return target


def _check(version: str) -> Path:
    snapshot = _snapshot_path(version)
    metadata_path = snapshot / METADATA_NAME
    if not metadata_path.is_file():
        raise SnapshotError(f"release snapshot is missing: {snapshot}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "msgspec_flatbuffers_version": version,
        "schema": SCHEMA.relative_to(ROOT).as_posix(),
        "schema_sha256": _sha256(SCHEMA),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SnapshotError(
                f"invalid {key} in {metadata_path}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    flatc_version = metadata.get("flatc_version")
    if not isinstance(flatc_version, str) or not flatc_version:
        raise SnapshotError(f"flatc_version is missing from {metadata_path}")
    actual_files = _generated_files(snapshot)
    if metadata.get("generated_files") != actual_files:
        raise SnapshotError(f"generated file hashes do not match {metadata_path}")
    generated_source = snapshot / "core.py"
    version_marker = f"__msgspec_flatbuffers_generated_version__ = {version!r}"
    if version_marker not in generated_source.read_text(encoding="utf-8"):
        raise SnapshotError(
            f"generated version marker is missing from {generated_source}"
        )
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze generated code for a release compatibility test"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed snapshot without changing files",
    )
    parser.add_argument(
        "--flatc",
        default="flatc",
        help="FlatBuffers compiler used when creating a snapshot",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        version = _release_version()
        snapshot = _check(version) if args.check else _freeze(version, args.flatc)
    except (KeyError, OSError, SnapshotError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    action = "validated" if args.check else "created"
    print(f"{action} {snapshot.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
