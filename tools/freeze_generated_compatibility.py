"""Freeze or validate generated-code compatibility for a minor release line."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tests" / "generated_compatibility" / "core.fbs"
RELEASES = ROOT / "tests" / "generated_compatibility" / "releases"
VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
GENERATED_VERSION_PATTERN = re.compile(
    r"^__msgspec_serde_generated_version__ = "
    r"['\"]((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))['\"]$",
    re.MULTILINE,
)


class SnapshotError(RuntimeError):
    """Raised when a minor-line snapshot is missing or invalid."""


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
    major, minor, _ = version.split(".")
    return RELEASES / f"v{major}_{minor}"


def _freeze(version: str, flatc: str) -> Path:
    if version.rsplit(".", 1)[1] != "0":
        raise SnapshotError(
            "compatibility snapshots are only created for the first release "
            f"in a minor line: {version}"
        )
    target = _snapshot_path(version)
    if target.exists():
        raise SnapshotError(f"minor-line snapshot already exists: {target}")

    # Release CI runs --check before installing the package.
    import msgspec_serde
    from msgspec_serde import generate

    if msgspec_serde.__version__ != version:
        raise SnapshotError(
            "installed msgspec-serde version does not match the release: "
            f"installed={msgspec_serde.__version__}, release={version}; "
            "run uv sync"
        )

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
        stage.rename(target)
    return target


def _check(version: str) -> Path:
    snapshot = _snapshot_path(version)
    generated_source = snapshot / "core.py"
    if not generated_source.is_file():
        raise SnapshotError(f"release snapshot is missing: {snapshot}")
    source = generated_source.read_text(encoding="utf-8")
    marker = GENERATED_VERSION_PATTERN.search(source)
    if marker is None:
        raise SnapshotError(
            f"generated version marker is missing from {generated_source}"
        )
    generated_version = marker.group(1)
    if _snapshot_path(generated_version) != snapshot:
        raise SnapshotError(
            f"generated version {generated_version} does not match "
            f"minor-line snapshot {snapshot.name}"
        )
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="freeze generated code for a minor-line compatibility test"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed snapshot for the configured minor release line",
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
    except (KeyError, OSError, SnapshotError) as error:
        parser.error(str(error))
    action = "validated" if args.check else "created"
    print(f"{action} {snapshot.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
