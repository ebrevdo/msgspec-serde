"""Installed package version helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("msgspec-serde")
except PackageNotFoundError:  # pragma: no cover - direct source-tree import
    __version__ = "0.0.0"


def _major_minor(value: str) -> tuple[int, int]:
    """Return the SemVer major and minor components."""

    parts = value.split(".", 2)
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"invalid msgspec-serde version: {value!r}")
    return int(parts[0]), int(parts[1])
