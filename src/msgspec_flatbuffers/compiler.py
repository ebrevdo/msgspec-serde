"""Invoke ``flatc`` and load its binary reflection schema."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from ._schema_reader import InvalidSchemaError, parse_bfbs
from .schema import Schema

type StrPath = str | os.PathLike[str]


class FlatcNotFoundError(FileNotFoundError):
    """Raised when the requested ``flatc`` executable is unavailable."""


class FlatcError(RuntimeError):
    """Raised when ``flatc`` rejects a schema or fails to create output."""

    def __init__(
        self,
        command: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = stderr.strip() or stdout.strip() or "flatc failed without output"
        if returncode == 0:
            message = f"flatc output error: {detail}"
        else:
            message = f"flatc exited with status {returncode}: {detail}"
        super().__init__(message)


def _resolve_flatc(flatc: StrPath) -> str:
    requested = os.fspath(flatc)
    executable = shutil.which(requested)
    if executable is None:
        raise FlatcNotFoundError(f"flatc executable not found: {requested!r}")
    return executable


def compile_schema_to_bfbs(
    schema_path: StrPath,
    *,
    include_dirs: Iterable[StrPath] = (),
    flatc: StrPath = "flatc",
    project_root: StrPath | None = None,
) -> bytes:
    """Compile one FlatBuffers IDL file into its ``.bfbs`` representation."""

    source = Path(schema_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    root = source.parent if project_root is None else Path(project_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    includes = [source.parent]
    for value in include_dirs:
        include = Path(value).resolve()
        if not include.is_dir():
            raise NotADirectoryError(include)
        if include not in includes:
            includes.append(include)

    executable = _resolve_flatc(flatc)
    with tempfile.TemporaryDirectory(prefix="msgspec-flatbuffers-") as output:
        output_path = Path(output)
        command = [
            executable,
            "-b",
            "--schema",
            "--bfbs-comments",
            "--bfbs-builtins",
            "--bfbs-filenames",
            str(root),
            "-o",
            output,
        ]
        for include in includes:
            command.extend(("-I", str(include)))
        command.append(source.name)

        result = subprocess.run(
            command,
            cwd=source.parent,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        command_tuple = tuple(command)
        if result.returncode != 0:
            raise FlatcError(
                command_tuple,
                result.returncode,
                result.stdout,
                result.stderr,
            )

        expected = output_path / f"{source.stem}.bfbs"
        if expected.is_file():
            return expected.read_bytes()

        artifacts = list(output_path.glob("*.bfbs"))
        if len(artifacts) == 1:
            return artifacts[0].read_bytes()
        raise FlatcError(
            command_tuple,
            result.returncode,
            result.stdout,
            result.stderr or "flatc completed without creating a .bfbs file",
        )


def compile_schema(
    schema_path: StrPath,
    *,
    include_dirs: Iterable[StrPath] = (),
    flatc: StrPath = "flatc",
    project_root: StrPath | None = None,
) -> Schema:
    """Compile FlatBuffers IDL and return its normalized reflected schema."""

    data = compile_schema_to_bfbs(
        schema_path,
        include_dirs=include_dirs,
        flatc=flatc,
        project_root=project_root,
    )
    return parse_bfbs(data)


def load_bfbs(path: StrPath) -> Schema:
    """Load a previously compiled ``.bfbs`` schema."""

    return parse_bfbs(Path(path).read_bytes())


__all__ = [
    "FlatcError",
    "FlatcNotFoundError",
    "InvalidSchemaError",
    "compile_schema",
    "compile_schema_to_bfbs",
    "load_bfbs",
    "parse_bfbs",
]
