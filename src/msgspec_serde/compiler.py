"""Invoke ``flatc`` and load its binary reflection schema.

Example:
    Compile an IDL file into a normalized schema:

    >>> schema = compile_schema("schemas/monster.fbs")
    >>> schema.root_table
    'example.Monster'
"""

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
    """Report that the requested ``flatc`` executable is unavailable.

    Example:
        Catch a missing compiler separately from schema compilation failures:

        >>> try:
        ...     compile_schema("monster.fbs", flatc="missing-flatc")
        ... except FlatcNotFoundError:
        ...     print("install flatc")
        install flatc
    """


class FlatcError(RuntimeError):
    """Report a failed ``flatc`` invocation.

    Args:
        command: The executed command and its arguments.
        returncode: The process exit status.
        stdout: Text written to standard output.
        stderr: Text written to standard error.

    Attributes:
        command: The executed command and its arguments.
        returncode: The process exit status.
        stdout: Text written to standard output.
        stderr: Text written to standard error.

    Example:
        Inspect compiler diagnostics:

        >>> error = FlatcError(("flatc", "broken.fbs"), 1, "", "syntax error")
        >>> error.returncode
        1
    """

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
    """Compile one FlatBuffers IDL file to binary schema bytes.

    Args:
        schema_path: The ``.fbs`` file to compile.
        include_dirs: Additional directories searched for included schemas.
        flatc: The ``flatc`` executable name or path.
        project_root: The root used for filenames stored in the binary schema.
            The schema's parent directory is used by default.

    Returns:
        The generated ``.bfbs`` data.

    Raises:
        FileNotFoundError: The schema file does not exist.
        NotADirectoryError: The project root or an include path is not a
            directory.
        FlatcNotFoundError: The requested compiler is unavailable.
        FlatcError: The compiler fails or does not create a binary schema.

    Example:
        Compile a schema before storing or parsing it:

        >>> bfbs = compile_schema_to_bfbs("schemas/monster.fbs")
        >>> bfbs[:4] != b""
        True
    """

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
    with tempfile.TemporaryDirectory(prefix="msgspec-serde-") as output:
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
    """Compile FlatBuffers IDL and return its normalized schema.

    Args:
        schema_path: The ``.fbs`` file to compile.
        include_dirs: Additional directories searched for included schemas.
        flatc: The ``flatc`` executable name or path.
        project_root: The root used for filenames stored in the binary schema.
            The schema's parent directory is used by default.

    Returns:
        The normalized reflected schema.

    Raises:
        FileNotFoundError: The schema file does not exist.
        NotADirectoryError: The project root or an include path is not a
            directory.
        FlatcNotFoundError: The requested compiler is unavailable.
        FlatcError: The compiler fails or does not create a binary schema.
        InvalidSchemaError: The generated binary schema is invalid or uses an
            unsupported feature.

    Example:
        Compile a schema and inspect its root table:

        >>> schema = compile_schema("schemas/monster.fbs")
        >>> schema.root_table
        'example.Monster'
    """

    data = compile_schema_to_bfbs(
        schema_path,
        include_dirs=include_dirs,
        flatc=flatc,
        project_root=project_root,
    )
    return parse_bfbs(data)


def load_bfbs(path: StrPath) -> Schema:
    """Load a normalized schema from a compiled ``.bfbs`` file.

    Args:
        path: The binary schema file to read.

    Returns:
        The normalized reflected schema.

    Raises:
        OSError: The file cannot be read.
        InvalidSchemaError: The file is malformed or uses an unsupported
            feature.

    Example:
        Load a binary schema generated by ``flatc --schema``:

        >>> schema = load_bfbs("monster.bfbs")
        >>> schema.root_table
        'example.Monster'
    """

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
