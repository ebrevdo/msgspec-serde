"""Command-line interface for ``msgspec-flatbuffers``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .compiler import FlatcError
from .generator import GenerationError, generate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msgspec-flatbuffers")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate_parser = subcommands.add_parser(
        "generate",
        help="generate msgspec models and lazy views from FlatBuffers IDL",
    )
    generate_parser.add_argument("schemas", nargs="+", type=Path)
    generate_parser.add_argument("-o", "--output", required=True, type=Path)
    generate_parser.add_argument(
        "-I",
        "--include",
        action="append",
        default=[],
        type=Path,
        dest="include_dirs",
    )
    generate_parser.add_argument("--flatc", default="flatc")
    generate_parser.add_argument("--project-root", type=Path)
    generate_parser.add_argument(
        "--package",
        help="optional Python package prefix before the FlatBuffers namespace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "generate":
        parser.error(f"unknown command: {args.command}")

    try:
        for schema in args.schemas:
            path = generate(
                schema,
                args.output,
                include_dirs=args.include_dirs,
                flatc=args.flatc,
                project_root=args.project_root,
                package=args.package,
            )
            print(path)
    except (FlatcError, GenerationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
