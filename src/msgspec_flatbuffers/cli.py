"""Command-line interface for ``msgspec-flatbuffers``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .compiler import FlatcError
from .generator import GenerationError, generate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msgspec-flatbuffers")
    subcommands = parser.add_subparsers(required=True)

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
    generate_parser.add_argument(
        "--gen-onefile",
        action="store_true",
        help="place all definitions from each schema in one Python module",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)

    try:
        for schema in args.schemas:
            path = generate(
                schema,
                args.output,
                include_dirs=args.include_dirs,
                flatc=args.flatc,
                project_root=args.project_root,
                package=args.package,
                gen_onefile=args.gen_onefile,
            )
            print(path)
    except (FlatcError, GenerationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
