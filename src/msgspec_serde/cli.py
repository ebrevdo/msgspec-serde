"""Command-line interface for ``msgspec_flatc``.

Example:
    Generate Python modules from a shell:

    .. code-block:: console

        $ msgspec_flatc generate monster.fbs -o generated
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .compiler import FlatcError
from .generator import GenerationError, generate

_DESCRIPTION = """Generate msgspec models and lazy FlatBuffers views from .fbs files.

The generated models work with msgspec_serde's JSON, MessagePack, and
FlatBuffer codecs. Generating code requires the FlatBuffers compiler, flatc.
"""

_EPILOG = """examples:
  Generate code from one schema:
    msgspec_flatc generate schemas/monster.fbs -o generated

  Show all generation options:
    msgspec_flatc generate --help
"""

_GENERATE_DESCRIPTION = """Compile one or more .fbs schemas and generate Python modules.

By default, each definition is written to its own module under the schema's
FlatBuffers namespace. Existing files are replaced only when they were created
by msgspec_flatc.
"""

_GENERATE_EPILOG = """examples:
  Generate one module per definition:
    msgspec_flatc generate schemas/monster.fbs -o generated \\
      --project-root schemas

  Add an include directory and Python package prefix:
    msgspec_flatc generate schemas/api.fbs -o src \\
      --project-root schemas -I schemas/common --package my_app.generated

  Generate one module per input schema:
    msgspec_flatc generate schemas/monster.fbs -o generated \\
      --project-root schemas --gen-onefile
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgspec_flatc",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subcommands = parser.add_subparsers(
        title="commands",
        required=True,
    )

    generate_parser = subcommands.add_parser(
        "generate",
        help="generate msgspec models and lazy views from FlatBuffers IDL",
        description=_GENERATE_DESCRIPTION,
        epilog=_GENERATE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate_parser.add_argument(
        "schemas",
        nargs="+",
        type=Path,
        metavar="SCHEMA",
        help="one or more .fbs schema files",
    )
    generate_parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="directory for generated Python packages",
    )
    generate_parser.add_argument(
        "-I",
        "--include",
        action="append",
        default=[],
        type=Path,
        dest="include_dirs",
        metavar="DIR",
        help="add a flatc include directory; may be repeated",
    )
    generate_parser.add_argument(
        "--flatc",
        default="flatc",
        metavar="PATH",
        help="flatc executable name or path (default: flatc)",
    )
    generate_parser.add_argument(
        "--project-root",
        type=Path,
        metavar="DIR",
        help="root for schema-relative paths (default: each schema's directory)",
    )
    generate_parser.add_argument(
        "--package",
        metavar="NAME",
        help="Python package prefix before the FlatBuffers namespace",
    )
    generate_parser.add_argument(
        "--gen-onefile",
        action="store_true",
        help="generate one Python module per input schema",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``msgspec_flatc`` command-line interface.

    Args:
        argv: Command arguments without the executable name. ``sys.argv`` is
            parsed when omitted.

    Returns:
        Zero after all requested schemas are generated successfully.

    Example:
        Generate Python modules programmatically through the CLI entry point:

        >>> main(["generate", "monster.fbs", "-o", "generated"])
        generated/example/monster.py
        0
    """

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
