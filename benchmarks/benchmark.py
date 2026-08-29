"""Run cross-library serialization benchmarks with pyperf."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pyperf

from ._suite import (
    PROFILE_NAMES,
    BenchmarkDefinition,
    BenchmarkSuite,
    benchmark_keys,
)


def _unreachable_benchmark() -> object:
    raise RuntimeError("pyperf selected an unavailable benchmark fixture")


def _copy_generated_root(
    command: list[str],
    args: argparse.Namespace,
) -> None:
    if args.generated_root is not None:
        command.extend(("--generated-root", str(args.generated_root)))


def _runner() -> pyperf.Runner:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return pyperf.Runner(
        _argparser=parser,
        add_cmdline_args=_copy_generated_root,
        program_args=("-m", "benchmarks.benchmark"),
        processes=6,
        values=5,
        min_time=0.05,
    )


def main(argv: Sequence[str] | None = None) -> int:
    runner = _runner()
    args = runner.parse_args(argv)
    keys = benchmark_keys()
    if args.worker_task is None:
        selected_profiles = PROFILE_NAMES
    else:
        key = keys[args.worker_task]
        selected_profiles = (key.profile,)
    suite = BenchmarkSuite(
        selected_profiles,
        generated_root=args.generated_root,
    )
    if args.generated_root is None:
        args.generated_root = suite.generated_root
    runner.metadata.update(suite.metadata)
    try:
        available_profiles = {case.profile.name for case in suite.cases}
        for key in keys:
            if key.profile in available_profiles:
                definition = suite.definition(key)
            else:
                definition = BenchmarkDefinition(
                    key.name,
                    _unreachable_benchmark,
                    {},
                )
            runner.bench_func(
                definition.name,
                definition.function,
                metadata=definition.metadata,
            )
    finally:
        suite.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
