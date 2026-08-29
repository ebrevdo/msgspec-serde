"""Run one benchmark while keeping ordinary user work off its CPU core."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

_HANDLED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


@dataclass(frozen=True, slots=True)
class _ThreadAffinity:
    start_time: int
    cpus: frozenset[int]


def _interrupt(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _start_time(thread_id: int) -> int:
    stat = Path(f"/proc/{thread_id}/stat").read_text(encoding="utf-8")
    fields_after_name = stat[stat.rfind(")") + 2 :].split()
    return int(fields_after_name[19])


def _user_thread_ids(user_id: int) -> tuple[int, ...]:
    thread_ids: list[int] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            status = (process / "status").read_text(encoding="utf-8")
            uid_line = next(
                line for line in status.splitlines() if line.startswith("Uid:")
            )
            if int(uid_line.split()[1]) != user_id:
                continue
            thread_ids.extend(
                int(thread.name)
                for thread in (process / "task").iterdir()
                if thread.name.isdecimal()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
            continue
    return tuple(thread_ids)


def _restrict_user_threads(reserved: frozenset[int]) -> dict[int, _ThreadAffinity]:
    available = frozenset(os.sched_getaffinity(0))
    housekeeping = available.difference(reserved)
    if not reserved.issubset(available):
        raise RuntimeError(
            f"reserved CPUs {sorted(reserved)} are not all available in "
            f"{sorted(available)}"
        )
    if not housekeeping:
        raise RuntimeError("CPU reservation leaves no housekeeping CPU")

    saved: dict[int, _ThreadAffinity] = {}
    try:
        for thread_id in _user_thread_ids(os.getuid()):
            try:
                original = frozenset(os.sched_getaffinity(thread_id))
                restricted = original.difference(reserved)
                if restricted == original:
                    continue
                if not restricted:
                    restricted = housekeeping
                start_time = _start_time(thread_id)
                os.sched_setaffinity(thread_id, restricted)
                saved[thread_id] = _ThreadAffinity(start_time, original)
            except OSError:
                continue
    except BaseException:
        _restore_user_threads(saved)
        raise
    return saved


def _restore_user_threads(saved: dict[int, _ThreadAffinity]) -> None:
    for thread_id, original in saved.items():
        try:
            if _start_time(thread_id) == original.start_time:
                os.sched_setaffinity(thread_id, original.cpus)
        except OSError:
            continue


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--sibling", type=int, default=6)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a benchmark command is required after --")

    previous_handlers = {
        signum: signal.signal(signum, _interrupt) for signum in _HANDLED_SIGNALS
    }
    reserved = frozenset((args.cpu, args.sibling))
    saved: dict[int, _ThreadAffinity] = {}
    try:
        saved = _restrict_user_threads(reserved)
        print(
            f"reserved CPU {args.cpu}; left sibling CPU {args.sibling} idle; "
            f"moved {len(saved)} user threads",
            flush=True,
        )
        completed = subprocess.run(
            ["taskset", "-c", str(args.cpu), *command],
            cwd=args.cwd,
            check=False,
        )
        return completed.returncode
    finally:
        _restore_user_threads(saved)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
