"""CPU-time benchmarks for unions, nested buffers, and dynamic payloads."""

from __future__ import annotations

import gc
import importlib
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from msgspec_flatbuffers import generate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
REPEAT = 9


def _measure(function: Callable[[], object], number: int) -> tuple[float, float]:
    for _ in range(3):
        function()
    gc.collect()
    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(REPEAT):
            start = time.process_time_ns()
            for _ in range(number):
                function()
            samples.append((time.process_time_ns() - start) / number / 1_000)
    finally:
        gc.enable()
    return min(samples), statistics.median(samples)


def _generate(output: Path) -> tuple[Any, Any, Any, Any]:
    dynamic = FIXTURES / "dynamic"
    nested = FIXTURES / "nested_unions"
    for schema, root in (
        (dynamic / "payload.fbs", dynamic),
        (dynamic / "envelope.fbs", dynamic),
        (nested / "payload.fbs", nested),
        (nested / "envelope.fbs", nested),
    ):
        generate(schema, output, project_root=root)

    sys.path.insert(0, str(output))
    try:
        dynamic_payload = importlib.import_module("example.dynamic.payload")
        dynamic_envelope = importlib.import_module("example.dynamic.envelope")
        union_payload = importlib.import_module("example.nested.payload")
        nested_envelope = importlib.import_module("example.nested.envelope")
    finally:
        sys.path.pop(0)
    return dynamic_payload, dynamic_envelope, union_payload, nested_envelope


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="msgspec-flatbuffers-polymorphism-") as tmp:
        dynamic_payload, dynamic_envelope, union_payload, nested_envelope = _generate(
            Path(tmp)
        )

        small_metric = dynamic_payload.Metric(
            name="small",
            values=np.array([1.0, 2.0], dtype=np.float32),
        )
        large_metric = dynamic_payload.Metric(
            name="large",
            values=np.arange(65_536, dtype=np.float32),
        )
        residents = [
            union_payload.Cat(name=f"cat-{index}")
            if index % 2 == 0
            else union_payload.Dog(name=f"dog-{index}")
            for index in range(1_024)
        ]
        union_model = union_payload.Payload(
            favorite=union_payload.Cat(name="favorite"),
            residents=residents,
        )
        models: dict[str, tuple[object, int]] = {
            "dynamic_inner_large": (large_metric, 5),
            "dynamic_known_small": (
                dynamic_envelope.Envelope(
                    payload=dynamic_envelope.EnvelopePayload(small_metric)
                ),
                500,
            ),
            "dynamic_known_large": (
                dynamic_envelope.Envelope(
                    payload=dynamic_envelope.EnvelopePayload(large_metric)
                ),
                5,
            ),
            "dynamic_opaque_small": (
                dynamic_envelope.Envelope(
                    payload=dynamic_envelope.EnvelopePayload.opaque(
                        "Example.Dynamic.Future",
                        b"opaque",
                    )
                ),
                500,
            ),
            "dynamic_opaque_1mib": (
                dynamic_envelope.Envelope(
                    payload=dynamic_envelope.EnvelopePayload.opaque(
                        "Example.Dynamic.Future",
                        bytes(1 << 20),
                    )
                ),
                5,
            ),
            "union_scalar_and_vector": (union_model, 5),
            "nested_union_buffer": (
                nested_envelope.Envelope(payload=union_model),
                5,
            ),
        }

        for name, (model, number) in models.items():
            function = model.to_flatbuffer
            function()
            best, median = _measure(function, number)
            print(f"{name:<28} best={best:>10.3f} us  median={median:>10.3f} us")


if __name__ == "__main__":
    main()
