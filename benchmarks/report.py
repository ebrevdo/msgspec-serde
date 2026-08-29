"""Generate the benchmark report's bar-chart images from saved result files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyperf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROFILE_LABELS = {
    "msgspec": "msgspec",
    "msgspec_array_hooks": "msgspec_array_hooks",
    "msgspec_array_native": "msgspec_array_native",
    "msgspec_fb_array_native": "msgspec_fb_array_native",
}
PROFILE_COLORS = {
    "msgspec": "#f97316",
    "msgspec_array_hooks": "#9333ea",
    "msgspec_array_native": "#16a34a",
    "msgspec_fb_array_native": "#2563eb",
}
EXTERNAL_LABELS = {"json": "orjson", "msgpack": "ormsgpack"}
EXTERNAL_COLOR = "#6b7280"
CODEC_COLORS = {
    "JSON": "#16a34a",
    "MessagePack": "#9333ea",
    "FlatBuffers": "#2563eb",
}
PYTHON_FLATBUFFERS_OPERATIONS = (
    "serialize",
    "deserialize",
    "partial_access",
    "full_traversal",
)
PYTHON_FLATBUFFERS_TITLES = {
    "serialize": "Encode materialized model to FlatBuffer",
    "deserialize": "Decode FlatBuffer to materialized model",
    "partial_access": "Open lazy view and read selected fields",
    "full_traversal": "Open lazy view and traverse all fields",
}
PYTHON_FLATBUFFERS_GENERATED_LABEL = "msgspec_serde generated API"
SCALING_PROFILES = ("small", "medium", "large")
RESULT_CONTEXT_KEYS = (
    "benchmark_schema_sha256",
    "hostname",
    "cpu_model_name",
    "platform",
    "python_version",
    "flatc_version",
    "msgspec_flatbuffers_version",
    "flatbuffers_version",
    "msgspec_version",
    "numpy_version",
)


@dataclass(frozen=True, slots=True)
class Measurement:
    operation: str
    profile: str
    adapter: str
    logical_elements: int
    wire_bytes: int
    values: tuple[float, ...]

    @property
    def mean_us(self) -> float:
        return statistics.mean(self.values) * 1e6

    @property
    def stdev_us(self) -> float:
        return statistics.stdev(self.values) * 1e6 if len(self.values) > 1 else 0.0


@dataclass(slots=True)
class _MeasurementGroup:
    logical_elements: int
    wire_bytes: int
    values: list[float]


@dataclass(frozen=True, slots=True)
class EncodingResult:
    benchmark: str
    objects: int
    vector_length: int
    protocol: str
    label: str
    encode_us: float
    decode_us: float
    wire_bytes: int
    encode_stdev_us: float = 0.0
    decode_stdev_us: float = 0.0


def _json_objects(paths: Sequence[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if isinstance(value, dict):
                yield path, line_number, value


def _required(value: dict[str, Any], name: str, path: Path, line_number: int) -> Any:
    try:
        return value[name]
    except KeyError as error:
        raise ValueError(f"missing {name!r} at {path}:{line_number}") from error


def _load_json_results(
    paths: Sequence[Path],
    benchmark: str,
) -> tuple[EncodingResult, ...]:
    results: list[EncodingResult] = []
    for path, line_number, value in _json_objects(paths):
        if value.get("benchmark") != benchmark:
            continue
        if benchmark == "msgspec-profile":
            encode_us = float(_required(value, "encode_us", path, line_number))
            decode_us = float(_required(value, "decode_us", path, line_number))
        else:
            encode_us = float(_required(value, "encode", path, line_number)) * 1e6
            decode_us = float(_required(value, "decode", path, line_number)) * 1e6
        results.append(
            EncodingResult(
                benchmark=benchmark,
                objects=int(_required(value, "objects", path, line_number)),
                vector_length=int(_required(value, "vector_length", path, line_number)),
                protocol=str(_required(value, "protocol", path, line_number)),
                label=str(_required(value, "label", path, line_number)),
                encode_us=encode_us,
                decode_us=decode_us,
                wire_bytes=int(_required(value, "wire_bytes", path, line_number)),
            )
        )
    if not results:
        raise ValueError(f"no {benchmark!r} rows found in the supplied result files")
    return _average_results(results)


def _average_results(results: Sequence[EncodingResult]) -> tuple[EncodingResult, ...]:
    grouped: dict[tuple[int, int, str, str], list[EncodingResult]] = defaultdict(list)
    for result in results:
        grouped[
            (result.objects, result.vector_length, result.protocol, result.label)
        ].append(result)
    averaged: list[EncodingResult] = []
    for (objects, vector_length, protocol, label), group in grouped.items():
        wire_sizes = {item.wire_bytes for item in group}
        if len(wire_sizes) != 1:
            raise ValueError(
                f"wire size changed for {protocol}/{vector_length}/{label}: "
                f"{sorted(wire_sizes)}"
            )
        averaged.append(
            EncodingResult(
                benchmark=group[0].benchmark,
                objects=objects,
                vector_length=vector_length,
                protocol=protocol,
                label=label,
                encode_us=statistics.mean(item.encode_us for item in group),
                decode_us=statistics.mean(item.decode_us for item in group),
                wire_bytes=wire_sizes.pop(),
                encode_stdev_us=(
                    statistics.stdev(item.encode_us for item in group)
                    if len(group) > 1
                    else 0.0
                ),
                decode_stdev_us=(
                    statistics.stdev(item.decode_us for item in group)
                    if len(group) > 1
                    else 0.0
                ),
            )
        )
    return tuple(averaged)


def _load_measurements(paths: Sequence[Path]) -> tuple[Measurement, ...]:
    grouped: dict[tuple[str, str, str], _MeasurementGroup] = {}
    expected_context: dict[str, Any] | None = None
    for path in paths:
        suite = pyperf.BenchmarkSuite.load(str(path))
        metadata = suite.get_metadata()
        context = {key: metadata.get(key) for key in RESULT_CONTEXT_KEYS}
        if expected_context is None:
            expected_context = context
        elif context != expected_context:
            changed = [
                key
                for key in RESULT_CONTEXT_KEYS
                if context[key] != expected_context[key]
            ]
            raise ValueError(
                f"{path} does not match the first result file: {', '.join(changed)}"
            )
        for benchmark in suite.get_benchmarks():
            if benchmark.get_unit() != "second":
                raise ValueError(
                    f"{path} contains non-timing benchmark {benchmark.get_name()!r}"
                )
            benchmark_metadata = benchmark.get_metadata()
            key = (
                str(benchmark_metadata["operation"]),
                str(benchmark_metadata["profile"]),
                str(benchmark_metadata["adapter"]),
            )
            group = grouped.setdefault(
                key,
                _MeasurementGroup(
                    logical_elements=int(benchmark_metadata["logical_elements"]),
                    wire_bytes=int(benchmark_metadata["wire_bytes"]),
                    values=[],
                ),
            )
            if group.wire_bytes != int(benchmark_metadata["wire_bytes"]):
                raise ValueError(f"wire size changed across result files for {key}")
            group.values.extend(benchmark.get_values())
    return tuple(
        Measurement(
            operation=operation,
            profile=profile,
            adapter=adapter,
            logical_elements=group.logical_elements,
            wire_bytes=group.wire_bytes,
            values=tuple(group.values),
        )
        for (operation, profile, adapter), group in grouped.items()
    )


def _format_time(value_us: float) -> str:
    if value_us >= 1_000:
        return f"{value_us / 1_000:.2f} ms"
    if value_us >= 10:
        return f"{value_us:.1f} µs"
    return f"{value_us:.2f} µs"


def _scaled_times(values_us: Sequence[float]) -> tuple[np.ndarray, str, float]:
    divisor = 1_000.0 if max(values_us) >= 1_000 else 1.0
    unit = "ms" if divisor == 1_000 else "µs"
    return np.asarray(values_us) / divisor, unit, divisor


def _only_int(values: Iterable[int], description: str) -> int:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"{description}: {unique}")
    return unique.pop()


def _save_figure(
    figure: Any,
    output: Path,
    *,
    layout_top: float,
) -> None:
    figure.text(
        0.995,
        0.005,
        "Error bars show ±1 standard deviation.",
        ha="right",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, layout_top))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _bar_panel(
    axis: Any,
    labels: Sequence[str],
    values_us: Sequence[float],
    colors: Sequence[str],
    *,
    title: str,
    errors_us: Sequence[float] | None = None,
    baseline_us: float | None = None,
    baseline_name: str = "baseline time",
) -> None:
    values, unit, divisor = _scaled_times(values_us)
    errors = (
        np.zeros(len(values)) if errors_us is None else np.asarray(errors_us) / divisor
    )
    positions = np.arange(len(labels))
    axis.barh(
        positions,
        values,
        xerr=errors,
        color=colors,
        height=0.68,
        error_kw={"ecolor": "#374151", "elinewidth": 1.2, "capsize": 3},
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=11)
    axis.set_xlabel(f"Elapsed time per call ({unit})")
    axis.grid(True, axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    maximum = float(max(values + errors))
    axis.set_xlim(0, maximum * 1.52)
    for position, (scaled, error, value_us) in enumerate(
        zip(values, errors, values_us)
    ):
        suffix = ""
        if baseline_us is not None:
            suffix = f" · {value_us / baseline_us:.1f}× {baseline_name}"
        axis.text(
            scaled + error + maximum * 0.025,
            position,
            f"{_format_time(value_us)}{suffix}",
            va="center",
            fontsize=8.5,
        )


def _matching(
    rows: Sequence[EncodingResult],
    *,
    objects: int,
    vector_length: int,
    protocol: str,
) -> tuple[EncodingResult, ...]:
    return tuple(
        row
        for row in rows
        if row.objects == objects
        and row.vector_length == vector_length
        and row.protocol == protocol
    )


def _one_by_label(rows: Sequence[EncodingResult], label: str) -> EncodingResult:
    matches = [row for row in rows if row.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r} row, found {len(matches)}")
    return matches[0]


def _plot_msgspec_extended(
    profile_rows: Sequence[EncodingResult],
    encoding_rows: Sequence[EncodingResult],
    output: Path,
) -> None:
    object_count = _only_int(
        (row.objects for row in profile_rows),
        "profile results contain different object counts",
    )
    cases = [
        (vector_length, protocol)
        for vector_length in sorted({row.vector_length for row in profile_rows})
        for protocol in ("json", "msgpack")
    ]
    figure, axes = plt.subplots(len(cases), 2, figsize=(18, 4.1 * len(cases)))
    for row_axes, (vector_length, protocol) in zip(axes, cases):
        profile_case = _matching(
            profile_rows,
            objects=object_count,
            vector_length=vector_length,
            protocol=protocol,
        )
        encoding_case = _matching(
            encoding_rows,
            objects=object_count,
            vector_length=vector_length,
            protocol=protocol,
        )
        by_label = {row.label: row for row in profile_case}
        missing = PROFILE_LABELS.keys() - by_label.keys()
        if missing:
            raise ValueError(
                f"missing {protocol}/{vector_length} profile rows: {sorted(missing)}"
            )
        external = _one_by_label(encoding_case, EXTERNAL_LABELS[protocol])
        selected = [
            by_label["msgspec"],
            external,
            by_label["msgspec_array_hooks"],
            by_label["msgspec_array_native"],
            by_label["msgspec_fb_array_native"],
        ]
        labels = [row.label for row in selected]
        colors = [
            PROFILE_COLORS["msgspec"],
            EXTERNAL_COLOR,
            PROFILE_COLORS["msgspec_array_hooks"],
            PROFILE_COLORS["msgspec_array_native"],
            PROFILE_COLORS["msgspec_fb_array_native"],
        ]
        baseline = by_label["msgspec"]
        for axis, operation in zip(row_axes, ("encode", "decode")):
            values = [getattr(row, f"{operation}_us") for row in selected]
            errors = [getattr(row, f"{operation}_stdev_us") for row in selected]
            _bar_panel(
                axis,
                labels,
                values,
                colors,
                title=(
                    f"{protocol.upper()} {operation}: {vector_length} values per vector"
                ),
                errors_us=errors,
                baseline_us=getattr(baseline, f"{operation}_us"),
                baseline_name="msgspec time",
            )
    figure.suptitle(
        "JSON and MessagePack serialization of numeric vectors",
        fontsize=18,
        y=0.99,
    )
    figure.text(
        0.5,
        0.965,
        (
            f"{object_count:,} objects; each has one int32 vector and one "
            "float64 vector."
        ),
        ha="center",
        fontsize=10,
    )
    _save_figure(figure, output, layout_top=0.94)


def _plot_codec_comparison(
    profile_rows: Sequence[EncodingResult],
    encoding_rows: Sequence[EncodingResult],
    output: Path,
) -> None:
    object_count = _only_int(
        (row.objects for row in profile_rows),
        "profile results contain different object counts",
    )
    vector_lengths = sorted({row.vector_length for row in profile_rows})
    figure, axes = plt.subplots(
        len(vector_lengths), 2, figsize=(15, 4.2 * len(vector_lengths))
    )
    if len(vector_lengths) == 1:
        axes = np.asarray([axes])
    for row_axes, vector_length in zip(axes, vector_lengths):
        json_row = _one_by_label(
            _matching(
                profile_rows,
                objects=object_count,
                vector_length=vector_length,
                protocol="json",
            ),
            "msgspec_fb_array_native",
        )
        msgpack_row = _one_by_label(
            _matching(
                profile_rows,
                objects=object_count,
                vector_length=vector_length,
                protocol="msgpack",
            ),
            "msgspec_fb_array_native",
        )
        flatbuffer_row = _one_by_label(
            _matching(
                encoding_rows,
                objects=object_count,
                vector_length=vector_length,
                protocol="flatbuffers",
            ),
            "msgspec-flatbuffers materialized",
        )
        rows = (json_row, msgpack_row, flatbuffer_row)
        labels = tuple(CODEC_COLORS)
        colors = tuple(CODEC_COLORS.values())
        for axis, operation in zip(row_axes, ("encode", "decode")):
            values = [getattr(row, f"{operation}_us") for row in rows]
            errors = [getattr(row, f"{operation}_stdev_us") for row in rows]
            _bar_panel(
                axis,
                labels,
                values,
                colors,
                title=f"{operation.capitalize()}: {vector_length} values per vector",
                errors_us=errors,
                baseline_us=min(values),
                baseline_name="fastest time",
            )
    figure.suptitle(
        "Full recursive File/Directory graph: JSON, MessagePack, and FlatBuffers",
        fontsize=18,
        y=0.99,
    )
    _save_figure(figure, output, layout_top=0.95)


def _measurement_map(
    measurements: Sequence[Measurement],
) -> dict[tuple[str, str, str], Measurement]:
    return {(item.operation, item.profile, item.adapter): item for item in measurements}


def _ratio_and_stdev(
    numerator: Measurement,
    denominator: Measurement,
) -> tuple[float, float]:
    ratio = numerator.mean_us / denominator.mean_us
    relative_variance = (numerator.stdev_us / numerator.mean_us) ** 2 + (
        denominator.stdev_us / denominator.mean_us
    ) ** 2
    return ratio, ratio * relative_variance**0.5


def _plot_python_flatbuffers(
    measurements: Sequence[Measurement],
    output: Path,
) -> None:
    by_key = _measurement_map(measurements)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    profile_labels = []
    for profile in SCALING_PROFILES:
        logical_elements = _only_int(
            (item.logical_elements for item in measurements if item.profile == profile),
            f"profile {profile!r} has different logical-element counts",
        )
        profile_labels.append(f"{profile}\n({logical_elements:,} logical elements)")
    for axis, operation in zip(axes.flat, PYTHON_FLATBUFFERS_OPERATIONS):
        ratios: list[float] = []
        ratio_errors: list[float] = []
        outcomes: list[str] = []
        for profile in SCALING_PROFILES:
            try:
                ours = by_key[(operation, profile, "msgspec_flatbuffers")]
                python = by_key[(operation, profile, "python_flatbuffers")]
            except KeyError as error:
                raise ValueError(
                    f"missing pyperf result for {error.args[0]}"
                ) from error
            ratio, ratio_error = _ratio_and_stdev(ours, python)
            ratios.append(ratio)
            ratio_errors.append(ratio_error)
            outcomes.append(
                f"{(1 / ratio):.1f}× faster" if ratio <= 1 else f"{ratio:.1f}× slower"
            )
        positions = np.arange(len(SCALING_PROFILES))
        axis.barh(
            positions,
            ratios,
            xerr=ratio_errors,
            color="#2563eb",
            height=0.64,
            error_kw={"ecolor": "#374151", "elinewidth": 1.2, "capsize": 3},
        )
        axis.axvline(1.0, color="#f97316", linestyle="--", linewidth=1.8)
        axis.set_yticks(positions, profile_labels)
        axis.invert_yaxis()
        maximum = max(ratio + error for ratio, error in zip(ratios, ratio_errors))
        axis.set_xlim(0, max(1.12, maximum * 1.25))
        axis.set_title(PYTHON_FLATBUFFERS_TITLES[operation], loc="left", fontsize=11)
        axis.set_xlabel("Relative elapsed time (official Python API = 1.0×)")
        axis.grid(True, axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        for position, (ratio, error, outcome) in enumerate(
            zip(ratios, ratio_errors, outcomes)
        ):
            axis.text(
                ratio + error + 0.025,
                position,
                outcome,
                va="center",
                fontsize=9,
            )
    figure.suptitle(
        f"{PYTHON_FLATBUFFERS_GENERATED_LABEL} vs. official Python FlatBuffers API",
        fontsize=18,
        y=0.995,
    )
    figure.legend(
        handles=(
            Patch(
                color="#2563eb",
                label=PYTHON_FLATBUFFERS_GENERATED_LABEL,
            ),
            Line2D(
                (0,),
                (0,),
                color="#f97316",
                linestyle="--",
                label="official Python FlatBuffers API (1.0×)",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=2,
    )
    _save_figure(figure, output, layout_top=0.91)


def generate_report(
    profile_results: Sequence[Path],
    encoding_results: Sequence[Path],
    flatbuffers_results: Sequence[Path],
    output_dir: Path,
) -> tuple[Path, ...]:
    profile_rows = _load_json_results(profile_results, "msgspec-profile")
    encoding_rows = _load_json_results(encoding_results, "encodings")
    flatbuffers_measurements = _load_measurements(flatbuffers_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "msgspec-extended.png",
        output_dir / "idl-codec-comparison.png",
        output_dir / "python-flatbuffers-comparison.png",
    )
    _plot_msgspec_extended(profile_rows, encoding_rows, outputs[0])
    _plot_codec_comparison(profile_rows, encoding_rows, outputs[1])
    _plot_python_flatbuffers(flatbuffers_measurements, outputs[2])
    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-results",
        nargs="+",
        type=Path,
        required=True,
        help="JSON-lines outputs from profile_msgspec_flatbuffers",
    )
    parser.add_argument(
        "--encoding-results",
        nargs="+",
        type=Path,
        required=True,
        help="JSON-lines outputs from bench_encodings",
    )
    parser.add_argument(
        "--flatbuffers-results",
        nargs="+",
        type=Path,
        required=True,
        help="pyperf JSON files from the first-party FlatBuffers comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/images"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = generate_report(
        args.profile_results,
        args.encoding_results,
        args.flatbuffers_results,
        args.output_dir,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
