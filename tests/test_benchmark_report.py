from pathlib import Path

import pytest

from benchmarks.report import (
    CODEC_COLORS,
    EXTERNAL_LABELS,
    PROFILE_LABELS,
    PYTHON_FLATBUFFERS_GENERATED_LABEL,
    PYTHON_FLATBUFFERS_OPERATIONS,
    EncodingResult,
    Measurement,
    _load_json_results,
    _plot_codec_comparison,
    _plot_msgspec_extended,
    _plot_python_flatbuffers,
)


def _profile_rows() -> tuple[EncodingResult, ...]:
    rows = []
    for vector_length in (16, 256):
        for protocol in ("json", "msgpack"):
            for index, label in enumerate(PROFILE_LABELS, 1):
                rows.append(
                    EncodingResult(
                        benchmark="msgspec-profile",
                        objects=1_000,
                        vector_length=vector_length,
                        protocol=protocol,
                        label=label,
                        encode_us=100.0 * index * vector_length,
                        decode_us=150.0 * index * vector_length,
                        wire_bytes=1_000 * index,
                    )
                )
    return tuple(rows)


def _encoding_rows() -> tuple[EncodingResult, ...]:
    rows = []
    for vector_length in (16, 256):
        rows.extend(
            (
                EncodingResult(
                    "encodings",
                    1_000,
                    vector_length,
                    "json",
                    "orjson",
                    120.0 * vector_length,
                    180.0 * vector_length,
                    2_000,
                ),
                EncodingResult(
                    "encodings",
                    1_000,
                    vector_length,
                    "msgpack",
                    "ormsgpack",
                    80.0 * vector_length,
                    140.0 * vector_length,
                    1_500,
                ),
                EncodingResult(
                    "encodings",
                    1_000,
                    vector_length,
                    "flatbuffers",
                    "msgspec-serde materialized",
                    60.0 * vector_length,
                    90.0 * vector_length,
                    1_200,
                ),
            )
        )
    return tuple(rows)


def _flatbuffer_measurements() -> tuple[Measurement, ...]:
    measurements = []
    for operation in PYTHON_FLATBUFFERS_OPERATIONS:
        for index, profile in enumerate(("small", "medium", "large"), 1):
            measurements.extend(
                (
                    Measurement(
                        operation,
                        profile,
                        "msgspec_serde",
                        index * 100,
                        index * 1_000,
                        (index * 1e-6,),
                    ),
                    Measurement(
                        operation,
                        profile,
                        "python_flatbuffers",
                        index * 100,
                        index * 1_000,
                        (index * 10e-6,),
                    ),
                )
            )
    return tuple(measurements)


def test_json_result_loader_ignores_runner_output_and_averages(tmp_path: Path) -> None:
    result = tmp_path / "profile.jsonl"
    result.write_text(
        "runner status\n"
        '{"benchmark":"msgspec-profile","objects":1000,'
        '"vector_length":16,"protocol":"json","label":"msgspec",'
        '"encode_us":10,"decode_us":20,"wire_bytes":100}\n'
        '{"benchmark":"msgspec-profile","objects":1000,'
        '"vector_length":16,"protocol":"json","label":"msgspec",'
        '"encode_us":14,"decode_us":24,"wire_bytes":100}\n',
        encoding="utf-8",
    )

    rows = _load_json_results((result,), "msgspec-profile")

    assert len(rows) == 1
    assert rows[0].encode_us == 12
    assert rows[0].decode_us == 22
    assert rows[0].encode_stdev_us == pytest.approx(2.828427)
    assert rows[0].decode_stdev_us == pytest.approx(2.828427)


def test_report_plotters_write_all_requested_images(tmp_path: Path) -> None:
    outputs = (
        tmp_path / "msgspec.png",
        tmp_path / "codecs.png",
        tmp_path / "python-flatbuffers.png",
    )

    _plot_msgspec_extended(_profile_rows(), _encoding_rows(), outputs[0])
    _plot_codec_comparison(_profile_rows(), _encoding_rows(), outputs[1])
    _plot_python_flatbuffers(_flatbuffer_measurements(), outputs[2])

    assert "lazy_open" not in PYTHON_FLATBUFFERS_OPERATIONS
    assert tuple(PROFILE_LABELS) == (
        "msgspec",
        "msgspec_array_hooks",
        "msgspec_array_native",
        "msgspec_fb_array_native",
    )
    assert EXTERNAL_LABELS == {"json": "orjson", "msgpack": "ormsgpack"}
    assert PYTHON_FLATBUFFERS_GENERATED_LABEL == "msgspec_serde generated API"
    assert tuple(CODEC_COLORS) == ("JSON", "MessagePack", "FlatBuffers")
    assert all(path.stat().st_size > 10_000 for path in outputs)
