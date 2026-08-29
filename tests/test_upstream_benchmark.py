from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import msgspec
import numpy as np

from benchmarks.upstream_msgspec.benchmarks import profile_msgspec_serde
from msgspec_serde import json as native_json
from msgspec_serde import msgpack as native_msgpack

ROOT = Path(__file__).parents[1]
VENDORED = ROOT / "benchmarks" / "upstream_msgspec" / "benchmarks"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_uses_written_numpy_struct_experiments() -> None:
    source = profile_msgspec_serde.make_filesystem_data(4, vector_length=3)
    model = profile_msgspec_serde._array_model(source)

    assert profile_msgspec_serde.PROFILE_LABELS == (
        "msgspec",
        "msgspec_array_hooks",
        "msgspec_array_native",
        "msgspec_fb_array_native",
    )
    assert model.int_values.dtype == np.dtype(np.int32)
    assert model.float_values.dtype == np.dtype(np.float64)

    codec_pairs = (
        (
            msgspec.json.Encoder(enc_hook=profile_msgspec_serde.numpy_enc_hook),
            msgspec.json.Decoder(
                profile_msgspec_serde.ArrayDirectory,
                dec_hook=profile_msgspec_serde.numpy_dec_hook,
            ),
        ),
        (
            msgspec.msgpack.Encoder(
                enc_hook=profile_msgspec_serde.numpy_enc_hook
            ),
            msgspec.msgpack.Decoder(
                profile_msgspec_serde.ArrayDirectory,
                dec_hook=profile_msgspec_serde.numpy_dec_hook,
            ),
        ),
        (
            native_json.Encoder(),
            native_json.Decoder(profile_msgspec_serde.ArrayDirectory),
        ),
        (
            native_msgpack.Encoder(),
            native_msgpack.Decoder(profile_msgspec_serde.ArrayDirectory),
        ),
    )
    for encoder, decoder in codec_pairs:
        restored = decoder.decode(encoder.encode(model))
        assert restored.int_values.dtype == np.dtype(np.int32)
        assert restored.float_values.dtype == np.dtype(np.float64)
        assert len(restored.contents) == len(model.contents)

    assert native_json.encode(model) == msgspec.json.encode(
        model,
        enc_hook=profile_msgspec_serde.numpy_enc_hook,
    )
    assert native_msgpack.encode(model) == msgspec.msgpack.encode(
        model,
        enc_hook=profile_msgspec_serde.numpy_enc_hook,
    )


def test_generated_upstream_benchmark_round_trips_all_codecs() -> None:
    generator = _load_module(
        "_test_upstream_generator",
        VENDORED / "generate_data.py",
    )
    benchmark = _load_module(
        "_test_upstream_msgspec_serde",
        VENDORED / "bench_msgspec_serde.py",
    )
    adapter = benchmark.GeneratedAdapter()
    try:
        source = generator.make_filesystem_data(4, vector_length=3)
        model = adapter.prepare(source)

        assert model.int_values.dtype == np.dtype(np.int32)
        assert model.float_values.dtype == np.dtype(np.float64)

        flatbuffer = adapter.encode_flatbuffer(model)
        materialized = adapter.decode_flatbuffer(flatbuffer)
        json_buffer = adapter.json_encoder.encode(model)
        msgpack_buffer = adapter.msgpack_encoder.encode(model)
        assert json_buffer == msgspec.json.encode(
            model,
            enc_hook=profile_msgspec_serde.numpy_enc_hook,
        )
        assert msgpack_buffer == msgspec.msgpack.encode(
            model,
            enc_hook=profile_msgspec_serde.numpy_enc_hook,
        )
        from_json = adapter.json_decoder.decode(json_buffer)
        from_msgpack = adapter.msgpack_decoder.decode(msgpack_buffer)

        json_destination = bytearray(b"prefix")
        adapter.json_encoder.encode_into(model, json_destination, -1)
        assert json_destination[6:] == json_buffer
        msgpack_destination = bytearray(b"prefix")
        adapter.msgpack_encoder.encode_into(model, msgpack_destination, -1)
        assert msgpack_destination[6:] == msgpack_buffer

        assert materialized.int_values.dtype == np.dtype(np.int32)
        for decoded in (from_json, from_msgpack):
            assert decoded.int_values.dtype == np.dtype(np.int32)
            assert (
                decoded.int_values.flags.owndata or decoded.int_values.base is not None
            )
            assert decoded.int_values.flags.writeable

        def move_tags_last(value: Any) -> Any:
            if isinstance(value, list):
                return [move_tags_last(item) for item in value]
            if not isinstance(value, dict):
                return value
            tag = value.get("__msgspec_serde_type__")
            reordered = {
                key: move_tags_last(item)
                for key, item in value.items()
                if key != "__msgspec_serde_type__"
            }
            if tag is not None:
                reordered["__msgspec_serde_type__"] = tag
            return reordered

        reordered = move_tags_last(msgspec.json.decode(json_buffer))
        reordered_json = adapter.json_decoder.decode(msgspec.json.encode(reordered))
        reordered_msgpack = adapter.msgpack_decoder.decode(
            msgspec.msgpack.encode(reordered)
        )
        expected = adapter._model_signature(model)
        assert adapter._model_signature(reordered_json) == expected
        assert adapter._model_signature(reordered_msgpack) == expected
    finally:
        adapter.close()
