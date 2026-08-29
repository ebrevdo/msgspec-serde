from __future__ import annotations

import argparse
import json
import timeit
from functools import cache
from typing import Any, Literal, get_args, get_origin

import msgspec
import numpy as np
import numpy.typing as npt

from msgspec_flatbuffers import json as generated_json
from msgspec_flatbuffers import msgpack as generated_msgpack

from .bench_encodings import Directory as UpstreamDirectory
from .bench_msgspec_flatbuffers import generated_adapter
from .generate_data import DEFAULT_VECTOR_LENGTH, make_filesystem_data

DEFAULT_PROFILE_VECTOR_LENGTHS = (DEFAULT_VECTOR_LENGTH, 256)
PROFILE_LABELS = (
    "msgspec",
    "msgspec_array_hooks",
    "msgspec_array_native",
    "msgspec_fb_array_native",
)


class ArrayFile(msgspec.Struct, kw_only=True, omit_defaults=True, tag="file"):
    name: str
    created_by: str
    created_at: str
    int_values: npt.NDArray[np.int32]
    float_values: npt.NDArray[np.float64]
    updated_by: str | None = None
    updated_at: str | None = None
    nbytes: int
    permissions: Literal["READ", "WRITE", "READ_WRITE"]


class ArrayDirectory(msgspec.Struct, kw_only=True, omit_defaults=True, tag="directory"):
    name: str
    created_by: str
    created_at: str
    int_values: npt.NDArray[np.int32]
    float_values: npt.NDArray[np.float64]
    updated_by: str | None = None
    updated_at: str | None = None
    contents: list[ArrayFile | ArrayDirectory]


@cache
def _ndarray_dtype(annotation: Any) -> np.dtype[Any] | None:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is npt.NDArray:
        return np.dtype(arguments[0])
    if origin is not np.ndarray:
        return None
    dtype_arguments = get_args(arguments[-1])
    return np.dtype(dtype_arguments[0])


def numpy_enc_hook(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        raise NotImplementedError
    if value.ndim != 1:
        raise TypeError("benchmark numeric vectors must be one-dimensional")
    return value.tolist()


def numpy_dec_hook(annotation: Any, value: Any) -> Any:
    dtype = _ndarray_dtype(annotation)
    if dtype is None:
        raise NotImplementedError
    result = np.asarray(value, dtype=dtype, order="C")
    if not result.flags.owndata or not result.flags.writeable:
        result = result.copy(order="C")
    if result.ndim != 1:
        raise TypeError("benchmark numeric vectors must be one-dimensional")
    return result


def _array_model(value):
    common = {
        "name": value["name"],
        "created_by": value["created_by"],
        "created_at": value["created_at"],
        "int_values": np.asarray(value["int_values"], dtype=np.int32),
        "float_values": np.asarray(value["float_values"], dtype=np.float64),
        "updated_by": value.get("updated_by"),
        "updated_at": value.get("updated_at"),
    }
    if value["type"] == "file":
        return ArrayFile(
            **common,
            nbytes=value["nbytes"],
            permissions=value["permissions"],
        )
    return ArrayDirectory(
        **common,
        contents=[_array_model(child) for child in value["contents"]],
    )


def _measure(function, value) -> float:
    timer = timeit.Timer(
        "function(value)", globals={"function": function, "value": value}
    )
    count, elapsed = timer.autorange()
    return elapsed / count


def _row(label, encoder, decoder, model):
    buffer = encoder.encode(model)
    return {
        "label": label,
        "encode_us": _measure(encoder.encode, model) * 1_000_000,
        "decode_us": _measure(decoder.decode, buffer) * 1_000_000,
        "wire_bytes": len(buffer),
    }


def _profile_rows(codec, native_codec, adapter, upstream, array_model, generated):
    upstream_encoder = codec.Encoder()
    upstream_decoder = codec.Decoder(UpstreamDirectory)
    array_hooks_encoder = codec.Encoder(enc_hook=numpy_enc_hook)
    array_hooks_decoder = codec.Decoder(ArrayDirectory, dec_hook=numpy_dec_hook)
    array_native_encoder = native_codec.Encoder()
    array_native_decoder = native_codec.Decoder(ArrayDirectory)
    generated_native_encoder = native_codec.Encoder()
    generated_native_decoder = native_codec.Decoder(adapter.module.Directory)
    return [
        _row(PROFILE_LABELS[0], upstream_encoder, upstream_decoder, upstream),
        _row(
            PROFILE_LABELS[1],
            array_hooks_encoder,
            array_hooks_decoder,
            array_model,
        ),
        _row(
            PROFILE_LABELS[2],
            array_native_encoder,
            array_native_decoder,
            array_model,
        ),
        _row(
            PROFILE_LABELS[3],
            generated_native_encoder,
            generated_native_decoder,
            generated,
        ),
    ]


def _run_profile(objects: int, vector_length: int):
    source = make_filesystem_data(objects, vector_length=vector_length)
    upstream = msgspec.convert(source, UpstreamDirectory)
    array_model = _array_model(source)
    adapter = generated_adapter()
    generated = adapter.prepare(source)

    return {
        "json": _profile_rows(
            msgspec.json,
            generated_json,
            adapter,
            upstream,
            array_model,
            generated,
        ),
        "msgpack": _profile_rows(
            msgspec.msgpack,
            generated_msgpack,
            adapter,
            upstream,
            array_model,
            generated,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument(
        "--vector-length",
        dest="vector_lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PROFILE_VECTOR_LENGTHS,
        help="vector lengths to profile, defaults to 16 and 256",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n must be at least 1")
    if any(length < 0 for length in args.vector_lengths):
        parser.error("--vector-length values must be non-negative")

    for vector_length in args.vector_lengths:
        results = _run_profile(args.n, vector_length)
        for protocol, rows in results.items():
            for row in rows:
                print(
                    json.dumps(
                        {
                            "benchmark": "msgspec-profile",
                            "objects": args.n,
                            "vector_length": vector_length,
                            "protocol": protocol,
                            **row,
                        }
                    )
                )


if __name__ == "__main__":
    main()
