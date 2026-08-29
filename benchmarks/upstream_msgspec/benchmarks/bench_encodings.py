from __future__ import annotations

import dataclasses
import json
import timeit
from typing import Any, Callable, Literal

import msgspec

from .bench_msgspec_serde import generated_adapter
from .generate_data import DEFAULT_VECTOR_LENGTH, make_filesystem_data


class File(msgspec.Struct, kw_only=True, omit_defaults=True, tag="file"):
    name: str
    created_by: str
    created_at: str
    int_values: list[int]
    float_values: list[float]
    updated_by: str | None = None
    updated_at: str | None = None
    nbytes: int
    permissions: Literal["READ", "WRITE", "READ_WRITE"]


class Directory(msgspec.Struct, kw_only=True, omit_defaults=True, tag="directory"):
    name: str
    created_by: str
    created_at: str
    int_values: list[int]
    float_values: list[float]
    updated_by: str | None = None
    updated_at: str | None = None
    contents: list[File | Directory]


@dataclasses.dataclass
class Benchmark:
    label: str
    encode: Callable
    decode: Callable
    prepare: Callable | None = None

    def run(self, data: Any) -> dict:
        if self.prepare is not None:
            data = self.prepare(data)
        timer = timeit.Timer("func(data)", globals={"func": self.encode, "data": data})
        n, t = timer.autorange()
        encode_time = t / n

        data = self.encode(data)

        timer = timeit.Timer("func(data)", globals={"func": self.decode, "data": data})
        n, t = timer.autorange()
        decode_time = t / n

        return {
            "label": self.label,
            "encode": encode_time,
            "decode": decode_time,
            "wire_bytes": len(data),
        }


def json_benchmarks():
    import orjson

    return [
        Benchmark("orjson", orjson.dumps, orjson.loads),
    ]


def msgpack_benchmarks():
    import ormsgpack

    return [
        Benchmark("ormsgpack", ormsgpack.packb, ormsgpack.unpackb),
    ]


def flatbuffers_benchmarks():
    generated = generated_adapter()
    return [
        Benchmark(
            "msgspec-serde materialized",
            generated.encode_flatbuffer,
            generated.decode_flatbuffer,
            prepare=generated.prepare,
        ),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark different python serialization libraries"
    )
    parser.add_argument(
        "-n",
        type=int,
        help="The number of objects in the generated data, defaults to 1000",
        default=1000,
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["json", "msgpack", "flatbuffers"],
        default="json",
        help="The protocol to benchmark, defaults to JSON",
    )
    parser.add_argument(
        "--vector-length",
        type=int,
        help=f"The number of integers and floats stored on every object, defaults to {DEFAULT_VECTOR_LENGTH}",
        default=DEFAULT_VECTOR_LENGTH,
    )
    args = parser.parse_args()

    if args.protocol == "json":
        benchmarks = json_benchmarks()
    elif args.protocol == "msgpack":
        benchmarks = msgpack_benchmarks()
    else:
        benchmarks = flatbuffers_benchmarks()

    data = make_filesystem_data(args.n, vector_length=args.vector_length)

    results = [benchmark.run(data) for benchmark in benchmarks]
    for result in results:
        print(
            json.dumps(
                {
                    "benchmark": "encodings",
                    "objects": args.n,
                    "vector_length": args.vector_length,
                    "protocol": args.protocol,
                    **result,
                }
            )
        )


if __name__ == "__main__":
    main()
