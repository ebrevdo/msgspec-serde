from __future__ import annotations

import functools
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from msgspec_serde import flatbuffer, generate
from msgspec_serde import json as generated_json
from msgspec_serde import msgpack as generated_msgpack

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "filesystem.fbs"
MODULE_NAME = "_upstream_benchmark_msgspec_serde_generated"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _source_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    common = (
        value["name"],
        value["created_by"],
        value["created_at"],
        value.get("updated_by"),
        value.get("updated_at"),
        tuple(value["int_values"]),
        tuple(value["float_values"]),
    )
    if value["type"] == "file":
        return ("file", *common, value["nbytes"], value["permissions"])
    return (
        "directory",
        *common,
        tuple(_source_signature(child) for child in value["contents"]),
    )


class GeneratedAdapter:
    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="msgspec-serde-upstream-benchmark-"
        )
        output = Path(self._temporary_directory.name)
        module_path = generate(SCHEMA, output, gen_onefile=True)
        self.module = _load_module(MODULE_NAME, module_path)
        self.json_encoder = generated_json.Encoder()
        self.json_decoder = generated_json.Decoder(self.module.Directory)
        self.msgpack_encoder = generated_msgpack.Encoder()
        self.msgpack_decoder = generated_msgpack.Decoder(self.module.Directory)
        self.flatbuffer_encoder = flatbuffer.Encoder()
        self.flatbuffer_decoder = flatbuffer.Decoder(self.module.Directory)
        self._source: dict[str, Any] | None = None
        self._model: Any = None

    def close(self) -> None:
        sys.modules.pop(MODULE_NAME, None)
        self._temporary_directory.cleanup()

    def _from_source(self, value: dict[str, Any]) -> Any:
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
            return self.module.File(
                **common,
                nbytes=value["nbytes"],
                permissions=value["permissions"],
            )
        return self.module.Directory(
            **common,
            contents=[self._from_source(child) for child in value["contents"]],
        )

    def _model_signature(self, value: Any) -> tuple[Any, ...]:
        common = (
            value.name,
            value.created_by,
            value.created_at,
            value.updated_by,
            value.updated_at,
            tuple(value.int_values.tolist()),
            tuple(value.float_values.tolist()),
        )
        if isinstance(value, self.module.File):
            return ("file", *common, value.nbytes, value.permissions)
        return (
            "directory",
            *common,
            tuple(self._model_signature(child) for child in value.contents),
        )

    def _validate(self, source: dict[str, Any], model: Any) -> None:
        expected = _source_signature(source)
        flatbuffer = self.flatbuffer_encoder.encode(model)
        round_trips = (
            (flatbuffer, self.flatbuffer_decoder.decode),
            (self.json_encoder.encode(model), self.json_decoder.decode),
            (self.msgpack_encoder.encode(model), self.msgpack_decoder.decode),
        )
        for buffer, decode in round_trips:
            if self._model_signature(decode(buffer)) != expected:
                raise AssertionError(
                    "msgspec-serde benchmark round trip changed data"
                )

    def prepare(self, source: dict[str, Any]) -> Any:
        if source is not self._source:
            model = self._from_source(source)
            self._validate(source, model)
            self._source = source
            self._model = model
        return self._model

    def encode_flatbuffer(self, model: Any) -> memoryview:
        return self.flatbuffer_encoder.encode(model)

    def decode_flatbuffer(self, buffer: bytes | memoryview) -> Any:
        return self.flatbuffer_decoder.decode(buffer)


@functools.cache
def generated_adapter() -> GeneratedAdapter:
    return GeneratedAdapter()
