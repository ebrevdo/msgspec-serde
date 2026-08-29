"""JSON codecs with direct NumPy array serialization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypeVar, cast

import msgspec

from ._serde import (
    decode_native_model,
    encode_native_model,
    make_dec_hook,
    make_enc_hook,
    register_serde_type,
    serde_fallback_annotations,
    write_encoded,
)

_ModelT = TypeVar("_ModelT", bound=msgspec.Struct)


class Encoder:
    """Encode Struct models as JSON without NumPy-to-list conversion."""

    __slots__ = ("_decimal_format", "_encoder", "_order", "_uuid_format")

    def __init__(
        self,
        *,
        enc_hook: Any = None,
        decimal_format: Literal["string", "number"] = "string",
        uuid_format: Literal["canonical", "hex"] = "canonical",
        order: Literal["deterministic", "sorted"] | None = None,
    ) -> None:
        self._encoder = msgspec.json.Encoder(
            enc_hook=make_enc_hook(user_hook=enc_hook),
            decimal_format=decimal_format,
            uuid_format=uuid_format,
            order=order,
        )
        self._decimal_format = decimal_format
        self._uuid_format = uuid_format
        self._order = order

    def _encode_native(self, value: Any) -> bytes | None:
        return encode_native_model(
            value,
            is_json=True,
            fallback_encoder=self._encoder.encode,
            order=self._order,
            decimal_format=self._decimal_format,
            uuid_format=self._uuid_format,
        )

    def encode(self, value: Any) -> bytes:
        """Encode one value."""

        encoded = self._encode_native(value)
        if encoded is not None:
            return encoded
        return self._encoder.encode(value)

    def encode_into(
        self,
        value: Any,
        buffer: bytearray,
        offset: int = 0,
    ) -> None:
        """Encode one value into an existing bytearray."""

        encoded = self._encode_native(value)
        if encoded is not None:
            write_encoded(buffer, encoded, offset)
            return
        self._encoder.encode_into(value, buffer, offset)

    def encode_lines(self, values: Iterable[Any]) -> bytes:
        """Encode an iterable as newline-delimited JSON."""

        output = bytearray()
        for value in values:
            encoded = self._encode_native(value)
            if encoded is None:
                encoded = self._encoder.encode(value)
            output.extend(encoded)
            output.append(0x0A)
        return bytes(output)


class Decoder:
    """Decode JSON into Struct models with owned NumPy arrays."""

    __slots__ = ("_decoder", "_fallback_decoders", "_strict", "_target_type")

    def __init__(
        self,
        type: type[_ModelT],
        *,
        strict: bool = True,
        dec_hook: Any = None,
        float_hook: Any = None,
    ) -> None:
        self._target_type = type
        combined_hook = make_dec_hook(dec_hook)
        self._decoder = msgspec.json.Decoder(
            type=type,
            strict=strict,
            dec_hook=combined_hook,
            float_hook=float_hook,
        )
        self._strict = strict
        self._fallback_decoders = {
            callback_id: msgspec.json.Decoder(
                type=annotation,
                strict=strict,
                dec_hook=combined_hook,
                float_hook=float_hook,
            ).decode
            for callback_id, annotation in serde_fallback_annotations(type)
        }

    def decode(self, buffer: Any) -> _ModelT:
        """Decode one JSON document."""

        decoded = decode_native_model(
            buffer,
            self._target_type,
            is_json=True,
            strict=self._strict,
            fallback_decoders=self._fallback_decoders,
        )
        if decoded is not None:
            return cast(_ModelT, decoded)
        return cast(_ModelT, self._decoder.decode(buffer))

    def decode_lines(self, buffer: Any) -> list[_ModelT]:
        """Decode newline-delimited JSON documents."""

        return cast(list[_ModelT], self._decoder.decode_lines(buffer))


def encode(
    value: Any,
    *,
    enc_hook: Any = None,
    order: Literal["deterministic", "sorted"] | None = None,
) -> bytes:
    """Encode one value as JSON."""

    return Encoder(enc_hook=enc_hook, order=order).encode(value)


def decode(
    buffer: Any,
    *,
    type: type[_ModelT],
    strict: bool = True,
    dec_hook: Any = None,
    float_hook: Any = None,
) -> _ModelT:
    """Decode one JSON document into a Struct model."""

    return Decoder(
        type,
        strict=strict,
        dec_hook=dec_hook,
        float_hook=float_hook,
    ).decode(buffer)


def register(type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Eagerly compile a native serde plan, returning fallback field ids."""

    return register_serde_type(type)


__all__ = ["Decoder", "Encoder", "decode", "encode", "register"]
