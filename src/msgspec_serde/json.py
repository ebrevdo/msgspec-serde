"""Encode and decode msgspec models with direct NumPy array support.

Example:
    Round-trip a model through JSON:

    >>> from msgspec_serde import json
    >>> encoded = json.encode(Monster(name="Orc"))
    >>> json.decode(encoded, type=Monster).name
    'Orc'
"""

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
    """Encode Struct models as JSON without NumPy-to-list conversion.

    Args:
        enc_hook: An optional callback for otherwise unsupported values.
        decimal_format: Whether to encode decimals as strings or numbers.
        uuid_format: Whether to encode UUIDs in canonical or hexadecimal form.
        order: Optional deterministic or sorted ordering for maps and structs.

    Example:
        Reuse a deterministically ordered encoder:

        >>> encoder = Encoder(order="deterministic")
        >>> encoder.encode(Monster(name="Orc"))
        b'{"name":"Orc"}'
    """

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
        """Encode one value as JSON.

        Args:
            value: The value to encode.

        Returns:
            The encoded UTF-8 JSON bytes.

        Example:
            Encode a model with a reusable encoder:

            >>> Encoder().encode(Monster(name="Orc"))
            b'{"name":"Orc"}'
        """

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
        """Encode one value into an existing bytearray.

        Args:
            value: The value to encode.
            buffer: The destination bytearray.
            offset: The starting byte offset. Use ``-1`` to append.

        Raises:
            ValueError: ``offset`` is less than ``-1``.

        Example:
            Append an encoded model to a bytearray:

            >>> destination = bytearray(b"prefix:")
            >>> Encoder().encode_into(Monster(name="Orc"), destination, -1)
            >>> destination
            bytearray(b'prefix:{"name":"Orc"}')
        """

        encoded = self._encode_native(value)
        if encoded is not None:
            write_encoded(buffer, encoded, offset)
            return
        self._encoder.encode_into(value, buffer, offset)

    def encode_lines(self, values: Iterable[Any]) -> bytes:
        """Encode an iterable as newline-delimited JSON.

        Args:
            values: The values to encode as separate documents.

        Returns:
            UTF-8 JSON bytes with one trailing-newline-terminated document per
            value.

        Example:
            Encode two models as JSON Lines:

            >>> Encoder().encode_lines([Monster(name="Orc"), Monster(name="Elf")])
            b'{"name":"Orc"}\n{"name":"Elf"}\n'
        """

        output = bytearray()
        for value in values:
            encoded = self._encode_native(value)
            if encoded is None:
                encoded = self._encoder.encode(value)
            output.extend(encoded)
            output.append(0x0A)
        return bytes(output)


class Decoder:
    """Decode JSON into Struct models with owned NumPy arrays.

    Args:
        type: The msgspec Struct type to decode.
        strict: Whether to reject coercions between related input types.
        dec_hook: An optional callback for otherwise unsupported target types.
        float_hook: An optional callable used to construct decoded JSON floats.

    Example:
        Reuse a decoder for one model type:

        >>> decoder = Decoder(Monster)
        >>> decoder.decode(b'{"name":"Orc"}').name
        'Orc'
    """

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
        """Decode one JSON document.

        Args:
            buffer: JSON data accepted by ``msgspec.json.Decoder``.

        Returns:
            A decoded instance of the configured Struct type.

        Raises:
            msgspec.DecodeError: The JSON is malformed.
            msgspec.ValidationError: The data does not match the target type.

        Example:
            Decode a JSON document:

            >>> Decoder(Monster).decode(b'{"name":"Orc"}')
            Monster(name='Orc')
        """

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
        """Decode newline-delimited JSON documents.

        Args:
            buffer: JSON Lines data accepted by ``msgspec.json.Decoder``.

        Returns:
            The decoded Struct instances in input order.

        Raises:
            msgspec.DecodeError: A JSON document is malformed.
            msgspec.ValidationError: A document does not match the target type.

        Example:
            Decode two JSON Lines documents:

            >>> Decoder(Monster).decode_lines(
            ...     b'{"name":"Orc"}\n{"name":"Elf"}\n'
            ... )
            [Monster(name='Orc'), Monster(name='Elf')]
        """

        return cast(list[_ModelT], self._decoder.decode_lines(buffer))


def encode(
    value: Any,
    *,
    enc_hook: Any = None,
    order: Literal["deterministic", "sorted"] | None = None,
) -> bytes:
    """Encode one value as JSON.

    Args:
        value: The value to encode.
        enc_hook: An optional callback for otherwise unsupported values.
        order: Optional deterministic or sorted ordering for maps and structs.

    Returns:
        The encoded UTF-8 JSON bytes.

    Example:
        Encode a generated model:

        >>> encode(Monster(name="Orc"))
        b'{"name":"Orc"}'
    """

    return Encoder(enc_hook=enc_hook, order=order).encode(value)


def decode(
    buffer: Any,
    *,
    type: type[_ModelT],
    strict: bool = True,
    dec_hook: Any = None,
    float_hook: Any = None,
) -> _ModelT:
    """Decode one JSON document into a Struct model.

    Args:
        buffer: JSON data accepted by ``msgspec.json.Decoder``.
        type: The msgspec Struct type to decode.
        strict: Whether to reject coercions between related input types.
        dec_hook: An optional callback for otherwise unsupported target types.
        float_hook: An optional callable used to construct decoded JSON floats.

    Returns:
        A decoded instance of ``type``.

    Raises:
        msgspec.DecodeError: The JSON is malformed.
        msgspec.ValidationError: The data does not match ``type``.

    Example:
        Decode JSON into a generated model:

        >>> decode(b'{"name":"Orc"}', type=Monster)
        Monster(name='Orc')
    """

    return Decoder(
        type,
        strict=strict,
        dec_hook=dec_hook,
        float_hook=float_hook,
    ).decode(buffer)


def register(type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Compile and cache a native JSON plan for a Struct graph.

    Registration is optional because the first encode or decode compiles the
    same plan lazily.

    Args:
        type: The root msgspec Struct type to register.

    Returns:
        Field identifiers that use msgspec's fallback path. An empty tuple means
        the graph is fully supported by the native path.

    Raises:
        TypeError: ``type`` is not a Struct type or its annotations cannot be
            represented by either path.

    Example:
        Warm the cache during application startup:

        >>> register(Monster)
        ()
    """

    return register_serde_type(type)


__all__ = ["Decoder", "Encoder", "decode", "encode", "register"]
