"""Encode and decode MessagePack with direct NumPy array support.

Example:
    Round-trip a model through MessagePack:

    >>> from msgspec_serde import msgpack
    >>> encoded = msgpack.encode(Monster(name="Orc"))
    >>> msgpack.decode(encoded, type=Monster).name
    'Orc'
"""

from __future__ import annotations

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
    """Encode Struct models as MessagePack without NumPy-to-list conversion.

    Args:
        enc_hook: An optional callback for otherwise unsupported values.
        decimal_format: Whether to encode decimals as strings or numbers.
        uuid_format: Whether to encode UUIDs in canonical, hexadecimal, or byte
            form.
        order: Optional deterministic or sorted ordering for maps and structs.

    Example:
        Reuse a deterministically ordered encoder:

        >>> encoder = Encoder(order="deterministic")
        >>> isinstance(encoder.encode(Monster(name="Orc")), bytes)
        True
    """

    __slots__ = ("_decimal_format", "_encoder", "_order", "_uuid_format")

    def __init__(
        self,
        *,
        enc_hook: Any = None,
        decimal_format: Literal["string", "number"] = "string",
        uuid_format: Literal["canonical", "hex", "bytes"] = "canonical",
        order: Literal["deterministic", "sorted"] | None = None,
    ) -> None:
        self._encoder = msgspec.msgpack.Encoder(
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
            is_json=False,
            fallback_encoder=self._encoder.encode,
            order=self._order,
            decimal_format=self._decimal_format,
            uuid_format=self._uuid_format,
        )

    def encode(self, value: Any) -> bytes:
        """Encode one value as MessagePack.

        Args:
            value: The value to encode.

        Returns:
            The encoded MessagePack bytes.

        Example:
            Encode a model with a reusable encoder:

            >>> packed = Encoder().encode(Monster(name="Orc"))
            >>> isinstance(packed, bytes)
            True
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
            >>> destination.startswith(b"prefix:")
            True
        """

        encoded = self._encode_native(value)
        if encoded is not None:
            write_encoded(buffer, encoded, offset)
            return
        self._encoder.encode_into(value, buffer, offset)


class Decoder:
    """Decode MessagePack into Struct models with owned NumPy arrays.

    Args:
        type: The msgspec Struct type to decode.
        strict: Whether to reject coercions between related input types.
        dec_hook: An optional callback for otherwise unsupported target types.
        ext_hook: An optional callback for MessagePack extension values.

    Example:
        Reuse a decoder for one model type:

        >>> decoder = Decoder(Monster)
        >>> decoder.decode(encoded).name
        'Orc'
    """

    __slots__ = ("_decoder", "_fallback_decoders", "_strict", "_target_type")

    def __init__(
        self,
        type: type[_ModelT],
        *,
        strict: bool = True,
        dec_hook: Any = None,
        ext_hook: Any = None,
    ) -> None:
        self._target_type = type
        combined_hook = make_dec_hook(dec_hook)
        self._decoder = msgspec.msgpack.Decoder(
            type=type,
            strict=strict,
            dec_hook=combined_hook,
            ext_hook=ext_hook,
        )
        self._strict = strict
        self._fallback_decoders = {
            callback_id: msgspec.msgpack.Decoder(
                type=annotation,
                strict=strict,
                dec_hook=combined_hook,
                ext_hook=ext_hook,
            ).decode
            for callback_id, annotation in serde_fallback_annotations(type)
        }

    def decode(self, buffer: Any) -> _ModelT:
        """Decode one MessagePack document.

        Args:
            buffer: MessagePack data accepted by ``msgspec.msgpack.Decoder``.

        Returns:
            A decoded instance of the configured Struct type.

        Raises:
            msgspec.DecodeError: The MessagePack data is malformed.
            msgspec.ValidationError: The data does not match the target type.

        Example:
            Decode a MessagePack document:

            >>> packed = Encoder().encode(Monster(name="Orc"))
            >>> Decoder(Monster).decode(packed)
            Monster(name='Orc')
        """

        decoded = decode_native_model(
            buffer,
            self._target_type,
            is_json=False,
            strict=self._strict,
            fallback_decoders=self._fallback_decoders,
        )
        if decoded is not None:
            return cast(_ModelT, decoded)
        return cast(_ModelT, self._decoder.decode(buffer))


def encode(
    value: Any,
    *,
    enc_hook: Any = None,
    order: Literal["deterministic", "sorted"] | None = None,
) -> bytes:
    """Encode one value as MessagePack.

    Args:
        value: The value to encode.
        enc_hook: An optional callback for otherwise unsupported values.
        order: Optional deterministic or sorted ordering for maps and structs.

    Returns:
        The encoded MessagePack bytes.

    Example:
        Encode a generated model:

        >>> isinstance(encode(Monster(name="Orc")), bytes)
        True
    """

    return Encoder(enc_hook=enc_hook, order=order).encode(value)


def decode(
    buffer: Any,
    *,
    type: type[_ModelT],
    strict: bool = True,
    dec_hook: Any = None,
    ext_hook: Any = None,
) -> _ModelT:
    """Decode one MessagePack document into a Struct model.

    Args:
        buffer: MessagePack data accepted by ``msgspec.msgpack.Decoder``.
        type: The msgspec Struct type to decode.
        strict: Whether to reject coercions between related input types.
        dec_hook: An optional callback for otherwise unsupported target types.
        ext_hook: An optional callback for MessagePack extension values.

    Returns:
        A decoded instance of ``type``.

    Raises:
        msgspec.DecodeError: The MessagePack data is malformed.
        msgspec.ValidationError: The data does not match ``type``.

    Example:
        Decode MessagePack into a generated model:

        >>> packed = encode(Monster(name="Orc"))
        >>> decode(packed, type=Monster)
        Monster(name='Orc')
    """

    return Decoder(
        type,
        strict=strict,
        dec_hook=dec_hook,
        ext_hook=ext_hook,
    ).decode(buffer)


def register(type: type[msgspec.Struct]) -> tuple[str, ...]:
    """Compile and cache a native MessagePack plan for a Struct graph.

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
