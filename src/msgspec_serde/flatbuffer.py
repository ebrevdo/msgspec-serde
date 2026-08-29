"""Encode generated models and decode models or lazy FlatBuffer views.

Example:
    Round-trip a generated model and open a lazy view over the same bytes:

    >>> from msgspec_serde import flatbuffer
    >>> model = Monster(name="Orc")
    >>> buffer = flatbuffer.encode(model)
    >>> flatbuffer.decode(buffer, type=Monster) == model
    True
    >>> flatbuffer.decode(buffer, type=MonsterView).name
    'Orc'
"""

from ._flatbuffer import Decoder, Encoder, decode, encode

__all__ = ["Decoder", "Encoder", "decode", "encode"]
