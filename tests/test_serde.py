from __future__ import annotations

import decimal
import uuid
from typing import Any

import msgspec
import numpy as np
import numpy.typing as npt
import pytest

import msgspec_flatbuffers
from msgspec_flatbuffers import json, msgpack


class NumericModel(msgspec.Struct, kw_only=True, eq=False):
    flags: npt.NDArray[np.bool_]
    int8s: npt.NDArray[np.int8]
    uint8s: npt.NDArray[np.uint8]
    int16s: npt.NDArray[np.int16]
    uint16s: npt.NDArray[np.uint16]
    int32s: npt.NDArray[np.int32]
    uint32s: npt.NDArray[np.uint32]
    int64s: npt.NDArray[np.int64]
    uint64s: npt.NDArray[np.uint64]
    float32s: npt.NDArray[np.float32]
    float64s: npt.NDArray[np.float64]
    optional: npt.NDArray[np.float64] | None = None


class Leaf(msgspec.Struct, tag="leaf", tag_field="kind", kw_only=True):
    values: npt.NDArray[np.int32]


class Branch(msgspec.Struct, tag="branch", tag_field="kind", kw_only=True):
    children: list[Leaf | Branch]


class RecursiveRoot(msgspec.Struct, kw_only=True):
    value: Leaf | Branch


class OrdinaryLeaf(msgspec.Struct, tag="leaf", tag_field="kind"):
    values: npt.NDArray[np.float64]


class OrdinaryRoot(msgspec.Struct):
    name: str
    children: list[OrdinaryLeaf]
    metadata: dict[str, int]
    payload: bytes
    counts: list[int]


class OmitDefaultsRoot(msgspec.Struct, omit_defaults=True):
    values: npt.NDArray[np.int32]
    count: int = 0


class RequiredFields(msgspec.Struct):
    count: int
    name: str
    values: list[int]


class IntegerChoice(msgspec.Struct, tag="integer", tag_field="kind"):
    value: int


class StringChoice(msgspec.Struct, tag="string", tag_field="kind"):
    value: str


class NullableUnionVector(msgspec.Struct):
    choices: list[IntegerChoice | StringChoice | None]


class TaggedUnionRoot(msgspec.Struct):
    choice: IntegerChoice | StringChoice


class IntegerModel(msgspec.Struct):
    value: int


class OptionModel(msgspec.Struct, tag="options", tag_field="kind"):
    z: int
    a: int
    identifier: uuid.UUID
    amount: decimal.Decimal


class CoercionModel(msgspec.Struct):
    integer: int
    floating: float
    boolean: bool
    values: npt.NDArray[np.int32]


class HookValue:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HookValue) and self.value == other.value


class HookModel(msgspec.Struct):
    native: int
    custom: HookValue


class AnyModel(msgspec.Struct):
    native: int
    custom: Any


def test_numpy_hooks_are_not_public() -> None:
    assert "enc_hook" not in msgspec_flatbuffers.__all__
    assert "dec_hook" not in msgspec_flatbuffers.__all__
    assert not hasattr(msgspec_flatbuffers, "enc_hook")
    assert not hasattr(msgspec_flatbuffers, "dec_hook")


def _numeric_model() -> NumericModel:
    return NumericModel(
        flags=np.array([True, False], dtype=np.bool_),
        int8s=np.array([-128, 127], dtype=np.int8),
        uint8s=np.array([0, 255], dtype=np.uint8),
        int16s=np.array([-32768, 32767], dtype=np.int16),
        uint16s=np.array([0, 65535], dtype=np.uint16),
        int32s=np.array([-(2**31), 2**31 - 1], dtype=np.int32),
        uint32s=np.array([0, 2**32 - 1], dtype=np.uint32),
        int64s=np.array([-(2**63), 2**63 - 1], dtype=np.int64),
        uint64s=np.array([0, 2**64 - 1], dtype=np.uint64),
        float32s=np.array([0.25, np.nan], dtype=np.float32),
        float64s=np.array([-1.5, 6.25], dtype=np.float64),
        optional=np.array([3.25, 4.5], dtype=np.float64),
    )


def _option_model() -> OptionModel:
    return OptionModel(
        z=2,
        a=1,
        identifier=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        amount=decimal.Decimal("1.25"),
    )


def _wire_codec(codec: Any) -> Any:
    return msgspec.json if codec is json else msgspec.msgpack


def _assert_numeric_equal(restored: NumericModel, expected: NumericModel) -> None:
    for field in msgspec.structs.fields(NumericModel):
        actual = getattr(restored, field.name)
        wanted = getattr(expected, field.name)
        if wanted is None:
            assert actual is None
            continue
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == wanted.dtype
        assert actual.flags.writeable
        assert actual.flags.c_contiguous
        # Rust Vec ownership is attached through NumPy's base owner, so NumPy
        # may report OWNDATA=False even though the result doesn't borrow input.
        assert actual.flags.owndata or actual.base is not None
        np.testing.assert_equal(actual, wanted)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_native_codecs_round_trip_all_numeric_dtypes(codec: Any) -> None:
    model = _numeric_model()

    encoded = codec.Encoder().encode(model)
    restored = codec.Decoder(NumericModel).decode(encoded)

    _assert_numeric_equal(restored, model)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_native_decoders_accept_regular_msgspec_array_wires(codec: Any) -> None:
    model = _numeric_model()
    encoded = _wire_codec(codec).encode(
        {
            field.name: (
                None
                if (value := getattr(model, field.name)) is None
                else value.tolist()
            )
            for field in msgspec.structs.fields(NumericModel)
        }
    )

    restored = codec.decode(encoded, type=NumericModel)

    _assert_numeric_equal(restored, model)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_native_codecs_preserve_recursive_tagged_union_vectors(codec: Any) -> None:
    model = RecursiveRoot(
        value=Branch(
            children=[
                Leaf(values=np.array([1, 2], dtype=np.int32)),
                Branch(children=[Leaf(values=np.array([3, 4], dtype=np.int32))]),
            ]
        )
    )

    restored = codec.decode(codec.encode(model), type=RecursiveRoot)

    assert isinstance(restored.value, Branch)
    assert isinstance(restored.value.children[0], Leaf)
    np.testing.assert_array_equal(restored.value.children[0].values, [1, 2])
    nested = restored.value.children[1]
    assert isinstance(nested, Branch)
    assert isinstance(nested.children[0], Leaf)
    np.testing.assert_array_equal(nested.children[0].values, [3, 4])


def test_deep_json_graph_falls_back_after_native_depth_limit() -> None:
    value: Leaf | Branch = Leaf(values=np.array([1], dtype=np.int32))
    for _ in range(130):
        value = Branch(children=[value])

    restored = json.decode(json.encode(RecursiveRoot(value=value)), type=RecursiveRoot)

    value = restored.value
    for _ in range(130):
        assert isinstance(value, Branch)
        value = value.children[0]
    assert isinstance(value, Leaf)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_optional_array_accepts_missing_and_explicit_null(codec: Any) -> None:
    model = _numeric_model()
    model.optional = None
    encoded = codec.encode(model)

    restored = codec.decode(encoded, type=NumericModel)

    assert restored.optional is None


@pytest.mark.parametrize("codec", [json, msgpack])
@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("count", "expected an integer"),
        ("name", "expected a string"),
        ("values", "Expected `array`"),
    ],
)
def test_native_decoder_rejects_null_for_nonnullable_fields(
    codec: Any,
    field: str,
    expected_message: str,
) -> None:
    values = {"count": 1, "name": "required", "values": [1, 2]}
    values[field] = None

    with pytest.raises(ValueError, match=expected_message):
        codec.decode(_wire_codec(codec).encode(values), type=RequiredFields)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_nullable_tagged_union_vector_elements_round_trip(codec: Any) -> None:
    model = NullableUnionVector([IntegerChoice(1), None, StringChoice("two")])

    restored = codec.decode(codec.encode(model), type=NullableUnionVector)

    assert restored == model


@pytest.mark.parametrize("codec", [json, msgpack])
def test_tagged_union_accepts_fields_before_discriminator(codec: Any) -> None:
    encoded = _wire_codec(codec).encode(
        {"choice": {"value": "before tag", "kind": "string"}}
    )

    restored = codec.decode(encoded, type=TaggedUnionRoot)

    assert restored == TaggedUnionRoot(StringChoice("before tag"))


@pytest.mark.parametrize("codec", [json, msgpack])
@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_native_codecs_preserve_unsigned_python_ints(
    codec: Any,
    value: int,
) -> None:
    model = IntegerModel(value)

    assert codec.decode(codec.encode(model), type=IntegerModel) == model


@pytest.mark.parametrize("value", [2**128, -(2**128)])
def test_native_json_preserves_arbitrary_python_ints(value: int) -> None:
    model = IntegerModel(value)

    encoded = json.encode(model)

    assert str(value).encode() in encoded
    assert json.decode(encoded, type=IntegerModel) == model


@pytest.mark.parametrize("codec", [json, msgpack])
def test_native_int_decoder_remains_strict(codec: Any) -> None:
    with pytest.raises(ValueError, match="expected an integer"):
        codec.decode(
            _wire_codec(codec).encode({"value": "123"}),
            type=IntegerModel,
        )


@pytest.mark.parametrize("codec", [json, msgpack])
def test_strict_false_coerces_native_scalars_and_arrays(codec: Any) -> None:
    encoded = _wire_codec(codec).encode(
        {
            "integer": "12",
            "floating": "1.5",
            "boolean": "true",
            "values": ["1", 2.0],
        }
    )

    restored = codec.decode(encoded, type=CoercionModel, strict=False)

    assert restored.integer == 12
    assert restored.floating == 1.5
    assert restored.boolean is True
    np.testing.assert_array_equal(restored.values, [1, 2])


@pytest.mark.parametrize("codec", [json, msgpack])
@pytest.mark.parametrize("order", ["deterministic", "sorted"])
def test_native_order_matches_msgspec(codec: Any, order: str) -> None:
    model = _option_model()
    native_encoder = codec.Encoder(order=order)
    msgspec_encoder = _wire_codec(codec).Encoder(order=order)

    assert native_encoder.encode(model) == msgspec_encoder.encode(model)


@pytest.mark.parametrize(
    ("codec", "uuid_format"),
    [
        (json, "canonical"),
        (json, "hex"),
        (msgpack, "canonical"),
        (msgpack, "hex"),
        (msgpack, "bytes"),
    ],
)
@pytest.mark.parametrize("decimal_format", ["string", "number"])
def test_native_uuid_and_decimal_formats_match_msgspec(
    codec: Any,
    uuid_format: str,
    decimal_format: str,
) -> None:
    model = _option_model()
    assert codec.register(OptionModel) == ()

    native_encoder = codec.Encoder(
        decimal_format=decimal_format,
        uuid_format=uuid_format,
    )
    msgspec_encoder = _wire_codec(codec).Encoder(
        decimal_format=decimal_format,
        uuid_format=uuid_format,
    )
    encoded = native_encoder.encode(model)

    assert encoded == msgspec_encoder.encode(model)
    assert codec.decode(encoded, type=OptionModel) == model


@pytest.mark.parametrize("codec", [json, msgpack])
def test_custom_hooks_only_handle_fallback_fields(
    codec: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_values: list[str] = []
    decoded_values: list[str] = []
    native_encode = codec.encode_native_model
    native_decode = codec.decode_native_model
    native_encoded = False
    native_decoded = False

    def tracked_native_encode(value: Any, **options: Any) -> bytes | None:
        nonlocal native_encoded
        result = native_encode(value, **options)
        native_encoded = result is not None
        return result

    def tracked_native_decode(*args: Any, **options: Any) -> Any:
        nonlocal native_decoded
        result = native_decode(*args, **options)
        native_decoded = result is not None
        return result

    def encode_hook(value: Any) -> Any:
        if isinstance(value, HookValue):
            encoded_values.append(value.value)
            return value.value
        raise NotImplementedError

    def decode_hook(annotation: Any, value: Any) -> Any:
        if annotation is HookValue:
            decoded_values.append(value)
            return HookValue(value)
        raise NotImplementedError

    monkeypatch.setattr(codec, "encode_native_model", tracked_native_encode)
    monkeypatch.setattr(codec, "decode_native_model", tracked_native_decode)
    model = HookModel(7, HookValue("custom"))
    encoded = codec.Encoder(enc_hook=encode_hook).encode(model)
    restored = codec.Decoder(HookModel, dec_hook=decode_hook).decode(encoded)

    assert restored == model
    assert encoded_values == ["custom"]
    assert decoded_values == ["custom"]
    assert native_encoded
    assert native_decoded


def test_messagepack_ext_hook_only_handles_fallback_field() -> None:
    model = AnyModel(7, msgspec.msgpack.Ext(3, b"payload"))
    calls: list[tuple[int, bytes]] = []

    def ext_hook(code: int, data: memoryview) -> Any:
        calls.append((code, bytes(data)))
        return (code, bytes(data))

    encoded = msgpack.encode(model)
    restored = msgpack.decode(encoded, type=AnyModel, ext_hook=ext_hook)

    assert restored.native == 7
    assert restored.custom == (3, b"payload")
    assert calls == [(3, b"payload")]


def test_json_float_hook_only_handles_fallback_field() -> None:
    encoded = msgspec.json.encode({"native": 7, "custom": 1.25})

    restored = json.decode(encoded, type=AnyModel, float_hook=decimal.Decimal)

    assert restored.native == 7
    assert restored.custom == decimal.Decimal("1.25")


def test_json_line_codec_uses_native_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _numeric_model()
    encoder = json.Encoder()
    decoder = json.Decoder(NumericModel)
    native_encode = json.encode_native_model
    native_calls = 0

    def tracked_native_encode(value: Any, **options: Any) -> bytes | None:
        nonlocal native_calls
        native_calls += 1
        return native_encode(value, **options)

    monkeypatch.setattr(json, "encode_native_model", tracked_native_encode)

    encoded = encoder.encode_lines([model, model])
    restored = decoder.decode_lines(encoded)

    assert native_calls == 2
    assert len(restored) == 2
    _assert_numeric_equal(restored[0], model)
    _assert_numeric_equal(restored[1], model)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_encoder_supports_noncontiguous_one_dimensional_arrays(codec: Any) -> None:
    model = _numeric_model()
    model.int32s = np.arange(10, dtype=np.int32)[::2]

    restored = codec.decode(codec.encode(model), type=NumericModel)

    np.testing.assert_array_equal(restored.int32s, [0, 2, 4, 6, 8])


@pytest.mark.parametrize("codec", [json, msgpack])
def test_encoder_rejects_invalid_array_shape(codec: Any) -> None:
    model = _numeric_model()
    model.int32s = np.zeros((2, 2), dtype=np.int32)

    with pytest.raises(TypeError, match="one-dimensional"):
        codec.encode(model)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_ordinary_struct_graph_uses_field_level_fallback(codec: Any) -> None:
    model = OrdinaryRoot(
        "root",
        [OrdinaryLeaf(np.array([1.25, 2.5], dtype=np.float64))],
        {"attempts": 3},
        b"\x00\x01\x02",
        [2, 4, 8],
    )

    fallbacks = codec.register(OrdinaryRoot)
    encoded = codec.encode(model)
    restored = codec.decode(encoded, type=OrdinaryRoot)

    assert {fallback.rpartition(":")[2] for fallback in fallbacks} == {
        "counts",
        "metadata",
        "payload",
    }
    assert restored.name == model.name
    assert restored.metadata == model.metadata
    assert restored.payload == model.payload
    assert restored.counts == model.counts
    assert isinstance(restored.children[0], OrdinaryLeaf)
    np.testing.assert_array_equal(restored.children[0].values, [1.25, 2.5])


def test_json_fallback_fields_preserve_large_integers() -> None:
    large = 2**64
    model = OrdinaryRoot("root", [], {"large": large}, b"", [])

    encoded = json.encode(model)
    restored = json.decode(encoded, type=OrdinaryRoot)

    assert str(large).encode() in encoded
    assert restored.metadata == {"large": large}


@pytest.mark.parametrize("codec", [json, msgpack])
def test_ordinary_struct_graph_auto_registers(codec: Any) -> None:
    class AutoModel(msgspec.Struct):
        values: npt.NDArray[np.int32]

    model = AutoModel(np.array([1, 2, 3], dtype=np.int32))

    restored = codec.decode(codec.encode(model), type=AutoModel)

    np.testing.assert_array_equal(restored.values, model.values)


@pytest.mark.parametrize("codec", [json, msgpack])
def test_omit_defaults_uses_native_plan(codec: Any) -> None:
    model = OmitDefaultsRoot(np.array([1, 2], dtype=np.int32))

    assert codec.register(OmitDefaultsRoot) == ()
    encoded = codec.encode(model)
    untyped = _wire_codec(codec).decode(encoded)
    restored = codec.decode(encoded, type=OmitDefaultsRoot)

    assert "count" not in untyped
    assert restored.count == 0
    np.testing.assert_array_equal(restored.values, model.values)

    nondefault = OmitDefaultsRoot(model.values, count=2)
    assert _wire_codec(codec).decode(codec.encode(nondefault))["count"] == 2
