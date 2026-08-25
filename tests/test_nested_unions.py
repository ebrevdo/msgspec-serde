from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import flatbuffers
import msgspec
import numpy as np
import numpy.typing as npt
import pytest

from msgspec_flatbuffers import (
    BufferBoundsError,
    GenerationError,
    InvalidBufferError,
    dec_hook,
    enc_hook,
    generate,
)


SCHEMAS = Path(__file__).parent / "fixtures" / "nested_unions"
PAYLOAD_SCHEMA = SCHEMAS / "payload.fbs"
ENVELOPE_SCHEMA = SCHEMAS / "envelope.fbs"
REQUIRED_SCHEMA = SCHEMAS / "required.fbs"
HAS_FLATC = shutil.which("flatc") is not None
pytestmark = pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path.remove(value)


def _clear_generated_modules() -> None:
    for name in tuple(sys.modules):
        if name == "example" or name.startswith("example."):
            sys.modules.pop(name, None)


@pytest.fixture
def generated_modules(tmp_path: Path) -> Iterator[tuple[ModuleType, ModuleType]]:
    output = tmp_path / "generated"
    generate(PAYLOAD_SCHEMA, output, project_root=SCHEMAS)
    generate(
        ENVELOPE_SCHEMA,
        output,
        include_dirs=[SCHEMAS],
        project_root=SCHEMAS,
    )

    _clear_generated_modules()
    with _temporary_sys_path(output):
        payload = importlib.import_module("example.nested.payload")
        envelope = importlib.import_module("example.nested.envelope")
        yield payload, envelope
    _clear_generated_modules()


def _payload_model(generated: ModuleType) -> Any:
    return generated.Payload(
        favorite=generated.Cat(name="Miso", lives=8),
        residents=[
            generated.Cat(name="Miso", lives=8),
            generated.Dog(name="Pip", good=True),
            generated.Cat(name="Nori", lives=7),
        ],
        serial=17,
        measurements=np.array([0.25, 1.5, 8.0], dtype=np.float64),
    )


def _outer_with_payload(payload: bytes | bytearray | memoryview) -> bytes:
    builder = flatbuffers.Builder(max(128, len(payload) + 64))
    vector = builder.CreateByteVector(bytes(payload))
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, vector, 0)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b"ENVP")
    return bytes(memoryview(builder.Bytes)[builder.Head() :])


def _payload_with_favorite_type(tag: int) -> bytes:
    builder = flatbuffers.Builder(64)
    builder.StartObject(5)
    builder.PrependUint8Slot(0, tag, 0)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b"PAYL")
    return bytes(memoryview(builder.Bytes)[builder.Head() :])


def test_scalar_union_and_union_vector_round_trip_and_cache(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    assert generated.Payload.__annotations__["residents"] == (
        "list[Cat | Dog] | None"
    )
    model = _payload_model(generated)
    buffer = model.to_flatbuffer()
    view = generated.PayloadView.from_buffer(buffer)

    favorite = view.favorite_view()
    assert view.favorite_type == generated.Pet.Cat
    assert isinstance(favorite, generated.CatView)
    assert favorite.name == "Miso"
    assert favorite is view.favorite_view()

    residents = view.residents
    assert residents is view.residents
    assert residents is not None
    assert residents.cached_count == 0
    first = tuple(residents)
    assert isinstance(first[0], generated.CatView)
    assert isinstance(first[1], generated.DogView)
    assert isinstance(first[2], generated.CatView)
    assert residents.cached_count == 3
    second = tuple(residents)
    assert all(left is right for left, right in zip(first, second))

    resident_types = view.residents_type
    assert resident_types is view.residents_type
    assert resident_types is not None
    assert not resident_types.flags.writeable
    assert list(map(int, resident_types)) == [
        int(generated.Pet.Cat),
        int(generated.Pet.Dog),
        int(generated.Pet.Cat),
    ]

    materialized = view.to_model()
    assert materialized == model
    rebuilt = generated.PayloadView.from_buffer(materialized.to_flatbuffer())
    assert rebuilt.to_model() == model
    assert materialized.residents is not None
    materialized.residents.append(generated.Dog(name="Tess", good=True))
    mutated = generated.PayloadView.from_buffer(materialized.to_flatbuffer())
    assert mutated.to_model() == materialized


def test_union_serialization_requires_the_declared_model_types(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules

    class Impostor(msgspec.Struct, tag="Example.Nested.Cat"):
        name: str
        lives: int = 9

    model = generated.Payload(favorite=Impostor(name="not a Cat"))
    with pytest.raises(TypeError, match="union field.*favorite.*Impostor"):
        model.to_flatbuffer()


def test_absent_union_stays_none(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    model = generated.Payload()
    view = generated.PayloadView.from_buffer(model.to_flatbuffer())

    assert view.favorite_type == generated.Pet.NONE
    assert view.favorite_view() is None
    assert view.residents_type is None
    assert view.residents is None
    assert view.to_model() == model


def test_annotated_nested_flatbuffer_is_typed_cached_and_zero_copy(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    nested_model = _payload_model(payload)
    model = envelope.Envelope(payload=nested_model, note="outer")
    buffer = model.to_flatbuffer()

    assert buffer[4:8] == b"ENVP"
    outer = envelope.EnvelopeView.from_buffer(buffer)
    raw = outer.payload
    assert raw is not None
    assert raw.readonly
    assert raw.obj is buffer.obj
    assert bytes(raw)[4:8] == b"PAYL"

    nested = outer.payload_view()
    assert isinstance(nested, payload.PayloadView)
    assert nested is outer.payload_view()
    assert nested.buffer is raw
    assert nested.favorite_type == payload.Pet.Cat
    assert nested.favorite_view().name == "Miso"

    assert outer.to_model() == model
    rebuilt = envelope.EnvelopeView.from_buffer(outer.to_model().to_flatbuffer())
    assert rebuilt.to_model() == model


def test_nested_flatbuffer_target_does_not_need_to_be_the_file_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "non_root_nested.fbs"
    source.write_text(
        " ".join(
            (
                "namespace Example.NonRootNested;",
                "table Inner { value:int; }",
                "table Container { payload:[ubyte] "
                '(nested_flatbuffer: "Inner"); }',
                "root_type Container;",
            )
        ),
        encoding="utf-8",
    )
    module_path = generate(source, tmp_path / "generated", project_root=tmp_path)
    spec = importlib.util.spec_from_file_location("non_root_nested_generated", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Container(payload=module.Inner(value=42))

    view = module.ContainerView.from_buffer(model.to_flatbuffer())

    assert view.payload_view().value == 42
    assert view.to_model() == model


def test_msgspec_hooks_preserve_arrays_and_tagged_unions(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    model = envelope.Envelope(payload=_payload_model(payload), note="outer")

    marker = "__msgspec_flatbuffers_type__"
    builtins_value = msgspec.to_builtins(model, enc_hook=enc_hook)
    assert marker not in builtins_value
    payload_value = builtins_value["payload"]
    assert isinstance(payload_value, dict)
    assert marker not in payload_value
    assert payload_value["measurements"] == [0.25, 1.5, 8.0]
    favorite = payload_value["favorite"]
    assert isinstance(favorite, dict)
    assert favorite[marker] == "Example.Nested.Cat"
    residents = payload_value["residents"]
    assert isinstance(residents, list)
    assert [item[marker] for item in residents] == [
        "Example.Nested.Cat",
        "Example.Nested.Dog",
        "Example.Nested.Cat",
    ]

    from_dict = msgspec.convert(
        builtins_value,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    )
    assert from_dict == model
    assert from_dict.payload is not None
    assert from_dict.payload.measurements is not None
    assert from_dict.payload.measurements.dtype == np.dtype(np.float64)
    assert from_dict.payload.measurements.flags.owndata
    assert from_dict.payload.measurements.flags.writeable

    encoded = msgspec.json.encode(model, enc_hook=enc_hook)
    from_json = msgspec.json.decode(
        encoded,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    )
    assert from_json == model
    assert from_json.payload is not None
    assert from_json.payload.measurements is not None
    assert from_json.payload.measurements.dtype == np.dtype(np.float64)

    assert payload.__file__ is not None
    spec = importlib.util.spec_from_file_location(
        "alternate_generated_payload",
        payload.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    alternate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alternate
    try:
        spec.loader.exec_module(alternate)
        cross_import = msgspec.convert(
            payload_value,
            type=alternate.Payload,
            dec_hook=dec_hook,
        )
        assert isinstance(cross_import.favorite, alternate.Cat)
        assert cross_import.measurements is not None
        assert cross_import.measurements.dtype == np.dtype(np.float64)
    finally:
        sys.modules.pop(spec.name, None)

    empty = envelope.Envelope()
    empty_builtins = msgspec.to_builtins(empty, enc_hook=enc_hook)
    assert msgspec.convert(
        empty_builtins,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    ) == empty
    assert msgspec.json.decode(
        msgspec.json.encode(empty, enc_hook=enc_hook),
        type=envelope.Envelope,
        dec_hook=dec_hook,
    ) == empty

    unknown = copy.deepcopy(builtins_value)
    unknown["payload"]["favorite"][marker] = "unknown.Variant"
    with pytest.raises(msgspec.ValidationError, match="Invalid value"):
        msgspec.convert(
            unknown,
            type=envelope.Envelope,
            dec_hook=dec_hook,
        )

    missing = copy.deepcopy(builtins_value)
    missing["payload"]["favorite"].pop(marker)
    with pytest.raises(msgspec.ValidationError, match="missing required field"):
        msgspec.convert(
            missing,
            type=envelope.Envelope,
            dec_hook=dec_hook,
        )

    assert envelope.Envelope.__struct_config__.tag is None
    assert payload.Payload.__struct_config__.tag is None
    assert payload.Cat.__struct_config__.tag == "Example.Nested.Cat"
    assert payload.Dog.__struct_config__.tag == "Example.Nested.Dog"


def test_msgspec_hooks_restore_representative_numeric_dtypes() -> None:
    class NumericKinds(msgspec.Struct, frozen=True):
        signed: npt.NDArray[np.int16]
        unsigned: npt.NDArray[np.uint32]
        floating: npt.NDArray[np.float32]
        flags: npt.NDArray[np.bool_]

    model = NumericKinds(
        signed=np.array([-2, 3], dtype=np.int16),
        unsigned=np.array([1, 2**31], dtype=np.uint32),
        floating=np.array([0.25, 1.5], dtype=np.float32),
        flags=np.array([True, False], dtype=np.bool_),
    )

    for restored in (
        msgspec.convert(
            msgspec.to_builtins(model, enc_hook=enc_hook),
            type=NumericKinds,
            dec_hook=dec_hook,
        ),
        msgspec.json.decode(
            msgspec.json.encode(model, enc_hook=enc_hook),
            type=NumericKinds,
            dec_hook=dec_hook,
        ),
    ):
        for field_name, dtype in (
            ("signed", np.dtype(np.int16)),
            ("unsigned", np.dtype(np.uint32)),
            ("floating", np.dtype(np.float32)),
            ("flags", np.dtype(np.bool_)),
        ):
            array = getattr(restored, field_name)
            assert array.dtype == dtype
            assert array.flags.owndata
            assert array.flags.writeable
            np.testing.assert_array_equal(array, getattr(model, field_name))


def test_nested_identifiers_and_size_prefixed_payload(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    nested_model = _payload_model(payload)
    prefixed = nested_model.to_flatbuffer(size_prefixed=True)
    assert prefixed[8:12] == b"PAYL"

    outer_buffer = _outer_with_payload(prefixed)
    outer = envelope.EnvelopeView.from_buffer(outer_buffer)
    nested = outer.payload_view(size_prefixed=True)
    assert nested is outer.payload_view(size_prefixed=True)
    assert nested.serial == 17
    assert nested.favorite_view().name == "Miso"

    bad_nested = bytearray(nested_model.to_flatbuffer())
    bad_nested[4:8] = b"NOPE"
    bad_outer = envelope.EnvelopeView.from_buffer(_outer_with_payload(bad_nested))
    with pytest.raises(InvalidBufferError, match="file identifier"):
        bad_outer.payload_view()

    bad_outer_identifier = bytearray(outer_buffer)
    bad_outer_identifier[4:8] = b"NOPE"
    with pytest.raises(InvalidBufferError, match="file identifier"):
        envelope.EnvelopeView.from_buffer(bad_outer_identifier)


def test_unknown_and_missing_scalar_union_payloads_fail_lazily(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    buffer = bytearray(_payload_model(generated).to_flatbuffer())
    probe = generated.PayloadView.from_buffer(buffer)
    type_position = probe._field_position(4, 1)
    assert type_position is not None
    buffer[type_position] = 99

    unknown = generated.PayloadView.from_buffer(buffer)
    assert int(unknown.favorite_type) == 99
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown.favorite_view()
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown.to_model()

    missing = generated.PayloadView.from_buffer(
        _payload_with_favorite_type(int(generated.Pet.Cat))
    )
    assert missing.favorite_type == generated.Pet.Cat
    with pytest.raises(InvalidBufferError, match="no payload"):
        missing.favorite_view()


def test_union_vector_rejects_unknown_tags_null_offsets_and_length_mismatch(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    original = _payload_model(generated).to_flatbuffer()

    unknown_tag_buffer = bytearray(original)
    probe = generated.PayloadView.from_buffer(unknown_tag_buffer)
    type_info = probe._vector_info(8, 1)
    assert type_info is not None
    type_start, type_length = type_info
    assert type_length == 3
    unknown_tag_buffer[type_start + 1] = 99
    unknown_tag = generated.PayloadView.from_buffer(unknown_tag_buffer)
    assert unknown_tag.residents_type is not None
    assert int(unknown_tag.residents_type[1]) == 99
    assert unknown_tag.residents is not None
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown_tag.residents[1]
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown_tag.to_model()

    none_tag_buffer = bytearray(original)
    probe = generated.PayloadView.from_buffer(none_tag_buffer)
    type_info = probe._vector_info(8, 1)
    assert type_info is not None
    type_start, _ = type_info
    none_tag_buffer[type_start + 1] = int(generated.Pet.NONE)
    none_tag = generated.PayloadView.from_buffer(none_tag_buffer)
    assert none_tag.residents is not None
    with pytest.raises(InvalidBufferError, match="cannot contain NONE"):
        none_tag.residents[1]


    null_offset_buffer = bytearray(original)
    probe = generated.PayloadView.from_buffer(null_offset_buffer)
    value_info = probe._vector_info(10, 4)
    assert value_info is not None
    value_start, value_length = value_info
    assert value_length == 3
    struct.pack_into("<I", null_offset_buffer, value_start, 0)
    null_offset = generated.PayloadView.from_buffer(null_offset_buffer)
    assert null_offset.residents is not None
    with pytest.raises(InvalidBufferError, match="null offset"):
        null_offset.residents[0]

    mismatched_buffer = bytearray(original)
    probe = generated.PayloadView.from_buffer(mismatched_buffer)
    type_info = probe._vector_info(8, 1)
    assert type_info is not None
    type_start, _ = type_info
    struct.pack_into("<I", mismatched_buffer, type_start - 4, 2)
    mismatched = generated.PayloadView.from_buffer(mismatched_buffer)
    with pytest.raises(InvalidBufferError, match="lengths differ"):
        _ = mismatched.residents


@pytest.mark.parametrize(
    ("name", "schema", "message"),
    [
        (
            "alias",
            """
            namespace Unsupported;
            table Cat { name:string; }
            union Choice { Start:Cat, Finish:Cat }
            table Root { value:Choice; }
            root_type Root;
            """,
            "alias",
        ),
        (
            "string",
            """
            namespace Unsupported;
            union Choice { Text:string }
            table Root { value:Choice; }
            root_type Root;
            """,
            "string",
        ),
        (
            "struct",
            """
            namespace Unsupported;
            struct Point { x:int; }
            union Choice { Point }
            table Root { value:Choice; }
            root_type Root;
            """,
            "struct",
        ),
    ],
)
def test_unsupported_union_arms_are_rejected_during_generation(
    tmp_path: Path,
    name: str,
    schema: str,
    message: str,
) -> None:
    source = tmp_path / f"{name}.fbs"
    source.write_text(schema, encoding="utf-8")

    with pytest.raises(GenerationError, match=message):
        generate(source, tmp_path / "generated")


def test_union_alternative_short_name_collisions_are_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.fbs").write_text(
        "namespace A; table Item { a:int; }",
        encoding="utf-8",
    )
    (tmp_path / "b.fbs").write_text(
        "namespace B; table Item { b:int; }",
        encoding="utf-8",
    )
    root = tmp_path / "root.fbs"
    root.write_text(
        "\n".join(
            [
                'include "a.fbs";',
                'include "b.fbs";',
                "namespace Collision;",
                "union Choice { A.Item, B.Item }",
                "table Root { value:Choice; }",
                "root_type Root;",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        GenerationError,
        match=r"generated symbol collision.*Item",
    ):
        generate(
            root,
            tmp_path / "generated",
            include_dirs=[tmp_path],
            project_root=tmp_path,
        )


def test_nested_target_short_name_collision_with_local_symbols_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "shared.fbs").write_text(
        "\n".join(
            [
                "namespace Shared;",
                "table Item { shared:int; }",
                "root_type Item;",
            ]
        ),
        encoding="utf-8",
    )
    outer = tmp_path / "outer.fbs"
    outer.write_text(
        "\n".join(
            [
                'include "shared.fbs";',
                "namespace Outer;",
                "table Item { local:int; }",
                "table Envelope {",
                '  payload:[ubyte] (nested_flatbuffer: "Shared.Item");',
                "}",
                "root_type Envelope;",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        GenerationError,
        match=r"generated symbol collision.*Item",
    ):
        generate(
            outer,
            tmp_path / "generated",
            include_dirs=[tmp_path],
            project_root=tmp_path,
        )


def test_reserved_msgspec_tag_field_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "reserved_tag.fbs"
    source.write_text(
        " ".join(
            (
                "namespace Collision;",
                "table Root { __msgspec_flatbuffers_type__:int; }",
                "root_type Root;",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(GenerationError, match="reserved msgspec tag field"):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_generated_union_buffer_decodes_with_flatc_json(
    generated_modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    generated, _ = generated_modules
    binary = tmp_path / "wire.bin"
    binary.write_bytes(_payload_model(generated).to_flatbuffer())

    subprocess.run(
        [
            "flatc",
            "-t",
            "--strict-json",
            "--defaults-json",
            "-o",
            str(tmp_path),
            str(PAYLOAD_SCHEMA),
            "--",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = json.loads((tmp_path / "wire.json").read_text(encoding="utf-8"))

    assert decoded["favorite_type"] == "Cat"
    assert decoded["favorite"] == {"name": "Miso", "lives": 8}
    assert decoded["residents_type"] == ["Cat", "Dog", "Cat"]
    assert decoded["residents"] == [
        {"name": "Miso", "lives": 8},
        {"name": "Pip", "good": True},
        {"name": "Nori", "lives": 7},
    ]
    assert decoded["serial"] == 17
    assert decoded["measurements"] == [0.25, 1.5, 8.0]


@pytest.fixture
def required_modules(tmp_path: Path) -> Iterator[tuple[ModuleType, ModuleType]]:
    output = tmp_path / "required-generated"
    generate(PAYLOAD_SCHEMA, output, project_root=SCHEMAS)
    generate(
        REQUIRED_SCHEMA,
        output,
        include_dirs=[SCHEMAS],
        project_root=SCHEMAS,
    )

    _clear_generated_modules()
    with _temporary_sys_path(output):
        payload = importlib.import_module("example.nested.payload")
        required = importlib.import_module("example.required.required")
        yield payload, required
    _clear_generated_modules()


def _empty_required_holder() -> bytes:
    builder = flatbuffers.Builder(64)
    builder.StartObject(5)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b"REQD")
    return bytes(memoryview(builder.Bytes)[builder.Head() :])


def test_required_nested_and_union_fields_have_no_model_defaults_and_build(
    required_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, required = required_modules
    assert set(required.RequiredHolder.__annotations__) == {
        "nested",
        "pet",
        "pets",
    }
    with pytest.raises(TypeError, match="[Mm]issing required argument"):
        required.RequiredHolder()

    nested = _payload_model(payload)
    cat = payload.Cat(name="Miso", lives=8)
    dog = payload.Dog(name="Pip", good=True)
    model = required.RequiredHolder(
        nested=nested,
        pet=cat,
        pets=[cat, dog],
    )
    view = required.RequiredHolderView.from_buffer(model.to_flatbuffer())

    assert view.nested_view().to_model() == nested
    assert view.pet_type == payload.Pet.Cat
    assert view.pet_view().to_model() == cat
    assert view.pets is not None
    assert [item.to_model() for item in view.pets] == [cat, dog]
    assert view.to_model() == model


@pytest.mark.parametrize("field", ["nested", "pet", "pets"])
def test_required_model_builders_reject_runtime_none(
    required_modules: tuple[ModuleType, ModuleType],
    field: str,
) -> None:
    payload, required = required_modules
    cat = payload.Cat(name="Miso", lives=8)
    values: dict[str, Any] = {
        "nested": _payload_model(payload),
        "pet": cat,
        "pets": (cat,),
    }
    values[field] = None
    model = required.RequiredHolder(**values)

    with pytest.raises(TypeError, match=field):
        model.to_flatbuffer()


def test_required_views_reject_absent_nested_and_union_fields(
    required_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, required = required_modules

    nested = required.RequiredHolderView.from_buffer(_empty_required_holder())
    with pytest.raises(InvalidBufferError, match="required field 'nested'"):
        _ = nested.nested
    with pytest.raises(InvalidBufferError, match="required field 'nested'"):
        nested.nested_view()

    pet = required.RequiredHolderView.from_buffer(_empty_required_holder())
    assert pet.pet_type == payload.Pet.NONE
    with pytest.raises(InvalidBufferError, match="required field 'pet'"):
        pet.pet_view()

    pets = required.RequiredHolderView.from_buffer(_empty_required_holder())
    with pytest.raises(InvalidBufferError, match="required field 'pets_type'"):
        _ = pets.pets_type
    with pytest.raises(InvalidBufferError, match="required field 'pets_type'"):
        _ = pets.pets


def test_nested_payload_view_is_confined_to_byte_vector_frame(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    original = envelope.Envelope(
        payload=_payload_model(payload),
        note="adjacent outer bytes",
    ).to_flatbuffer()

    shortened = bytearray(original)
    probe = envelope.EnvelopeView.from_buffer(shortened)
    info = probe._vector_info(4, 1)
    assert info is not None
    payload_start, payload_length = info
    assert payload_length > 8
    struct.pack_into("<I", shortened, payload_start - 4, 8)
    shortened_view = envelope.EnvelopeView.from_buffer(shortened)
    assert shortened_view.payload is not None
    assert len(shortened_view.payload) == 8
    with pytest.raises(BufferBoundsError):
        shortened_view.payload_view()

    escaped_root = bytearray(original)
    probe = envelope.EnvelopeView.from_buffer(escaped_root)
    info = probe._vector_info(4, 1)
    assert info is not None
    payload_start, payload_length = info
    struct.pack_into("<I", escaped_root, payload_start, payload_length + 4)
    escaped_view = envelope.EnvelopeView.from_buffer(escaped_root)
    with pytest.raises(BufferBoundsError):
        escaped_view.payload_view()


def test_union_schema_evolution_preserves_known_values_and_rejects_unknown_arms(
    tmp_path: Path,
) -> None:
    v1_schema = tmp_path / "v1.fbs"
    v1_schema.write_text(
        "\n".join(
            [
                "namespace Evolution;",
                "table Cat { name:string (required); }",
                "union Pet { Cat }",
                "table Root { favorite:Pet; }",
                "root_type Root;",
                'file_identifier "EVOL";',
            ]
        ),
        encoding="utf-8",
    )
    v2_schema = tmp_path / "v2.fbs"
    v2_schema.write_text(
        "\n".join(
            [
                "namespace Evolution;",
                "table Cat { name:string (required); }",
                "table Dog { name:string (required); }",
                "union Pet { Cat, Dog }",
                "table Root { favorite:Pet; }",
                "root_type Root;",
                'file_identifier "EVOL";',
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evolution-generated"
    generate(v1_schema, output, project_root=tmp_path)
    generate(v2_schema, output, project_root=tmp_path)

    for name in tuple(sys.modules):
        if name == "evolution" or name.startswith("evolution."):
            sys.modules.pop(name, None)
    with _temporary_sys_path(output):
        v1 = importlib.import_module("evolution.v1")
        v2 = importlib.import_module("evolution.v2")

        old_buffer = v1.Root(favorite=v1.Cat(name="Miso")).to_flatbuffer()
        upgraded = v2.RootView.from_buffer(old_buffer)
        assert upgraded.favorite_type == v2.Pet.Cat
        assert upgraded.favorite_view().name == "Miso"
        assert upgraded.to_model() == v2.Root(favorite=v2.Cat(name="Miso"))

        new_buffer = v2.Root(favorite=v2.Dog(name="Pip")).to_flatbuffer()
        legacy = v1.RootView.from_buffer(new_buffer)
        assert int(legacy.favorite_type) == int(v2.Pet.Dog)
        assert not hasattr(legacy, "favorite")
        with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
            legacy.favorite_view()
        with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
            legacy.to_model()

    for name in tuple(sys.modules):
        if name == "evolution" or name.startswith("evolution."):
            sys.modules.pop(name, None)
