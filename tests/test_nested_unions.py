from __future__ import annotations

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
import pytest

from msgspec_flatbuffers import (
    BufferBoundsError,
    GenerationError,
    InvalidBufferError,
    flatbuffer,
    generate,
)
from msgspec_flatbuffers import json as generated_json
from msgspec_flatbuffers import msgpack as generated_msgpack

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


def _clear_generated_modules(package: str = "example") -> None:
    for name in tuple(sys.modules):
        if name == package or name.startswith(f"{package}."):
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


def _require_vector_info(
    view: Any,
    field_offset: int,
    item_size: int,
) -> tuple[int, int]:
    info = view._vector_info(field_offset, item_size)
    assert info is not None
    return info


def test_scalar_union_and_union_vector_round_trip_and_cache(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    residents_annotation = generated.Payload.__annotations__["residents"]
    assert residents_annotation.startswith("list[")
    assert ".Cat" in residents_annotation and ".Dog" in residents_annotation
    assert residents_annotation.endswith(" | None] | None")
    model = _payload_model(generated)
    buffer = flatbuffer.encode(model)
    view = flatbuffer.decode(buffer, type=generated.PayloadView)
    assert flatbuffer.decode(buffer, type=generated.Payload) == model

    favorite = view.favorite
    assert view.favorite_type == generated.Pet.Cat
    assert isinstance(favorite, generated.CatView)
    assert favorite.name == "Miso"
    assert favorite is view.favorite

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
    rebuilt = flatbuffer.decode(
        flatbuffer.encode(materialized), type=generated.PayloadView
    )
    assert rebuilt.to_model() == model
    assert materialized.residents is not None
    materialized.residents.append(generated.Dog(name="Tess", good=True))
    mutated = flatbuffer.decode(
        flatbuffer.encode(materialized), type=generated.PayloadView
    )
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
        flatbuffer.encode(model)


def test_union_annotations_select_path_specific_model_subclasses(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    namespace = {
        "Cat": generated.Cat,
        "Dog": generated.Dog,
        "Payload": generated.Payload,
    }
    exec(
        compile(
            """
class FavoriteCat(Cat, dict=True):
    def __post_init__(self):
        self.role = "favorite"

class ResidentCat(Cat, dict=True):
    def __post_init__(self):
        self.role = "resident"

class ValidatedPayload(Payload, dict=True):
    favorite: FavoriteCat | Dog | None
    residents: list[ResidentCat | Dog] | None
""",
            "<union-model-subclasses>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    favorite_cat = namespace["FavoriteCat"]
    resident_cat = namespace["ResidentCat"]
    validated_payload = namespace["ValidatedPayload"]
    model = generated.Payload(
        favorite=generated.Cat(name="Miso"),
        residents=[generated.Cat(name="Nori"), generated.Dog(name="Pip")],
    )
    buffer = flatbuffer.encode(model)

    direct = flatbuffer.decode(buffer, type=validated_payload)
    from_view = flatbuffer.decode(buffer, type=generated.PayloadView).to_model(
        validated_payload
    )
    for restored in (direct, from_view):
        assert type(restored) is validated_payload
        assert type(restored.favorite) is favorite_cat
        assert restored.favorite.role == "favorite"
        assert restored.residents is not None
        assert type(restored.residents[0]) is resident_cat
        assert restored.residents[0].role == "resident"
        assert flatbuffer.encode(restored).readonly


def test_absent_union_stays_none(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    model = generated.Payload()
    view = flatbuffer.decode(flatbuffer.encode(model), type=generated.PayloadView)

    assert view.favorite_type == generated.Pet.NONE
    assert view.favorite is None
    assert view.residents_type is None
    assert view.residents is None
    assert view.to_model() == model


def test_annotated_nested_flatbuffer_is_typed_cached_and_zero_copy(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    nested_model = _payload_model(payload)
    model = envelope.Envelope(payload=nested_model, note="outer")
    buffer = flatbuffer.encode(model)

    assert buffer[4:8] == b"ENVP"
    outer = flatbuffer.decode(buffer, type=envelope.EnvelopeView)
    assert flatbuffer.decode(buffer, type=envelope.Envelope) == model
    raw = outer.payload_raw
    assert raw is not None
    assert raw.readonly
    assert raw.obj is buffer.obj
    assert bytes(raw)[4:8] == b"PAYL"

    nested = outer.payload
    assert isinstance(nested, payload.PayloadView)
    assert nested is outer.payload
    assert nested.buffer is raw
    assert nested.favorite_type == payload.Pet.Cat
    assert nested.favorite.name == "Miso"

    assert outer.to_model() == model
    rebuilt = flatbuffer.decode(
        flatbuffer.encode(outer.to_model()), type=envelope.EnvelopeView
    )
    assert rebuilt.to_model() == model


def test_nested_flatbuffer_annotations_select_model_subclasses(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    namespace: dict[str, Any] = {
        "Envelope": envelope.Envelope,
        "Payload": payload.Payload,
    }
    exec(
        compile(
            """
class ValidatedPayload(Payload, dict=True):
    def __post_init__(self):
        self.was_validated = True

class ValidatedEnvelope(Envelope, dict=True):
    payload: ValidatedPayload | None
""",
            "<nested-model-subclasses>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    validated_payload = namespace["ValidatedPayload"]
    validated_envelope = namespace["ValidatedEnvelope"]
    model = envelope.Envelope(payload=_payload_model(payload), note="outer")
    buffer = flatbuffer.encode(model)

    direct = flatbuffer.decode(buffer, type=validated_envelope)
    from_view = flatbuffer.decode(buffer, type=envelope.EnvelopeView).to_model(
        validated_envelope
    )
    for restored in (direct, from_view):
        assert type(restored) is validated_envelope
        assert type(restored.payload) is validated_payload
        assert restored.payload.was_validated


def test_nested_flatbuffer_target_does_not_need_to_be_the_file_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "non_root_nested.fbs"
    source.write_text(
        " ".join(
            (
                "namespace Example.NonRootNested;",
                "table Inner { value:int; }",
                'table Container { payload:[ubyte] (nested_flatbuffer: "Inner"); }',
                "root_type Container;",
            )
        ),
        encoding="utf-8",
    )
    module_path = generate(source, tmp_path / "generated", project_root=tmp_path)
    output = tmp_path / "generated"
    with _temporary_sys_path(output):
        module = importlib.import_module("example.non_root_nested.container")
        model = module.Container(payload=module.Inner(value=42))

        view = flatbuffer.decode(flatbuffer.encode(model), type=module.ContainerView)

        assert module_path == output / "example" / "non_root_nested" / "container.py"
        assert view.payload.value == 42
        assert view.to_model() == model


@pytest.mark.parametrize("codec", [generated_json, generated_msgpack])
def test_native_codecs_preserve_generated_nested_models_and_unions(
    generated_modules: tuple[ModuleType, ModuleType],
    codec: Any,
) -> None:
    payload, envelope = generated_modules
    model = envelope.Envelope(payload=_payload_model(payload), note="outer")

    restored = codec.decode(codec.encode(model), type=envelope.Envelope)

    assert restored == model
    assert restored.payload is not None
    assert restored.payload.measurements is not None
    assert restored.payload.measurements.dtype == np.dtype(np.float64)


def test_nested_identifiers_and_size_prefixed_payload(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    nested_model = _payload_model(payload)
    prefixed = flatbuffer.encode(nested_model, size_prefixed=True)
    assert prefixed[8:12] == b"PAYL"

    outer_buffer = _outer_with_payload(prefixed)
    outer = flatbuffer.decode(outer_buffer, type=envelope.EnvelopeView)
    nested = outer.payload_view(size_prefixed=True)
    assert nested is outer.payload_view(size_prefixed=True)
    assert nested.serial == 17
    assert nested.favorite.name == "Miso"

    bad_nested = bytearray(flatbuffer.encode(nested_model))
    bad_nested[4:8] = b"NOPE"
    bad_outer = flatbuffer.decode(
        _outer_with_payload(bad_nested), type=envelope.EnvelopeView
    )
    with pytest.raises(InvalidBufferError, match="file identifier"):
        _ = bad_outer.payload

    bad_outer_identifier = bytearray(outer_buffer)
    bad_outer_identifier[4:8] = b"NOPE"
    with pytest.raises(InvalidBufferError, match="file identifier"):
        flatbuffer.decode(bad_outer_identifier, type=envelope.EnvelopeView)


def test_unknown_and_missing_scalar_union_payloads_fail_lazily(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    buffer = bytearray(flatbuffer.encode(_payload_model(generated)))
    probe = flatbuffer.decode(buffer, type=generated.PayloadView)
    type_position = probe._field_position(4, 1)
    assert type_position is not None
    buffer[type_position] = 99

    unknown = flatbuffer.decode(buffer, type=generated.PayloadView)
    assert int(unknown.favorite_type) == 99
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        _ = unknown.favorite
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown.to_model()

    missing = flatbuffer.decode(
        _payload_with_favorite_type(int(generated.Pet.Cat)),
        type=generated.PayloadView,
    )
    assert missing.favorite_type == generated.Pet.Cat
    with pytest.raises(InvalidBufferError, match="no payload"):
        _ = missing.favorite


def test_union_vector_rejects_unknown_tags_null_offsets_and_length_mismatch(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    generated, _ = generated_modules
    original = flatbuffer.encode(_payload_model(generated))

    unknown_tag_buffer = bytearray(original)
    probe = flatbuffer.decode(unknown_tag_buffer, type=generated.PayloadView)
    type_start, type_length = _require_vector_info(probe, 8, 1)
    assert type_length == 3
    unknown_tag_buffer[type_start + 1] = 99
    unknown_tag = flatbuffer.decode(unknown_tag_buffer, type=generated.PayloadView)
    assert unknown_tag.residents_type is not None
    assert int(unknown_tag.residents_type[1]) == 99
    assert unknown_tag.residents is not None
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown_tag.residents[1]
    with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
        unknown_tag.to_model()

    none_tag_buffer = bytearray(original)
    probe = flatbuffer.decode(none_tag_buffer, type=generated.PayloadView)
    type_start, _ = _require_vector_info(probe, 8, 1)
    none_tag_buffer[type_start + 1] = int(generated.Pet.NONE)
    none_tag = flatbuffer.decode(none_tag_buffer, type=generated.PayloadView)
    assert none_tag.residents is not None
    with pytest.raises(InvalidBufferError, match="NONE.*has a payload"):
        none_tag.residents[1]

    null_offset_buffer = bytearray(original)
    probe = flatbuffer.decode(null_offset_buffer, type=generated.PayloadView)
    value_start, value_length = _require_vector_info(probe, 10, 4)
    assert value_length == 3
    struct.pack_into("<I", null_offset_buffer, value_start, 0)
    null_offset = flatbuffer.decode(null_offset_buffer, type=generated.PayloadView)
    assert null_offset.residents is not None
    with pytest.raises(InvalidBufferError, match="null offset"):
        null_offset.residents[0]

    mismatched_buffer = bytearray(original)
    probe = flatbuffer.decode(mismatched_buffer, type=generated.PayloadView)
    type_start, _ = _require_vector_info(probe, 8, 1)
    struct.pack_into("<I", mismatched_buffer, type_start - 4, 2)
    mismatched = flatbuffer.decode(mismatched_buffer, type=generated.PayloadView)
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


def test_union_alternative_short_name_collisions_are_qualified(
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

    output = tmp_path / "generated"
    generate(tmp_path / "a.fbs", output, project_root=tmp_path)
    generate(tmp_path / "b.fbs", output, project_root=tmp_path)
    generate(
        root,
        output,
        include_dirs=[tmp_path],
        project_root=tmp_path,
    )

    with _temporary_sys_path(output):
        a = importlib.import_module("a.item")
        b = importlib.import_module("b.item")
        generated = importlib.import_module("collision.root")

        a_model = generated.Root(value=a.Item(a=1))
        b_model = generated.Root(value=b.Item(b=2))

        assert (
            flatbuffer.decode(flatbuffer.encode(a_model), type=generated.Root)
            == a_model
        )
        assert (
            flatbuffer.decode(flatbuffer.encode(b_model), type=generated.Root)
            == b_model
        )


def test_nested_target_short_name_collision_uses_qualified_type(
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
    output = tmp_path / "generated"
    generate(tmp_path / "shared.fbs", output, project_root=tmp_path)
    generate(
        outer,
        output,
        include_dirs=[tmp_path],
        project_root=tmp_path,
    )

    _clear_generated_modules("shared")
    _clear_generated_modules("outer")
    with _temporary_sys_path(output):
        shared = importlib.import_module("shared.item")
        generated = importlib.import_module("outer.envelope")
        model = generated.Envelope(payload=shared.Item(shared=42))

        assert (
            flatbuffer.decode(flatbuffer.encode(model), type=generated.Envelope)
            == model
        )
    _clear_generated_modules("shared")
    _clear_generated_modules("outer")


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


def test_nested_raw_property_name_collisions_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "raw_property_collision.fbs"
    source.write_text(
        " ".join(
            (
                "table Payload { value:int; }",
                "table Envelope { payload:[ubyte] ",
                '(nested_flatbuffer: "Payload"); payload_raw:int; }',
                "root_type Envelope;",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(GenerationError, match="raw property.*conflicts"):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_generated_union_buffer_decodes_with_flatc_json(
    generated_modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    generated, _ = generated_modules
    binary = tmp_path / "wire.bin"
    binary.write_bytes(flatbuffer.encode(_payload_model(generated)))

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
        required = importlib.import_module("example.required.required_holder")
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
    buffer = flatbuffer.encode(model)
    view = flatbuffer.decode(buffer, type=required.RequiredHolderView)
    assert flatbuffer.decode(buffer, type=required.RequiredHolder) == model

    assert view.nested.to_model() == nested
    assert view.pet_type == payload.Pet.Cat
    assert view.pet.to_model() == cat
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
        flatbuffer.encode(model)


def test_required_views_reject_absent_nested_and_union_fields(
    required_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, required = required_modules
    empty_holder = _empty_required_holder()

    nested = flatbuffer.decode(empty_holder, type=required.RequiredHolderView)
    with pytest.raises(InvalidBufferError, match="required field 'nested'"):
        _ = nested.nested

    pet = flatbuffer.decode(empty_holder, type=required.RequiredHolderView)
    assert pet.pet_type == payload.Pet.NONE
    with pytest.raises(InvalidBufferError, match="required field 'pet'"):
        _ = pet.pet

    pets = flatbuffer.decode(empty_holder, type=required.RequiredHolderView)
    with pytest.raises(InvalidBufferError, match="required field 'pets_type'"):
        _ = pets.pets_type
    with pytest.raises(InvalidBufferError, match="required field 'pets_type'"):
        _ = pets.pets


def test_nested_payload_view_is_confined_to_byte_vector_frame(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    original = flatbuffer.encode(
        envelope.Envelope(
            payload=_payload_model(payload),
            note="adjacent outer bytes",
        )
    )

    shortened = bytearray(original)
    probe = flatbuffer.decode(shortened, type=envelope.EnvelopeView)
    payload_start, payload_length = _require_vector_info(probe, 4, 1)
    assert payload_length > 8
    struct.pack_into("<I", shortened, payload_start - 4, 8)
    shortened_view = flatbuffer.decode(shortened, type=envelope.EnvelopeView)
    raw = shortened_view.payload_raw
    assert raw is not None
    assert len(raw) == 8
    with pytest.raises(BufferBoundsError, match="table header"):
        _ = shortened_view.payload

    escaped_root = bytearray(original)
    probe = flatbuffer.decode(escaped_root, type=envelope.EnvelopeView)
    payload_start, payload_length = _require_vector_info(probe, 4, 1)
    struct.pack_into("<I", escaped_root, payload_start, payload_length + 4)
    escaped_view = flatbuffer.decode(escaped_root, type=envelope.EnvelopeView)
    with pytest.raises(BufferBoundsError, match="table header"):
        _ = escaped_view.payload


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
    generate(
        v1_schema,
        output,
        project_root=tmp_path,
        package="evolution",
        gen_onefile=True,
    )
    generate(
        v2_schema,
        output,
        project_root=tmp_path,
        package="evolution",
        gen_onefile=True,
    )

    _clear_generated_modules("evolution")
    with _temporary_sys_path(output):
        v1 = importlib.import_module("evolution.v1")
        v2 = importlib.import_module("evolution.v2")

        old_buffer = flatbuffer.encode(v1.Root(favorite=v1.Cat(name="Miso")))
        upgraded = flatbuffer.decode(old_buffer, type=v2.RootView)
        assert upgraded.favorite_type == v2.Pet.Cat
        assert upgraded.favorite.name == "Miso"
        assert upgraded.to_model() == v2.Root(favorite=v2.Cat(name="Miso"))

        new_buffer = flatbuffer.encode(v2.Root(favorite=v2.Dog(name="Pip")))
        legacy = flatbuffer.decode(new_buffer, type=v1.RootView)
        assert int(legacy.favorite_type) == int(v2.Pet.Dog)
        with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
            _ = legacy.favorite
        with pytest.raises(InvalidBufferError, match="unknown .* discriminator"):
            legacy.to_model()

    _clear_generated_modules("evolution")
