from __future__ import annotations

import shutil
from pathlib import Path

import flatbuffers
import msgspec
import pytest

from msgspec_flatbuffers import (
    AdvancedFeature,
    BaseType,
    FlatcError,
    FlatcNotFoundError,
    InvalidSchemaError,
    compile_schema,
    compile_schema_to_bfbs,
    parse_bfbs,
)
from msgspec_flatbuffers._reflection import Schema as ReflectedSchema

FIXTURE = Path(__file__).parent / "fixtures" / "monster.fbs"
HAS_FLATC = shutil.which("flatc") is not None


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_compile_schema_builds_normalized_msgspec_models() -> None:
    schema = compile_schema(FIXTURE, project_root=FIXTURE.parent)

    assert schema.root_table == "Example.Monster"
    assert schema.root is schema.object("Example.Monster")
    assert schema.file_identifier == "MONS"
    assert schema.file_extension == "mon"
    assert [item.name for item in schema.objects] == [
        "Example.Monster",
        "Example.Vec3",
        "Example.Weapon",
    ]
    assert schema.object("Example.Vec3").is_struct
    assert schema.object("Example.Monster").documentation == (
        "A representative root table.",
    )

    fields = {field.name: field for field in schema.root.fields}
    assert fields["mana"].default_integer == 150
    assert fields["inventory"].type.base_type is BaseType.VECTOR
    assert fields["inventory"].type.element is BaseType.UBYTE
    assert fields["pos"].type.base_type is BaseType.OBJECT

    color = schema.enum("Example.Color")
    assert color.underlying_type.base_type is BaseType.BYTE
    assert [(item.name, item.value) for item in color.values] == [
        ("Red", 1),
        ("Green", 2),
        ("Blue", 3),
    ]
    assert color.documentation == ("A color used by a monster.",)
    assert schema.services[0].name == "Example.MonsterService"
    assert schema.services[0].calls[0].request == "Example.Monster"
    assert schema.services[0].calls[0].response == "Example.Monster"
    assert b'"root_table":"Example.Monster"' in msgspec.json.encode(schema)


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_compile_schema_to_bfbs_round_trips() -> None:
    data = compile_schema_to_bfbs(FIXTURE)

    schema = parse_bfbs(data)

    assert schema.root_table == "Example.Monster"


def test_parse_bfbs_rejects_other_buffers() -> None:
    with pytest.raises(InvalidSchemaError, match="BFBS"):
        parse_bfbs(b"not a binary schema")


def test_parse_bfbs_wraps_malformed_reflection_data() -> None:
    with pytest.raises(InvalidSchemaError, match="malformed"):
        parse_bfbs(b"\x04\x00\x00\x00BFBS")


def test_parse_bfbs_rejects_default_vector_and_string_features() -> None:
    builder = flatbuffers.Builder(32)
    ReflectedSchema.SchemaStart(builder)
    ReflectedSchema.SchemaAddAdvancedFeatures(
        builder,
        int(AdvancedFeature.DEFAULT_VECTORS_AND_STRINGS),
    )
    schema = ReflectedSchema.SchemaEnd(builder)
    builder.Finish(schema, file_identifier=b"BFBS")

    with pytest.raises(InvalidSchemaError, match="does not preserve"):
        parse_bfbs(builder.Output())


def test_compile_schema_reports_missing_compiler() -> None:
    with pytest.raises(FlatcNotFoundError, match="flatc executable not found"):
        compile_schema(FIXTURE, flatc="definitely-not-a-flatc-executable")


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_compile_schema_resolves_includes(tmp_path: Path) -> None:
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    (definitions / "common.fbs").write_text(
        "namespace Shared; table Child { value:int; }",
        encoding="utf-8",
    )
    schema_file = tmp_path / "root.fbs"
    schema_file.write_text(
        '\n'.join(
            [
                'include "common.fbs";',
                "namespace Example;",
                "table Root { child:Shared.Child; }",
                "root_type Root;",
            ]
        ),
        encoding="utf-8",
    )

    schema = compile_schema(
        schema_file,
        include_dirs=[definitions],
        project_root=tmp_path,
    )

    assert schema.root_table == "Example.Root"
    assert schema.object("Shared.Child").fields[0].name == "value"
    files = {item.filename: item for item in schema.files}
    assert "//root.fbs" in files
    assert "//definitions/common.fbs" in files
    assert files["//root.fbs"].included_filenames == (
        "//definitions/common.fbs",
    )


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_compile_schema_reports_idl_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.fbs"
    invalid.write_text("table Broken {", encoding="utf-8")

    with pytest.raises(FlatcError, match="flatc exited"):
        compile_schema(invalid)
