from __future__ import annotations

import importlib
import importlib.util
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pytest

from msgspec_flatbuffers import (
    BaseType,
    BufferBoundsError,
    FieldDefinition,
    GenerationError,
    InvalidBufferError,
    ObjectDefinition,
    Schema,
    TypeReference,
    generate,
    render_module,
)
from msgspec_flatbuffers.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "monster.fbs"
HAS_FLATC = shutil.which("flatc") is not None
pytestmark = pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")


def _load_module(name: str, path: Path) -> ModuleType:
    generated_root = path.parent
    while (generated_root / "__init__.py").is_file():
        generated_root = generated_root.parent
    root_string = str(generated_root)
    generated_packages = {
        child.name
        for child in generated_root.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }
    generated_modules = {
        child.stem for child in generated_root.glob("*.py")
    }
    for imported_name in tuple(sys.modules):
        if (
            imported_name.split(".", 1)[0] in generated_packages
            or imported_name in generated_modules
            or imported_name == name
        ):
            sys.modules.pop(imported_name, None)
    sys.path.insert(0, root_string)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root_string)
    return module


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_string = str(path)
    sys.path.insert(0, path_string)
    try:
        yield
    finally:
        sys.path.remove(path_string)


def _write_schema(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _assert_owned_writable(array: np.ndarray | None) -> np.ndarray:
    assert array is not None
    assert array.flags.owndata
    assert array.flags.writeable
    return array


def test_generated_module_round_trips_and_caches_views(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    module_path = generate(FIXTURE, generated_root)
    generated_source = module_path.read_text(encoding="utf-8")
    assert "scores: npt.NDArray[np.float32] | None" in generated_source
    assert "weapons: list[" in generated_source
    assert ".Weapon] | None" in generated_source
    assert "tags: list[str] | None" in generated_source
    assert ".Vec3] | None" in generated_source
    assert ".Color] | None" in generated_source

    assert module_path.relative_to(generated_root) == Path("example/monster.py")
    assert generate(FIXTURE, generated_root) == module_path
    assert module_path.read_text(encoding="utf-8") == generated_source

    generated = _load_module("test_generated_monster", module_path)
    model = generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        mana=175,
        hp=80,
        name="fred",
        inventory=b"\x00\x01\x02\xff",
        color=generated.Color.Green,
        weapons=[
            generated.Weapon(name="Sword", damage=3),
            generated.Weapon(name="Axe", damage=5),
        ],
        scores=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        tags=["one", "two"],
        path=[
            generated.Vec3(x=4.0, y=5.0, z=6.0),
            generated.Vec3(x=7.0, y=8.0, z=9.0),
        ],
        optional_energy=123,
        colors=[generated.Color.Red, generated.Color.Blue],
    )

    buffer = model.to_flatbuffer()
    assert isinstance(buffer, memoryview)
    assert buffer.readonly
    assert type(buffer.obj).__name__ == "NativeBuffer"
    with pytest.raises(TypeError):
        buffer[0] = 0
    view = generated.MonsterView.from_buffer(buffer)
    assert generated.Monster.from_flatbuffer(buffer) == model

    assert view.name == "fred"
    assert view.name is view.name
    assert view.pos.z == 3.0
    assert view.inventory.readonly
    assert bytes(view.inventory) == b"\x00\x01\x02\xff"
    assert view.scores.dtype == np.dtype("<f4")
    assert not view.scores.flags.owndata
    assert not view.scores.flags.writeable
    np.testing.assert_array_equal(view.scores, [1.5, 2.5, 3.5])
    assert view.scores is view.scores
    with pytest.raises(ValueError):
        view.scores.setflags(write=True)
    with pytest.raises(TypeError):
        view.inventory[0] = 9

    weapons = view.weapons
    assert weapons is view.weapons
    assert weapons.cached_count == 0
    first_scan = tuple(weapons)
    assert weapons.cached_count == 2
    second_scan = tuple(weapons)
    assert all(first is second for first, second in zip(first_scan, second_scan))

    tags = view.tags
    assert tags.cached_count == 0
    assert tuple(tags) == ("one", "two")
    assert tags.cached_count == 2

    path = view.path
    assert path.cached_count == 0
    assert path[1].y == 8.0
    assert path[1] is path[1]
    assert path.cached_count == 1
    assert view.optional_energy == 123

    first_model = view.to_model()
    second_model = view.to_model()
    assert first_model == model
    assert second_model == model
    assert not (first_model != model)
    wrong_dtype = model.__replace__(
        scores=np.array([1.5, 2.5, 3.5], dtype=np.float64)
    )
    assert first_model != wrong_dtype
    assert first_model is not second_model
    first_scores = _assert_owned_writable(first_model.scores)
    second_scores = _assert_owned_writable(second_model.scores)
    assert not np.shares_memory(first_scores, view.scores)
    assert not np.shares_memory(first_scores, second_scores)
    assert first_model.weapons is not None
    assert first_model.tags is not None
    assert first_model.path is not None
    assert first_model.colors is not None
    first_model.weapons.append(generated.Weapon(name="Bow", damage=2))
    first_model.tags.append("three")
    first_model.path.append(generated.Vec3(x=10.0, y=11.0, z=12.0))
    first_model.colors.append(generated.Color.Green)
    first_scores[0] = 99.0
    assert view.scores[0] == 1.5
    assert len(view.weapons) == 2
    assert tuple(view.tags) == ("one", "two")
    assert generated.MonsterView.from_buffer(
        first_model.to_flatbuffer()
    ).to_model() == first_model

    untouched_view = generated.MonsterView.from_buffer(buffer)
    untouched_weapons = untouched_view.weapons
    assert untouched_weapons is not None
    assert untouched_weapons.cached_count == 0
    assert untouched_view.to_model() == model
    assert untouched_weapons.cached_count == 0

    model.hp = 1
    assert model.hp == 1
    with pytest.raises(AttributeError):
        view.extra_state = "not allowed"

    size_prefixed = model.to_flatbuffer(size_prefixed=True)
    size_prefixed_view = generated.MonsterView.from_buffer(
        size_prefixed,
        size_prefixed=True,
    )
    assert size_prefixed_view.hp == 1
    assert bytes(size_prefixed_view.buffer) == size_prefixed[4:]
    assert generated.Monster.from_flatbuffer(
        size_prefixed,
        size_prefixed=True,
    ) == model
    undersized_identifier = bytearray(size_prefixed)
    struct.pack_into("<I", undersized_identifier, 0, 4)
    with pytest.raises(BufferBoundsError, match="file identifier"):
        generated.MonsterView.from_buffer(
            undersized_identifier,
            size_prefixed=True,
        )

    default_model = generated.Monster()
    empty_buffer = default_model.to_flatbuffer()
    defaults = generated.MonsterView.from_buffer(empty_buffer)
    assert defaults.optional_energy is None
    assert defaults.weapons is None
    assert defaults.scores is None
    assert defaults.tags is None
    assert defaults.path is None
    assert defaults.colors is None

    assert len(empty_buffer) == 16
    assert empty_buffer.obj._allocation_size == 17

    empty_vectors = generated.Monster(
        weapons=[],
        scores=np.array([], dtype=np.float32),
        tags=[],
        path=[],
        colors=[],
    )
    empty_materialized = generated.MonsterView.from_buffer(
        empty_vectors.to_flatbuffer()
    ).to_model()
    assert empty_materialized == empty_vectors
    assert empty_materialized.weapons == []
    assert empty_materialized.tags == []
    assert empty_materialized.path == []
    assert empty_materialized.colors == []
    empty_scores = _assert_owned_writable(empty_materialized.scores)
    assert empty_scores.shape == (0,)

    wrong_identifier = bytearray(buffer)
    wrong_identifier[4:8] = b"NOPE"
    with pytest.raises(InvalidBufferError, match="file identifier"):
        generated.MonsterView.from_buffer(wrong_identifier)
    with pytest.raises(InvalidBufferError, match="file identifier"):
        generated.Monster.from_flatbuffer(wrong_identifier)

    invalid_vector = bytearray(buffer)
    unchecked = generated.MonsterView.from_buffer(invalid_vector)
    inventory_field = unchecked._field_position(12, 4)
    assert inventory_field is not None
    inventory_length = inventory_field + struct.unpack_from(
        "<I",
        invalid_vector,
        inventory_field,
    )[0]
    struct.pack_into("<I", invalid_vector, inventory_length, 2**32 - 1)
    corrupt_inventory = generated.MonsterView.from_buffer(invalid_vector)
    with pytest.raises(BufferBoundsError, match="vector data"):
        _ = corrupt_inventory.inventory
    with pytest.raises(BufferBoundsError, match="vector data"):
        corrupt_inventory.to_model()
    with pytest.raises(BufferBoundsError, match="vector data"):
        generated.Monster.from_flatbuffer(invalid_vector)

    invalid_table_vector = bytearray(buffer)
    unchecked = generated.MonsterView.from_buffer(invalid_table_vector)
    weapons_info = unchecked._vector_info(16, 4)
    assert weapons_info is not None
    weapons_start, weapons_length = weapons_info
    assert weapons_length == 2
    struct.pack_into("<I", invalid_table_vector, weapons_start, 0)
    corrupt_weapons = generated.MonsterView.from_buffer(
        invalid_table_vector
    ).weapons
    assert corrupt_weapons is not None
    with pytest.raises(InvalidBufferError, match="table vector element"):
        corrupt_weapons[0]
    with pytest.raises(InvalidBufferError, match="table vector offset"):
        generated.MonsterView.from_buffer(invalid_table_vector).to_model()
    with pytest.raises(InvalidBufferError, match="table vector offset"):
        generated.Monster.from_flatbuffer(invalid_table_vector)

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    subprocess.run(
        [
            "flatc",
            "--python",
            "--python-typing",
            "-o",
            str(reference_root),
            str(FIXTURE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with _temporary_sys_path(reference_root):
        reference_path = reference_root / "Example" / "Monster.py"
        reference = _load_module("Example.Monster", reference_path)
        reference_view = reference.Monster.GetRootAs(buffer)
        assert reference_view.Name() == b"fred"
        assert reference_view.Hp() == 80
        assert reference_view.WeaponsLength() == 2
        assert reference_view.ScoresLength() == 3
        assert reference_view.TagsLength() == 2
        assert reference_view.PathLength() == 2
        assert reference_view.OptionalEnergy() == 123


def test_model_subclasses_drive_native_materialization(tmp_path: Path) -> None:
    module_path = generate(FIXTURE, tmp_path / "generated")
    generated = _load_module("test_generated_subclasses", module_path)
    namespace: dict[str, Any] = {
        "ClassVar": ClassVar,
        "Monster": generated.Monster,
        "Vec3": generated.Vec3,
        "Weapon": generated.Weapon,
    }
    exec(
        compile(
            """
class ValidatedWeapon(Weapon, dict=True):
    validation_kind: ClassVar[str] = "weapon"

    def __post_init__(self):
        if self.damage < 0:
            raise ValueError("negative damage")
        self.was_validated = True

class ValidatedVec3(Vec3, dict=True):
    def __post_init__(self):
        self.was_validated = True

class ValidatedMonster(Monster, dict=True):
    pos: ValidatedVec3 | None
    weapons: list[ValidatedWeapon] | None
    path: list[ValidatedVec3] | None
    validation_kind: ClassVar[str] = "monster"

    def __post_init__(self):
        self.was_validated = True

class InvalidMonster(Monster):
    application_note: str = ""
""",
            "<model-subclasses>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    validated_weapon = namespace["ValidatedWeapon"]
    validated_vec3 = namespace["ValidatedVec3"]
    validated_monster = namespace["ValidatedMonster"]
    invalid_monster = namespace["InvalidMonster"]

    model = generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        name="subclasses",
        weapons=[generated.Weapon(name="Axe", damage=5)],
        path=[generated.Vec3(x=4.0, y=5.0, z=6.0)],
    )
    buffer = model.to_flatbuffer()

    direct = validated_monster.from_flatbuffer(buffer)
    from_view = generated.MonsterView.from_buffer(buffer).to_model(validated_monster)
    for restored in (direct, from_view):
        assert type(restored) is validated_monster
        assert restored.was_validated
        assert restored.weapons is not None
        assert type(restored.weapons[0]) is validated_weapon
        assert restored.weapons[0].was_validated
        assert type(restored.pos) is validated_vec3
        assert restored.pos.was_validated
        assert restored.path is not None
        assert type(restored.path[0]) is validated_vec3
        assert restored.path[0].was_validated
        assert restored.to_flatbuffer().readonly

    assert set(validated_monster.__struct_fields__) == set(
        generated.Monster.__struct_fields__
    )
    with pytest.raises(TypeError, match="serialized msgspec fields"):
        invalid_monster.from_flatbuffer(buffer)
    with pytest.raises(TypeError, match="serialized msgspec fields"):
        invalid_monster(name="invalid").to_flatbuffer()


def test_generate_resolves_types_from_another_fbs_module(tmp_path: Path) -> None:
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    common = definitions / "common.fbs"
    _write_schema(
        common,
        "namespace Shared; table Child { value:int; }",
    )
    root = tmp_path / "root.fbs"
    _write_schema(
        root,
        'include "common.fbs";',
        "namespace Example;",
        "table Root { child:Shared.Child; }",
        "root_type Root;",
    )
    output = tmp_path / "output"

    generate(common, output, project_root=tmp_path)
    module_path = generate(
        root,
        output,
        include_dirs=[definitions],
        project_root=tmp_path,
    )

    with _temporary_sys_path(output):
        generated = _load_module("example.root", module_path)

        model = generated.Root(child=generated.Child(value=42))
        view = generated.RootView.from_buffer(model.to_flatbuffer())
        assert view.child.value == 42
        assert view.child is view.child
        assert view.to_model() == model


def test_byte_vector_can_be_loaded_as_a_cached_nested_view(tmp_path: Path) -> None:
    base_schema = tmp_path / "base_extensions.fbs"
    _write_schema(
        base_schema,
        "namespace Example;",
        "table Extension { type_id:ulong; data:[ubyte]; }",
        "table Base { extensions:[Extension]; }",
        "root_type Base;",
    )
    adam_schema = tmp_path / "adam_extension.fbs"
    _write_schema(
        adam_schema,
        "namespace Example;",
        "table AdamExtension { beta1:double; beta2:double; }",
        "root_type AdamExtension;",
        'file_identifier "ADAM";',
    )
    output = tmp_path / "output"
    base_path = generate(base_schema, output, project_root=tmp_path)
    adam_path = generate(adam_schema, output, project_root=tmp_path)

    with _temporary_sys_path(output):
        base = _load_module("example.base_extensions", base_path)
        adam = _load_module("example.adam_extension", adam_path)

        type_id = 0xA813
        payload = adam.AdamExtension(beta1=0.9, beta2=0.999).to_flatbuffer()
        outer = base.Base(
            extensions=[base.Extension(type_id=type_id, data=payload)]
        ).to_flatbuffer()
        extensions = base.BaseView.from_buffer(outer).extensions
        assert extensions is not None
        extension = extensions[0]
        payload_view = extension.data
        assert payload_view is not None

        nested = extension.data_as(adam.AdamExtensionView)

        assert nested is not None
        assert nested.beta1 == 0.9
        assert nested.beta2 == 0.999
        assert nested is extension.data_as(adam.AdamExtensionView)
        assert nested.buffer is payload_view
        assert payload_view.obj is outer.obj

        missing_data = base.Base(
            extensions=[base.Extension(type_id=type_id)]
        ).to_flatbuffer()
        missing_extensions = base.BaseView.from_buffer(missing_data).extensions
        assert missing_extensions is not None
        assert missing_extensions[0].data_as(adam.AdamExtensionView) is None

        prefixed_payload = adam.AdamExtension(
            beta1=0.8,
            beta2=0.888,
        ).to_flatbuffer(size_prefixed=True)
        prefixed_outer = base.Base(
            extensions=[base.Extension(type_id=type_id, data=prefixed_payload)]
        ).to_flatbuffer()
        prefixed_extensions = base.BaseView.from_buffer(prefixed_outer).extensions
        assert prefixed_extensions is not None
        prefixed_extension = prefixed_extensions[0]
        prefixed_nested = prefixed_extension.data_as(
            adam.AdamExtensionView,
            size_prefixed=True,
        )
        assert prefixed_nested is not None
        assert prefixed_nested.beta1 == 0.8
        assert prefixed_nested is prefixed_extension.data_as(
            adam.AdamExtensionView,
            size_prefixed=True,
        )

        invalid_payload = bytearray(payload)
        invalid_payload[4:8] = b"NOPE"
        invalid_outer = base.Base(
            extensions=[base.Extension(type_id=type_id, data=invalid_payload)]
        ).to_flatbuffer()
        invalid_extensions = base.BaseView.from_buffer(invalid_outer).extensions
        assert invalid_extensions is not None
        with pytest.raises(InvalidBufferError, match="file identifier"):
            invalid_extensions[0].data_as(adam.AdamExtensionView)


def test_native_builder_presizes_shallow_vectors(tmp_path: Path) -> None:
    module_path = generate(FIXTURE, tmp_path / "generated")
    generated = _load_module("test_generated_presizing", module_path)
    inventory = b"x" * 65_536
    scores = np.arange(16_384, dtype=np.float32)
    model = generated.Monster(inventory=inventory, scores=scores)

    buffer = model.to_flatbuffer()

    assert buffer.obj._allocation_size >= len(buffer)
    assert buffer.obj._allocation_size * 100 <= len(buffer) * 102
    rebuilt = generated.MonsterView.from_buffer(buffer).to_model()
    assert rebuilt == model


def test_numeric_vector_serialization_requires_generated_numpy_dtype(
    tmp_path: Path,
) -> None:
    module_path = generate(FIXTURE, tmp_path / "generated")
    generated = _load_module("test_generated_numeric_dtype", module_path)

    with pytest.raises(TypeError, match="native Float32 data"):
        generated.Monster(
            scores=np.array([1.0, 2.0], dtype=np.float64)
        ).to_flatbuffer()

    non_native = np.dtype(np.float32).newbyteorder("S")
    with pytest.raises(TypeError, match="native Float32 data"):
        generated.Monster(
            scores=np.array([1.0, 2.0], dtype=non_native)
        ).to_flatbuffer()


def test_native_builder_samples_variable_size_vectors(tmp_path: Path) -> None:
    module_path = generate(FIXTURE, tmp_path / "generated")
    source = module_path.read_text(encoding="utf-8")
    assert "import flatbuffers" not in source
    assert "def _build_" not in source
    assert "def _estimate_" not in source
    assert source.count("_FB_NATIVE_MODULE.unpack_view(") == 2
    assert "_FB_UNPACK_" not in source
    generated = _load_module("test_generated_sampled_presizing", module_path)
    weapons = [
        generated.Weapon(name="w" * (2_000 + index), damage=index)
        for index in range(100)
    ]
    tags = ["t" * (1_000 + index) for index in range(100)]
    model = generated.Monster(weapons=weapons, tags=tags)

    buffer = model.to_flatbuffer(initial_size=16)

    assert buffer.obj._allocation_size > 250_000
    assert buffer.obj._allocation_size * 100 <= len(buffer) * 102
    assert generated.MonsterView.from_buffer(buffer).to_model() == model


@pytest.mark.parametrize("field", ["from_flatbuffer", "to_flatbuffer"])
def test_model_method_name_collisions_are_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    source = tmp_path / "method_collision.fbs"
    _write_schema(
        source,
        f"table Root {{ {field}:int; }} root_type Root;",
    )

    with pytest.raises(GenerationError, match="generated model method"):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_cli_generates_a_module(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "output"

    assert main(["generate", str(FIXTURE), "-o", str(output)]) == 0

    generated = output / "example" / "monster.py"
    assert generated.is_file()
    assert str(generated) in capsys.readouterr().out


def test_multiple_namespaces_generate_separate_modules(tmp_path: Path) -> None:
    source = tmp_path / "multiple_namespaces.fbs"
    _write_schema(
        source,
        "namespace MultipleAlpha;",
        "table Item { value:int; }",
        "namespace MultipleBeta;",
        "table Item { alpha:MultipleAlpha.Item; }",
        "root_type Item;",
    )
    output = tmp_path / "generated"

    root_path = generate(source, output, project_root=tmp_path)

    assert root_path == output / "multiple_beta" / "item.py"
    assert (output / "multiple_alpha" / "item.py").is_file()
    with _temporary_sys_path(output):
        alpha = importlib.import_module("multiple_alpha.item")
        beta = importlib.import_module("multiple_beta.item")
        model = beta.Item(alpha=alpha.Item(value=42))

        assert beta.Item.from_flatbuffer(model.to_flatbuffer()) == model


def test_gen_onefile_flattens_unique_namespaces(tmp_path: Path) -> None:
    source = tmp_path / "one_file.fbs"
    _write_schema(
        source,
        "namespace OneFileAlpha;",
        "table Child { value:int; }",
        "namespace OneFileBeta;",
        "table Root { child:OneFileAlpha.Child; }",
        "root_type Root;",
    )
    output = tmp_path / "generated"

    module_path = generate(
        source,
        output,
        project_root=tmp_path,
        gen_onefile=True,
    )
    generated = _load_module("one_file_generated", module_path)
    model = generated.Root(child=generated.Child(value=42))

    assert module_path == output / "one_file.py"
    assert generated.Root.from_flatbuffer(model.to_flatbuffer()) == model


def test_gen_onefile_resolves_included_onefile_modules(tmp_path: Path) -> None:
    common = tmp_path / "onefile_common.fbs"
    _write_schema(
        common,
        "namespace OneFileShared; table Child { value:int; }",
    )
    root = tmp_path / "onefile_root.fbs"
    _write_schema(
        root,
        'include "onefile_common.fbs";',
        "namespace OneFileRoot;",
        "table Root { child:OneFileShared.Child; }",
        "root_type Root;",
    )
    output = tmp_path / "generated"
    generate(common, output, project_root=tmp_path, gen_onefile=True)
    root_path = generate(
        root,
        output,
        include_dirs=[tmp_path],
        project_root=tmp_path,
        gen_onefile=True,
    )

    with _temporary_sys_path(output):
        common_module = importlib.import_module("onefile_common")
        root_module = importlib.import_module("onefile_root")
        model = root_module.Root(child=common_module.Child(value=42))

        assert root_path == output / "onefile_root.py"
        assert root_module.Root.from_flatbuffer(model.to_flatbuffer()) == model


def test_multiple_namespace_modules_can_reference_each_other(
    tmp_path: Path,
) -> None:
    source = tmp_path / "namespace_cycle.fbs"
    _write_schema(
        source,
        "namespace NamespaceCycleA;",
        "table A { b:NamespaceCycleB.B; }",
        "namespace NamespaceCycleB;",
        "table B { a:NamespaceCycleA.A; }",
        "root_type B;",
    )
    output = tmp_path / "generated"
    generate(source, output, project_root=tmp_path)

    with _temporary_sys_path(output):
        cycle_a = importlib.import_module("namespace_cycle_a.a")
        cycle_b = importlib.import_module("namespace_cycle_b.b")
        model = cycle_b.B(a=cycle_a.A())

        assert cycle_b.B.from_flatbuffer(model.to_flatbuffer()) == model


def test_gen_onefile_rejects_namespace_name_collisions(tmp_path: Path) -> None:
    source = tmp_path / "one_file_collision.fbs"
    _write_schema(
        source,
        "namespace CollisionAlpha; table Item {}",
        "namespace CollisionBeta; table Item {}",
        "root_type Item;",
    )

    with pytest.raises(GenerationError, match="Python symbol 'Item'"):
        generate(
            source,
            tmp_path / "generated",
            project_root=tmp_path,
            gen_onefile=True,
        )


def test_generation_rejects_colliding_module_names(tmp_path: Path) -> None:
    source = tmp_path / "module_collision.fbs"
    _write_schema(
        source,
        "namespace Collision;",
        "table FooBar {}",
        "table foo_bar {}",
        "root_type FooBar;",
    )
    output = tmp_path / "generated"

    with pytest.raises(GenerationError, match="both generate module"):
        generate(source, output, project_root=tmp_path)

    assert not output.exists()


def test_generation_rejects_colliding_field_names(tmp_path: Path) -> None:
    source = tmp_path / "field_collision.fbs"
    _write_schema(
        source,
        "table Collision { class:int; class_:int; }",
        "root_type Collision;",
    )

    with pytest.raises(GenerationError, match="both generate Python name 'class_'"):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_generation_rejects_colliding_enum_value_names(tmp_path: Path) -> None:
    source = tmp_path / "enum_collision.fbs"
    _write_schema(
        source,
        "enum Collision:byte { class, class_ }",
    )

    with pytest.raises(GenerationError, match="both generate Python name 'class_'"):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_runtime_support_names_can_be_schema_names(tmp_path: Path) -> None:
    source = tmp_path / "support_name.fbs"
    _write_schema(source, "table TableView {}", "root_type TableView;")
    path = generate(source, tmp_path / "generated", project_root=tmp_path)
    generated = _load_module("support_name_generated", path)

    model = generated.TableView()
    buffer = model.to_flatbuffer()

    assert generated.TableView.from_flatbuffer(buffer) == model
    assert generated.TableViewView.from_buffer(buffer).to_model() == model


def test_generation_rejects_offset64_before_rendering() -> None:
    schema = Schema(
        objects=(
            ObjectDefinition(
                name="Offset64.Root",
                fields=(
                    FieldDefinition(
                        name="value",
                        type=TypeReference(base_type=BaseType.STRING),
                        id=0,
                        offset=4,
                        offset64=True,
                    ),
                ),
                declaration_file="//offset64.fbs",
            ),
        ),
        enums=(),
        root_table="Offset64.Root",
    )

    with pytest.raises(GenerationError, match="64-bit offset field"):
        render_module(schema, "//offset64.fbs")


def test_cli_gen_onefile_omits_namespace_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output"

    assert main(
        ["generate", str(FIXTURE), "-o", str(output), "--gen-onefile"]
    ) == 0

    generated = output / "monster.py"
    assert generated.is_file()
    assert str(generated) in capsys.readouterr().out
