"""Shared data, adapters, and semantic checks for serialization benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import flatbuffers
import numpy as np

from msgspec_flatbuffers import flatbuffer, generate

SCHEMA = Path(__file__).with_name("schemas") / "monster.fbs"
FILE_IDENTIFIER = b"MONS"

PROFILE_NAMES = ("small", "medium", "large")
OPERATIONS = (
    "serialize",
    "deserialize",
    "partial_access",
    "full_traversal",
)
ADAPTERS = (
    "msgspec_flatbuffers",
    "python_flatbuffers",
)


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    inventory_bytes: int
    score_count: int
    weapon_count: int
    tag_count: int
    path_count: int
    color_count: int

    @property
    def logical_elements(self) -> int:
        return (
            self.inventory_bytes
            + self.score_count
            + self.weapon_count
            + self.tag_count
            + self.path_count
            + self.color_count
        )


PROFILES = {
    "small": Profile("small", 64, 16, 4, 4, 4, 8),
    "medium": Profile("medium", 4_096, 1_024, 64, 32, 64, 128),
    "large": Profile("large", 1 << 20, 65_536, 1_024, 512, 1_024, 2_048),
}


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    name: str
    function: Callable[[], object]
    metadata: dict[str, str | int]


@dataclass(frozen=True, slots=True)
class BenchmarkKey:
    profile: str
    operation: str
    adapter: str

    @property
    def name(self) -> str:
        return f"{self.operation}.{self.profile}.{self.adapter}"


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


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _flatc_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def _generate_official_module(output: Path, flatc: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        flatc,
        "--python",
        "--gen-object-api",
        "--gen-onefile",
        "-o",
        str(output),
        str(SCHEMA),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"official FlatBuffers generation failed: {detail}")
    path = output / "monster_generated.py"
    if not path.is_file():
        raise RuntimeError(f"flatc did not create {path}")
    return path


def _fill_bytes(length: int) -> bytes:
    block = bytes(range(256))
    return (block * (length // len(block) + 1))[:length]


def _make_model(generated: ModuleType, profile: Profile) -> Any:
    scores = (
        np.arange(profile.score_count, dtype=np.float32)
        - np.float32(profile.score_count / 2)
    ) / np.float32(8)
    weapons = [
        generated.Weapon(
            name=f"weapon-{index:05d}",
            damage=(index * 17) % 30_000,
        )
        for index in range(profile.weapon_count)
    ]
    path = [
        generated.Vec3(
            x=float(index),
            y=float(index) + 0.25,
            z=-float(index),
        )
        for index in range(profile.path_count)
    ]
    colors = [
        (generated.Color.Red, generated.Color.Green, generated.Color.Blue)[index % 3]
        for index in range(profile.color_count)
    ]
    return generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        mana=175,
        hp=80,
        name=f"benchmark-{profile.name}",
        inventory=_fill_bytes(profile.inventory_bytes),
        color=generated.Color.Green,
        weapons=weapons,
        scores=scores,
        tags=[f"tag-{index:05d}" for index in range(profile.tag_count)],
        path=path,
        optional_energy=123,
        colors=colors,
    )


def _new_official_vec3(module: ModuleType, x: float, y: float, z: float) -> Any:
    value = module.Vec3T()
    value.x = x
    value.y = y
    value.z = z
    return value


def _make_official_model(module: ModuleType, model: Any) -> Any:
    value = module.MonsterT()
    if model.pos is not None:
        value.pos = _new_official_vec3(
            module,
            model.pos.x,
            model.pos.y,
            model.pos.z,
        )
    value.mana = model.mana
    value.hp = model.hp
    value.name = model.name
    value.inventory = (
        None
        if model.inventory is None
        else np.frombuffer(model.inventory, dtype=np.uint8).copy()
    )
    value.color = int(model.color)
    if model.weapons is not None:
        value.weapons = []
        for weapon in model.weapons:
            output = module.WeaponT()
            output.name = weapon.name
            output.damage = weapon.damage
            value.weapons.append(output)
    if model.scores is not None:
        value.scores = model.scores.copy()
    value.tags = None if model.tags is None else list(model.tags)
    if model.path is not None:
        value.path = [
            _new_official_vec3(module, point.x, point.y, point.z)
            for point in model.path
        ]
    value.optionalEnergy = model.optional_energy
    if model.colors is not None:
        value.colors = np.asarray([int(item) for item in model.colors], dtype=np.int8)
    return value


def _serialize_official(model: Any) -> bytearray:
    builder = flatbuffers.Builder(0)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=FILE_IDENTIFIER)
    return builder.Output()


def _open_official(module: ModuleType, buffer: bytes) -> Any:
    if not module.Monster.MonsterBufferHasIdentifier(buffer, 0):
        raise ValueError("benchmark buffer has the wrong FlatBuffers identifier")
    return module.Monster.GetRootAs(buffer)


def _model_signature(model: Any) -> tuple[Any, ...]:
    pos = None if model.pos is None else (model.pos.x, model.pos.y, model.pos.z)
    scores = (
        None
        if model.scores is None
        else np.asarray(model.scores, dtype=np.float32).tobytes()
    )
    return (
        pos,
        model.mana,
        model.hp,
        model.name,
        None if model.inventory is None else bytes(model.inventory),
        int(model.color),
        None
        if model.weapons is None
        else tuple((item.name, item.damage) for item in model.weapons),
        scores,
        None if model.tags is None else tuple(model.tags),
        None
        if model.path is None
        else tuple((item.x, item.y, item.z) for item in model.path),
        model.optional_energy,
        None if model.colors is None else tuple(int(item) for item in model.colors),
    )


def _decoded_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _official_signature(model: Any) -> tuple[Any, ...]:
    pos = None if model.pos is None else (model.pos.x, model.pos.y, model.pos.z)
    scores = (
        None
        if model.scores is None
        else np.asarray(model.scores, dtype=np.float32).tobytes()
    )
    inventory = (
        None
        if model.inventory is None
        else bytes(np.asarray(model.inventory, dtype=np.uint8))
    )
    return (
        pos,
        model.mana,
        model.hp,
        _decoded_text(model.name),
        inventory,
        int(model.color),
        None
        if model.weapons is None
        else tuple((_decoded_text(item.name), item.damage) for item in model.weapons),
        scores,
        None
        if model.tags is None
        else tuple(_decoded_text(item) for item in model.tags),
        None
        if model.path is None
        else tuple((item.x, item.y, item.z) for item in model.path),
        model.optionalEnergy,
        None if model.colors is None else tuple(int(item) for item in model.colors),
    )


def _msgspec_flatbuffers_partial(
    decode: Callable[[bytes], Any],
    buffer: bytes,
) -> object:
    view = decode(buffer)
    scores = view.scores
    return (
        view.hp,
        view.name,
        None if scores is None else float(scores[-1]),
    )


def _python_flatbuffers_partial(module: ModuleType, buffer: bytes) -> object:
    view = _open_official(module, buffer)
    length = view.ScoresLength()
    return (
        view.Hp(),
        view.Name(),
        None if length == 0 else float(view.Scores(length - 1)),
    )


def _msgspec_flatbuffers_traverse(
    decode: Callable[[bytes], Any],
    buffer: bytes,
) -> float:
    view = decode(buffer)
    total = float(view.mana + view.hp + int(view.color))
    if view.pos is not None:
        total += view.pos.x + view.pos.y + view.pos.z
    if view.name is not None:
        total += len(view.name)
    if view.inventory is not None:
        total += len(view.inventory) + view.inventory[0] + view.inventory[-1]
    if view.scores is not None:
        total += len(view.scores) + float(np.sum(view.scores, dtype=np.float64))
    if view.weapons is not None:
        for weapon in view.weapons:
            total += weapon.damage + len(weapon.name)
    if view.tags is not None:
        total += sum(len(tag) for tag in view.tags)
    if view.path is not None:
        for point in view.path:
            total += point.x + point.y + point.z
    if view.optional_energy is not None:
        total += view.optional_energy
    if view.colors is not None:
        total += len(view.colors) + int(np.sum(view.colors, dtype=np.int64))
    return total


def _python_flatbuffers_traverse(module: ModuleType, buffer: bytes) -> float:
    view = _open_official(module, buffer)
    total = float(view.Mana() + view.Hp() + view.Color())
    pos = view.Pos()
    if pos is not None:
        total += pos.X() + pos.Y() + pos.Z()
    name = view.Name()
    if name is not None:
        total += len(name)
    inventory_length = view.InventoryLength()
    if inventory_length:
        inventory = view.InventoryAsNumpy()
        total += inventory_length + int(inventory[0]) + int(inventory[-1])
    score_length = view.ScoresLength()
    if score_length:
        scores = view.ScoresAsNumpy()
        total += score_length + float(np.sum(scores, dtype=np.float64))
    for index in range(view.WeaponsLength()):
        weapon = view.Weapons(index)
        if weapon is not None:
            weapon_name = weapon.Name()
            total += weapon.Damage()
            if weapon_name is not None:
                total += len(weapon_name)
    for index in range(view.TagsLength()):
        total += len(view.Tags(index))
    for index in range(view.PathLength()):
        point = view.Path(index)
        if point is not None:
            total += point.X() + point.Y() + point.Z()
    optional_energy = view.OptionalEnergy()
    if optional_energy is not None:
        total += optional_energy
    color_length = view.ColorsLength()
    if color_length:
        colors = view.ColorsAsNumpy()
        total += color_length + int(np.sum(colors, dtype=np.int64))
    return total


def benchmark_keys(
    profiles: Iterable[str] = PROFILE_NAMES,
) -> tuple[BenchmarkKey, ...]:
    return tuple(
        BenchmarkKey(profile, operation, adapter)
        for profile in dict.fromkeys(profiles)
        for operation in OPERATIONS
        for adapter in ADAPTERS
    )


class BenchmarkCase:
    """Prepared values and operations for one data profile."""

    def __init__(
        self,
        profile: Profile,
        generated: ModuleType,
        official: ModuleType,
    ) -> None:
        self.profile = profile
        self.generated = generated
        self.official = official
        self.model = _make_model(generated, profile)
        self.flatbuffer_encoder = flatbuffer.Encoder()
        self.flatbuffer_model_decoder = flatbuffer.Decoder(self.generated.Monster)
        self.flatbuffer_view_decoder = flatbuffer.Decoder(self.generated.MonsterView)
        self.flatbuffer = bytes(self.flatbuffer_encoder.encode(self.model))
        self.official_model = _make_official_model(self.official, self.model)
        self.official_flatbuffer = self.serialize_python_flatbuffers()
        self.wire_sizes = {
            "msgspec_flatbuffers": len(self.flatbuffer),
            "python_flatbuffers": len(self.official_flatbuffer),
        }
        self._validate()

    def serialize_msgspec_flatbuffers(self) -> object:
        return self.flatbuffer_encoder.encode(self.model)

    def serialize_python_flatbuffers(self) -> bytearray:
        return _serialize_official(self.official_model)

    def deserialize_msgspec_flatbuffers(self) -> object:
        return self.flatbuffer_model_decoder.decode(self.flatbuffer)

    def deserialize_python_flatbuffers(self) -> object:
        if not self.official.Monster.MonsterBufferHasIdentifier(self.flatbuffer, 0):
            raise ValueError("benchmark buffer has the wrong FlatBuffers identifier")
        return self.official.MonsterT.InitFromPackedBuf(self.flatbuffer)

    def partial_access_msgspec_flatbuffers(self) -> object:
        return _msgspec_flatbuffers_partial(
            self.flatbuffer_view_decoder.decode,
            self.flatbuffer,
        )

    def partial_access_python_flatbuffers(self) -> object:
        return _python_flatbuffers_partial(self.official, self.flatbuffer)

    def full_traversal_msgspec_flatbuffers(self) -> object:
        return _msgspec_flatbuffers_traverse(
            self.flatbuffer_view_decoder.decode,
            self.flatbuffer,
        )

    def full_traversal_python_flatbuffers(self) -> object:
        return _python_flatbuffers_traverse(self.official, self.flatbuffer)

    def _validate(self) -> None:
        expected = _model_signature(self.model)
        for adapter in ADAPTERS:
            decoded = _adapter_method(self, "deserialize", adapter)()
            signature = (
                _official_signature(decoded)
                if adapter == "python_flatbuffers"
                else _model_signature(decoded)
            )
            if signature != expected:
                raise AssertionError(f"{adapter} decode changed the logical value")

        official_roundtrip = flatbuffer.decode(
            self.official_flatbuffer,
            type=self.generated.Monster,
        )
        if _model_signature(official_roundtrip) != expected:
            raise AssertionError(
                "official FlatBuffers encode changed the logical value"
            )
        ours = self.full_traversal_msgspec_flatbuffers()
        official = self.full_traversal_python_flatbuffers()
        if ours != official:
            raise AssertionError(
                f"lazy traversal checksums differ: {ours!r} != {official!r}"
            )


def _adapter_method(
    case: BenchmarkCase,
    operation: str,
    adapter: str,
) -> Callable[[], object]:
    return getattr(case, f"{operation}_{adapter}")


class BenchmarkSuite:
    """Generate both APIs and expose equivalent benchmark operations."""

    def __init__(
        self,
        profiles: Iterable[str] = PROFILE_NAMES,
        *,
        generated_root: Path | None = None,
    ) -> None:
        flatc = shutil.which("flatc")
        if flatc is None:
            raise RuntimeError("flatc is required to run the benchmark suite")
        self._module_names = (
            "_benchmark_msgspec_flatbuffers_generated",
            "_benchmark_python_flatbuffers_generated",
        )
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            if generated_root is None:
                self._temporary_directory = tempfile.TemporaryDirectory(
                    prefix="msgspec-flatbuffers-comparison-"
                )
                self.generated_root = Path(self._temporary_directory.name)
                generated_path = generate(
                    SCHEMA,
                    self.generated_root / "msgspec_flatbuffers",
                    flatc=flatc,
                    gen_onefile=True,
                )
                official_path = _generate_official_module(
                    self.generated_root / "python_flatbuffers",
                    flatc,
                )
            else:
                self.generated_root = generated_root.resolve()
                generated_path = (
                    self.generated_root / "msgspec_flatbuffers" / "monster.py"
                )
                official_path = (
                    self.generated_root / "python_flatbuffers" / "monster_generated.py"
                )
                if not generated_path.is_file() or not official_path.is_file():
                    raise RuntimeError(
                        f"shared generated benchmark modules are missing under "
                        f"{self.generated_root}"
                    )
            self.generated = _load_module(self._module_names[0], generated_path)
            self.official = _load_module(self._module_names[1], official_path)
            selected_profiles = tuple(dict.fromkeys(profiles))
            self.cases = tuple(
                BenchmarkCase(
                    PROFILES[name],
                    self.generated,
                    self.official,
                )
                for name in selected_profiles
            )
            self._cases_by_profile = {case.profile.name: case for case in self.cases}
        except BaseException:
            self.close()
            raise
        self.metadata: dict[str, str | int] = {
            "benchmark_schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
            "flatc_version": _flatc_version(flatc),
            "msgspec_flatbuffers_version": _package_version("msgspec-flatbuffers"),
            "flatbuffers_version": _package_version("flatbuffers"),
            "msgspec_version": _package_version("msgspec"),
            "numpy_version": _package_version("numpy"),
        }

    def close(self) -> None:
        for name in self._module_names:
            sys.modules.pop(name, None)
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

    def definition(self, key: BenchmarkKey) -> BenchmarkDefinition:
        case = self._cases_by_profile[key.profile]
        profile = case.profile
        return BenchmarkDefinition(
            name=key.name,
            function=_adapter_method(case, key.operation, key.adapter),
            metadata={
                "operation": key.operation,
                "profile": profile.name,
                "adapter": key.adapter,
                "wire_bytes": case.wire_sizes[key.adapter],
                "logical_elements": profile.logical_elements,
            },
        )


__all__ = [
    "ADAPTERS",
    "OPERATIONS",
    "PROFILES",
    "PROFILE_NAMES",
    "BenchmarkDefinition",
    "BenchmarkKey",
    "BenchmarkSuite",
    "benchmark_keys",
]
