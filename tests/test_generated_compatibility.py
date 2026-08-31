from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import msgspec_serde
from msgspec_serde import flatbuffer, generate, json
from msgspec_serde._flatbuffer import register_type

ROOT = Path(__file__).parents[1]
COMPATIBILITY = ROOT / "tests" / "generated_compatibility"
SCHEMA = COMPATIBILITY / "core.fbs"
RELEASES = COMPATIBILITY / "releases"
SNAPSHOT_PATTERN = re.compile(r"v(\d+)_(\d+)")
GENERATED_VERSION_PATTERN = re.compile(
    r"^__msgspec_serde_generated_version__ = "
    r"['\"]((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))['\"]$",
    re.MULTILINE,
)
HAS_FLATC = shutil.which("flatc") is not None


@dataclass(frozen=True)
class GeneratedCase:
    name: str
    version: str
    module_path: Path | None


def _release_cases() -> list[GeneratedCase]:
    cases: list[GeneratedCase] = []
    for directory in RELEASES.glob("v*"):
        match = SNAPSHOT_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        module_path = directory / "core.py"
        version_match = GENERATED_VERSION_PATTERN.search(
            module_path.read_text(encoding="utf-8")
        )
        assert version_match is not None, f"generated version missing from {module_path}"
        version = version_match.group(1)
        assert version.split(".")[:2] == list(match.groups()), (
            f"generated version {version} does not match snapshot {directory.name}"
        )
        cases.append(
            GeneratedCase(
                name=directory.name,
                version=version,
                module_path=module_path,
            )
        )
    cases.sort(key=lambda case: tuple(int(part) for part in case.version.split(".")))
    return cases


def _major(version: str) -> int:
    return int(version.split(".", 1)[0])


CURRENT = GeneratedCase(
    name="current",
    version=msgspec_serde.__version__,
    module_path=None,
)
RELEASE_CASES = _release_cases()
CURRENT_MAJOR = _major(CURRENT.version)
SAME_MAJOR_CASES = [
    case for case in RELEASE_CASES if _major(case.version) == CURRENT_MAJOR
]
if HAS_FLATC:
    SAME_MAJOR_CASES.insert(0, CURRENT)


def _load_module(name: str, path: Path) -> ModuleType:
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module", params=SAME_MAJOR_CASES, ids=lambda case: case.name)
def generated_case(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[GeneratedCase, ModuleType]:
    case: GeneratedCase = request.param
    module_path = case.module_path
    if module_path is None:
        module_path = generate(
            SCHEMA,
            tmp_path_factory.mktemp("current-generated-compatibility"),
            project_root=ROOT,
            gen_onefile=True,
        )
    module_name = f"_generated_compatibility_{case.name}"
    module = _load_module(module_name, module_path)
    register_type(
        module.Monster,
        module._FB_NATIVE_MODULE,
        "Compatibility.Monster",
        "COMP",
    )
    return case, module


def test_generated_public_api_contract(
    generated_case: tuple[GeneratedCase, ModuleType],
) -> None:
    case, generated = generated_case
    assert generated.__msgspec_serde_generated_version__ == case.version

    model = generated.Monster(
        pos=generated.Vec3(x=1.0, y=2.0, z=3.0),
        mana=175,
        hp=80,
        name="compatibility",
        inventory=b"\x00\x01\xff",
        color=generated.Color.Green,
        weapons=[generated.Weapon(name="Axe", damage=12)],
        scores=np.array([1.5, 2.5], dtype=np.float32),
        tags=["one", "two"],
        path=[generated.Vec3(x=4.0, y=5.0, z=6.0)],
        optional_energy=123,
        colors=[generated.Color.Red, generated.Color.Blue],
    )

    buffer = flatbuffer.encode(model)
    assert buffer.readonly
    assert flatbuffer.decode(buffer, type=generated.Monster) == model

    view = flatbuffer.decode(buffer, type=generated.MonsterView)
    assert view.name == "compatibility"
    assert view.name is view.name
    assert view.pos.z == 3.0
    assert bytes(view.inventory) == b"\x00\x01\xff"
    assert view.scores.dtype == np.dtype("<f4")
    assert not view.scores.flags.writeable
    assert view.scores is view.scores
    assert view.weapons[0].damage == 12
    assert view.weapons[0] is view.weapons[0]
    assert tuple(view.tags) == ("one", "two")

    restored = view.to_model()
    assert restored == model
    assert restored.scores.flags.owndata
    assert restored.scores.flags.writeable
    restored.scores[0] = 99.0
    assert view.scores[0] == 1.5

    framed = flatbuffer.encode(model, size_prefixed=True)
    assert (
        flatbuffer.decode(
            framed,
            type=generated.MonsterView,
            size_prefixed=True,
        ).to_model()
        == model
    )

    encoded = json.encode(model)
    decoded = json.decode(encoded, type=generated.Monster)
    assert decoded == model

    class ApplicationMonster(generated.Monster, dict=True):
        def __post_init__(self) -> None:
            self.was_validated = True

    application_model = flatbuffer.decode(buffer, type=ApplicationMonster)
    assert application_model.was_validated
    assert flatbuffer.encode(application_model).readonly
