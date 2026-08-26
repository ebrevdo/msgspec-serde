from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import msgspec
import numpy as np
import pytest

import msgspec_flatbuffers
from msgspec_flatbuffers import dec_hook, enc_hook, generate

ROOT = Path(__file__).parents[1]
COMPATIBILITY = ROOT / "tests" / "generated_compatibility"
SCHEMA = COMPATIBILITY / "core.fbs"
RELEASES = COMPATIBILITY / "releases"
VERSION_PATTERN = re.compile(r"v(\d+)_(\d+)_(\d+)")
HAS_FLATC = shutil.which("flatc") is not None


@dataclass(frozen=True)
class GeneratedCase:
    name: str
    version: str
    module_path: Path | None


def _release_cases() -> list[GeneratedCase]:
    cases: list[GeneratedCase] = []
    for directory in RELEASES.glob("v*"):
        match = VERSION_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        version = ".".join(match.groups())
        cases.append(
            GeneratedCase(
                name=directory.name,
                version=version,
                module_path=directory / "core.py",
            )
        )
    cases.sort(key=lambda case: tuple(int(part) for part in case.version.split(".")))
    return cases


def _major(version: str) -> int:
    return int(version.split(".", 1)[0])


CURRENT = GeneratedCase(
    name="current",
    version=msgspec_flatbuffers.__version__,
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
    return case, _load_module(module_name, module_path)


def test_generated_public_api_contract(
    generated_case: tuple[GeneratedCase, ModuleType],
) -> None:
    case, generated = generated_case
    assert generated.__msgspec_flatbuffers_generated_version__ == case.version

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

    buffer = model.to_flatbuffer()
    assert buffer.readonly
    assert generated.Monster.from_flatbuffer(buffer) == model

    view = generated.MonsterView.from_buffer(buffer)
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

    framed = model.to_flatbuffer(size_prefixed=True)
    assert (
        generated.MonsterView.from_buffer(
            framed,
            size_prefixed=True,
        ).to_model()
        == model
    )

    encoded = msgspec.json.encode(model, enc_hook=enc_hook)
    decoded = msgspec.json.decode(
        encoded,
        type=generated.Monster,
        dec_hook=dec_hook,
    )
    assert decoded == model

    class ApplicationMonster(generated.Monster, dict=True):
        def __post_init__(self) -> None:
            self.was_validated = True

    application_model = ApplicationMonster.from_flatbuffer(buffer)
    assert application_model.was_validated
    assert application_model.to_flatbuffer().readonly
