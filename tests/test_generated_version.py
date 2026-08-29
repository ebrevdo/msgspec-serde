from __future__ import annotations

import warnings

import pytest

import msgspec_serde
from msgspec_serde import (
    BaseType,
    FieldDefinition,
    GeneratedCodeVersionError,
    GeneratedCodeVersionWarning,
    ObjectDefinition,
    Schema,
    TypeReference,
    render_module,
)

INSTALLED_VERSION = msgspec_serde.__version__


@pytest.fixture(autouse=True)
def reset_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msgspec_serde, "__version__", "1.2.3")
    monkeypatch.setattr(msgspec_serde, "warn_on_older_runtime", True)
    monkeypatch.setattr(msgspec_serde, "_warned_version_pairs", set())


@pytest.mark.parametrize("generated_version", ["1.1.0", "1.2.0"])
def test_same_or_older_generated_version_does_not_warn(
    generated_version: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msgspec_serde._check_generated_code_version(generated_version)


def test_older_minor_runtime_warns_once() -> None:
    with pytest.warns(GeneratedCodeVersionWarning, match="runtime 1.2.3 is older"):
        msgspec_serde._check_generated_code_version("1.3.0")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msgspec_serde._check_generated_code_version("1.3.0")


def test_older_minor_runtime_warning_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgspec_serde, "warn_on_older_runtime", False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msgspec_serde._check_generated_code_version("1.3.0")


@pytest.mark.parametrize("generated_version", ["0.9.0", "2.0.0"])
def test_major_version_mismatch_raises_when_warnings_are_disabled(
    generated_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(msgspec_serde, "warn_on_older_runtime", False)

    with pytest.raises(GeneratedCodeVersionError, match="major versions differ"):
        msgspec_serde._check_generated_code_version(generated_version)


def test_rendered_module_embeds_and_checks_generator_version() -> None:
    declaration_file = "//versioned.fbs"
    root = ObjectDefinition(
        name="Example.Versioned",
        fields=(
            FieldDefinition(
                name="value",
                type=TypeReference(base_type=BaseType.INT),
                id=0,
                offset=4,
            ),
        ),
        declaration_file=declaration_file,
    )

    source = render_module(
        Schema(objects=(root,), enums=(), root_table=root.name),
        declaration_file,
    )

    check_import = "from msgspec_serde import _check_generated_code_version"
    runtime_import = "from msgspec_serde import (  # noqa: E402"
    assert source.index(check_import) < source.index(runtime_import)
    assert (
        f"__msgspec_serde_generated_version__ = {INSTALLED_VERSION!r}" in source
    )
    check_call = "_check_generated_code_version("
    assert source.index(check_call) < source.index(runtime_import)
