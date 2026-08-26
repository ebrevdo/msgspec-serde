"""Run the examples from the project tutorial."""

from __future__ import annotations

from typing import ClassVar

import msgspec
import numpy as np

# These modules are created by the tutorial's generation command.
from tutorial.dynamic_envelope import (  # ty: ignore[unresolved-import]
    DynamicEnvelope,
    DynamicEnvelopePayload,
    DynamicEnvelopeView,
)
from tutorial.extension_list import (  # ty: ignore[unresolved-import]
    Extension,
    ExtensionList,
    ExtensionListView,
)
from tutorial.metric import Metric, MetricView  # ty: ignore[unresolved-import]
from tutorial.monster import (  # ty: ignore[unresolved-import]
    Color,
    Monster,
    MonsterView,
    Vec3,
    Weapon,
)
from tutorial.nested_envelope import (  # ty: ignore[unresolved-import]
    NestedEnvelope,
    NestedEnvelopeView,
)
from tutorial.pet_list import (  # ty: ignore[unresolved-import]
    Cat,
    CatView,
    Dog,
    Pet,
    PetList,
    PetListView,
)

from msgspec_flatbuffers import DynamicModelOverrides, dec_hook, enc_hook


class ValidatedWeapon(Weapon, dict=True):
    validation_kind: ClassVar[str] = "weapon"

    def __post_init__(self) -> None:
        if self.damage < 0:
            raise ValueError("damage cannot be negative")
        self.was_validated = True


class ValidatedMonster(Monster, dict=True):
    weapons: list[ValidatedWeapon] | None = None

    def __post_init__(self) -> None:
        self.was_validated = True


class ValidatedMetric(Metric, dict=True):
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name cannot be empty")
        self.was_validated = True


dynamic_overrides = DynamicModelOverrides({Metric: ValidatedMetric})


def main() -> None:
    monster = Monster(
        pos=Vec3(x=1.0, y=2.0, z=3.0),
        name="Orc",
        inventory=b"\x00\x01\x02",
        color=Color.Green,
        weapons=[Weapon(name="Axe", damage=12)],
        scores=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        tags=["boss", "cave"],
        optional_energy=80,
    )
    buffer = monster.to_flatbuffer()
    view = MonsterView.from_buffer(buffer)
    position = view.pos
    weapons = view.weapons

    assert buffer.readonly
    assert view.name == "Orc"
    assert position is not None
    assert position.z == 3.0
    assert view.color is Color.Green
    assert weapons is not None
    assert weapons[0].damage == 12
    assert weapons[0] is weapons[0]

    restored = view.to_model()
    assert restored == monster
    assert restored.scores is not None
    assert restored.scores.flags.owndata
    assert restored.scores.flags.writeable
    assert Monster.from_flatbuffer(buffer) == monster

    validated = ValidatedMonster.from_flatbuffer(buffer)
    validated_from_view = view.to_model(ValidatedMonster)
    assert validated.was_validated
    assert validated.weapons is not None
    assert isinstance(validated.weapons[0], ValidatedWeapon)
    assert validated_from_view.was_validated

    encoded = msgspec.json.encode(monster, enc_hook=enc_hook)
    decoded = msgspec.json.decode(encoded, type=Monster, dec_hook=dec_hook)
    assert decoded == monster

    pets = PetList(
        favorite=Cat(name="Miso"),
        residents=[Cat(name="Miso"), Dog(name="Pip")],
    )
    pets_view = PetListView.from_buffer(pets.to_flatbuffer())
    assert pets_view.favorite_type is Pet.Cat
    assert isinstance(pets_view.favorite, CatView)

    nested = NestedEnvelope(snapshot=pets, note="nightly snapshot")
    nested_view = NestedEnvelopeView.from_buffer(nested.to_flatbuffer())
    snapshot = nested_view.snapshot
    snapshot_raw = nested_view.snapshot_raw
    assert snapshot is not None
    assert snapshot.favorite_type is Pet.Cat
    assert snapshot_raw is not None
    assert snapshot_raw.readonly

    metric = Metric(
        name="latency",
        values=np.array([1.25, 2.5], dtype=np.float32),
    )
    dynamic = DynamicEnvelope(payload=DynamicEnvelopePayload(metric))
    dynamic_buffer = dynamic.to_flatbuffer()
    dynamic_view = DynamicEnvelopeView.from_buffer(dynamic_buffer)
    dynamic_payload = dynamic_view.payload
    dynamic_raw = dynamic_view.payload_raw
    assert dynamic_payload is not None
    assert dynamic_payload.tag == "Tutorial.Metric"
    assert isinstance(dynamic_payload.value, MetricView)
    assert dynamic_view.payload_type_raw == "Tutorial.Metric"
    assert dynamic_raw is not None
    assert dynamic_raw.readonly
    assert DynamicEnvelope.from_flatbuffer(dynamic_buffer) == dynamic

    validated_dynamic = DynamicEnvelope.from_flatbuffer(
        dynamic_buffer,
        dynamic_overrides=dynamic_overrides,
    )
    assert validated_dynamic.payload is not None
    assert isinstance(validated_dynamic.payload.value, ValidatedMetric)
    dynamic_json = msgspec.json.encode(dynamic, enc_hook=enc_hook)
    validated_json = msgspec.json.decode(
        dynamic_json,
        type=DynamicEnvelope,
        dec_hook=dynamic_overrides.dec_hook,
    )
    assert validated_json.payload is not None
    assert isinstance(validated_json.payload.value, ValidatedMetric)

    metric_bytes = bytes(metric.to_flatbuffer())
    extension_list = ExtensionList(
        extensions=[Extension(type_id=0xA813, data=metric_bytes)]
    )
    extension_list_view = ExtensionListView.from_buffer(extension_list.to_flatbuffer())
    extensions = extension_list_view.extensions
    assert extensions is not None
    metric_view = extensions[0].data_as(MetricView)
    assert metric_view is not None
    assert metric_view.name == "latency"
    assert extensions[0].data_as(MetricView) is metric_view

    framed = monster.to_flatbuffer(size_prefixed=True)
    assert Monster.from_flatbuffer(framed, size_prefixed=True) == monster

    print(view.name, weapons[0].name)
    print(pets_view.favorite_type.name, nested_view.note)
    print(dynamic_payload.tag)
    print(encoded.decode())


if __name__ == "__main__":
    main()
