"""Demonstrate forward- and backward-compatible FlatBuffers readers."""

import numpy as np

from schema_evolution_generated.example.evolution.reading_v1 import (
    Reading as ReadingV1,
)
from schema_evolution_generated.example.evolution.reading_v1 import (
    ReadingView as ReadingViewV1,
)
from schema_evolution_generated.example.evolution.reading_v2 import (
    Reading as ReadingV2,
)
from schema_evolution_generated.example.evolution.reading_v2 import (
    ReadingView as ReadingViewV2,
)


def main() -> None:
    old_buffer = ReadingV1(
        id=7,
        label="legacy sensor",
        samples=np.array([1.25, 2.5], dtype=np.float32),
    ).to_flatbuffer()
    upgraded = ReadingViewV2.from_buffer(old_buffer)
    assert upgraded.samples is not None
    print(
        "v2 reading v1:",
        upgraded.label,
        upgraded.samples.tolist(),
        upgraded.quality,
        upgraded.note,
    )

    new_buffer = ReadingV2(
        id=9,
        label="current sensor",
        samples=np.array([3.5], dtype=np.float32),
        quality=87,
        note="added by schema v2",
    ).to_flatbuffer()
    legacy = ReadingViewV1.from_buffer(new_buffer)
    assert legacy.samples is not None
    print("v1 reading v2:", legacy.label, legacy.samples.tolist())


if __name__ == "__main__":
    main()
