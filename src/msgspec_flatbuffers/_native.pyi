from collections.abc import Sequence
from typing import Any

type BasicField = tuple[int, str, str, Any, bool, bool]

class BasicPlan:
    def __init__(
        self,
        fields: Sequence[BasicField],
        identifier: str | None = None,
    ) -> None: ...

    def pack(
        self,
        model: object,
        *,
        size_prefixed: bool = False,
        initial_size: int = 0,
    ) -> bytes: ...

class NativePlan:
    def __init__(self, data: bytes) -> None: ...

    def bind_types(self, types: dict[str, type[object]]) -> None: ...

    def pack(
        self,
        root: str,
        model: object,
        *,
        identifier: str | None = None,
        size_prefixed: bool = False,
        initial_size: int = 0,
    ) -> memoryview: ...
