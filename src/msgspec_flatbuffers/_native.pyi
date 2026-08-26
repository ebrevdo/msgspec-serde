from collections.abc import Sequence
from typing import Any

class NativeBuffer:
    @property
    def _allocation_size(self) -> int: ...

class NativeModelTypes: ...

class NativePlan:
    def __init__(self, data: bytes) -> None: ...

    def bind_types(self, types: dict[str, type[object]]) -> None: ...

    def model_types(
        self,
        generated: type[object],
        requested: type[object],
        bindings: Sequence[tuple[object, ...]],
    ) -> NativeModelTypes: ...

    def unpack(
        self,
        root: str,
        buffer: bytes | bytearray | memoryview,
        *,
        identifier: str | None = None,
        offset: int = 0,
        size_prefixed: bool = False,
        check_identifier: bool = True,
        model_types: NativeModelTypes | None = None,
        dynamic_overrides: object | None = None,
    ) -> Any: ...

    def unpack_view(
        self,
        object: str,
        buffer: bytes | bytearray | memoryview,
        offset: int,
        *,
        model_types: NativeModelTypes | None = None,
        dynamic_overrides: object | None = None,
    ) -> Any: ...

    def pack(
        self,
        root: str,
        model: object,
        *,
        identifier: str | None = None,
        size_prefixed: bool = False,
        initial_size: int = 0,
    ) -> memoryview: ...
