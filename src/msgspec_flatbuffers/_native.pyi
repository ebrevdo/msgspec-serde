from typing import Any

class NativeBuffer:
    @property
    def _allocation_size(self) -> int: ...

class NativePlan:
    def __init__(self, data: bytes) -> None: ...

    def bind_types(self, types: dict[str, type[object]]) -> None: ...

    def unpack(
        self,
        root: str,
        buffer: bytes | bytearray | memoryview,
        *,
        identifier: str | None = None,
        offset: int = 0,
        size_prefixed: bool = False,
        check_identifier: bool = True,
    ) -> Any: ...

    def unpack_view(
        self,
        object: str,
        buffer: bytes | bytearray | memoryview,
        offset: int,
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
