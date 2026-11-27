"""File storage for uploaded photos and generated meshes.

``Storage`` is the interface the services depend on. ``LocalStorage`` writes under a root
directory. A cloud backend (Cloudflare R2, S3) only has to implement the same async methods.
File access runs on a worker thread so a multi-megabyte mesh never blocks the event loop.
"""

import asyncio
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    async def write(self, key: str, data: bytes) -> None: ...
    async def read(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return path

    async def write(self, key: str, data: bytes) -> None:
        path = self._resolve(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self._resolve(key).read_bytes)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(path.unlink, True)

    def path(self, key: str) -> Path:
        """Local filesystem path, used to stream downloads with FileResponse."""
        return self._resolve(key)
