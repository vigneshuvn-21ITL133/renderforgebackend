"""File storage for uploaded photos and generated meshes.

``LocalStorage`` writes under a root directory. A cloud backend such as
Cloudflare R2 only needs to implement the same four methods.
"""

from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return path

    def write(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def path(self, key: str) -> Path:
        return self._resolve(key)

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)
