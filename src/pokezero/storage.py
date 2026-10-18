"""Artifact storage behind a small interface so S3/GCS backends can be added later.

Only ``LocalStore`` exists today. Keys are '/'-separated relative paths.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Protocol

import torch


class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def local_path(self, key: str) -> Path | None:
        """A filesystem path for ``key`` if the backend is local, else None."""
        ...


class LocalStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key escapes store root: {key}")
        return p

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic: readers never see a half-written checkpoint

    def get_bytes(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root.resolve()
        return sorted(
            str(p.relative_to(base))
            for p in base.rglob("*")
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        )

    def local_path(self, key: str) -> Path | None:
        return self._p(key)


def save_checkpoint(store: ArtifactStore, key: str, payload: dict[str, Any]) -> None:
    buf = io.BytesIO()
    torch.save(payload, buf)
    store.put_bytes(key, buf.getvalue())


def load_checkpoint(store: ArtifactStore, key: str) -> dict[str, Any]:
    return torch.load(io.BytesIO(store.get_bytes(key)), map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
