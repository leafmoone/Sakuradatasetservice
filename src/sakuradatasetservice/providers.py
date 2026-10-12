"""Two-operation provider seam and a filesystem provider for local corpora."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    RemoteShardRecord,
    ShardRecord,
    build_dataset_manifest,
)
from .modelscope import (
    DatasetTransportError,
    DownloadProgress,
    FetchedShard,
    fetch_dataset_shard,
)


class DatasetProvider(Protocol):
    def fetch_manifest(self, source: DatasetSourceIdentity) -> DatasetManifest: ...

    def fetch_shard(
        self,
        manifest: DatasetManifest,
        shard_path: str,
        cache_root: Path,
        *,
        progress: DownloadProgress | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FetchedShard: ...


class LocalDatasetProvider:
    """Read normalized .tar files under one trusted local directory."""

    stream_chunk_bytes = 1048576
    streams_per_shard = 1

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("local provider root must be a directory")

    def _path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise DatasetTransportError("local shard escapes provider root")
        return path

    def fetch_manifest(self, source: DatasetSourceIdentity) -> DatasetManifest:
        records = tuple(
            RemoteShardRecord(
                path=p.relative_to(self.root).as_posix(),
                bytes=self._path(p.relative_to(self.root).as_posix()).stat().st_size,
            )
            for p in sorted(self.root.rglob("*.tar"))
        )
        return build_dataset_manifest(source, records)

    def fetch_shard(
        self,
        manifest: DatasetManifest,
        shard_path: str,
        cache_root: Path,
        *,
        progress: DownloadProgress | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FetchedShard:
        return fetch_dataset_shard(
            self,
            manifest,
            shard_path,
            cache_root,
            progress=progress,
            cancelled=cancelled,
        )

    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: Writer,
        *,
        start_offset: int = 0,
        end_offset: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        path = self._path(shard.path)
        if path.stat().st_size != shard.bytes:
            raise DatasetTransportError("local shard size differs from manifest")
        end = shard.bytes - 1 if end_offset is None else end_offset
        if not 0 <= start_offset <= end < shard.bytes:
            raise DatasetTransportError("invalid local shard range")
        with path.open("rb") as source:
            source.seek(start_offset)
            remaining = end - start_offset + 1
            while remaining:
                if cancelled is not None and cancelled():
                    raise DatasetTransportError("local shard download was cancelled")
                chunk = source.read(min(remaining, self.stream_chunk_bytes))
                if not chunk:
                    raise DatasetTransportError("local shard ended early")
                output.write(chunk)
                remaining -= len(chunk)


class Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...
