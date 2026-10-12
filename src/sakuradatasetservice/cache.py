"""Single-process shard cache with explicit byte quota and LRU eviction."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sakuradatasetservice.manifest import DatasetManifest
from sakuradatasetservice.modelscope import (
    DownloadProgress,
    FetchedShard,
)
from sakuradatasetservice.providers import DatasetProvider


class ShardCacheError(RuntimeError):
    """The shard cache cannot satisfy its explicit quota."""


@dataclass(frozen=True)
class CacheQuota:
    low_bytes: int
    high_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.low_bytes) is not int
            or type(self.high_bytes) is not int
            or self.low_bytes < 0
            or self.low_bytes >= self.high_bytes
        ):
            raise ValueError("cache quota requires 0 <= low_bytes < high_bytes")


@dataclass(frozen=True)
class CachedShard:
    fetched: FetchedShard
    evicted_paths: tuple[str, ...]
    usage_bytes: int


class ShardCache:
    """Cache verified manifest shards without a service or background worker."""

    def __init__(
        self,
        root: Path,
        manifest: DatasetManifest,
        transport: DatasetProvider,
        quota: CacheQuota,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.transport = transport
        self.quota = quota
        self._lock = threading.Lock()
        self._reservations: dict[str, int] = {}

    def _path(self, shard_path: str) -> Path:
        return self.root / shard_path

    def _usage_bytes(self) -> int:
        total = 0
        for shard in self.manifest.shards:
            path = self._path(shard.path)
            if path.is_file():
                total += path.stat().st_size
        return total

    def usage_bytes(self) -> int:
        with self._lock:
            return self._usage_bytes()

    def _evict_for(
        self, shard_path: str, *, protected_paths: frozenset[str]
    ) -> tuple[str, ...]:
        shard = self.manifest.shard(shard_path)
        destination = self._path(shard_path)
        additional = 0 if destination.is_file() else shard.bytes
        usage = self._usage_bytes()
        reserved = sum(self._reservations.values())
        if additional > self.quota.high_bytes:
            raise ShardCacheError("shard is larger than the cache high watermark")
        if usage + reserved + additional <= self.quota.high_bytes:
            return ()

        target = max(self.quota.low_bytes, reserved + additional)
        candidates: list[tuple[int, str, Path, int]] = []
        for candidate in self.manifest.shards:
            if (
                candidate.path == shard_path
                or candidate.path in protected_paths
                or candidate.path in self._reservations
            ):
                continue
            path = self._path(candidate.path)
            if path.is_file():
                stat = path.stat()
                candidates.append(
                    (stat.st_mtime_ns, candidate.path, path, stat.st_size)
                )
        evicted: list[str] = []
        projected = usage + reserved + additional
        for _, relative_path, path, size in sorted(candidates):
            if projected <= target:
                break
            path.unlink()
            projected -= size
            evicted.append(relative_path)
        if projected > self.quota.high_bytes:
            raise ShardCacheError("cache quota cannot be met without protected shards")
        return tuple(evicted)

    def fetch(
        self,
        shard_path: str,
        *,
        protected_paths: frozenset[str] = frozenset(),
        progress: DownloadProgress | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> CachedShard:
        shard = self.manifest.shard(shard_path)
        with self._lock:
            if shard_path in self._reservations:
                raise ShardCacheError("shard download is already in flight")
            evicted = self._evict_for(shard_path, protected_paths=protected_paths)
            reserved = 0 if self._path(shard_path).is_file() else shard.bytes
            self._reservations[shard_path] = reserved
        try:
            fetched = self.transport.fetch_shard(
                self.manifest,
                shard_path,
                self.root,
                progress=progress,
                cancelled=cancelled,
            )
            with self._lock:
                try:
                    os.utime(fetched.path, None)
                except OSError:
                    raise ShardCacheError("cache file could not be updated") from None
                usage = self._usage_bytes()
                other_reserved = sum(
                    value
                    for path, value in self._reservations.items()
                    if path != shard_path
                )
                if usage + other_reserved > self.quota.high_bytes:
                    raise ShardCacheError("cache usage exceeds the high watermark")
        finally:
            with self._lock:
                self._reservations.pop(shard_path, None)
        return CachedShard(fetched=fetched, evicted_paths=evicted, usage_bytes=usage)

    def resumable_partials(self) -> tuple[tuple[str, int], ...]:
        """List interrupted downloads without discarding their progress."""

        found: list[tuple[str, int]] = []
        with self._lock:
            if self._reservations:
                raise ShardCacheError(
                    "partial inspection requires no in-flight downloads"
                )
            try:
                for shard in self.manifest.shards:
                    destination = self._path(shard.path)
                    partial = destination.with_name(f"{destination.name}.partial")
                    candidates = (partial,)
                    if partial.is_file():
                        candidates += tuple(
                            partial.parent.glob(f"{partial.name}.range-*")
                        )
                    for candidate in candidates:
                        if candidate.is_symlink():
                            raise ShardCacheError("cache partial must not be a symlink")
                        if candidate.exists() and not candidate.is_file():
                            raise ShardCacheError(
                                "cache partial must be a regular file"
                            )
                        if candidate.is_file():
                            found.append((shard.path, candidate.stat().st_size))
            except OSError:
                raise ShardCacheError("cache partial inspection failed") from None
        return tuple(found)
