"""Single-process shard cache with explicit byte quota and LRU eviction."""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sakuradatasetservice.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetManifestExistsError,
    parse_dataset_manifest_bytes,
    write_dataset_manifest,
)
from sakuradatasetservice.modelscope import DownloadProgress, FetchedShard
from sakuradatasetservice.providers import DatasetProvider


class ShardCacheError(RuntimeError):
    """The shard cache cannot satisfy its explicit quota or ownership."""


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
    """Cache one exact manifest in a dedicated, single-process directory.

    Quota counts payload and transfer files, including interrupted downloads.
    The readable manifest marker and filesystem allocation metadata are excluded.
    Keep service state and ownership locks outside this directory. Providers must
    use the shared transfer helper (or respect its two-shard-size peak footprint).
    """

    _MARKER = ".sakuradatasetservice-manifest.json"

    def __init__(
        self,
        root: Path,
        manifest: DatasetManifest,
        transport: DatasetProvider,
        quota: CacheQuota,
    ) -> None:
        self.root = root.absolute()
        self.manifest = manifest
        self.transport = transport
        self.quota = quota
        self._lock = threading.Lock()
        # Each value is the whole peak footprint, not additional unused space.
        self._reservations: dict[str, int] = {}
        self._shard_paths = frozenset(shard.path for shard in manifest.shards)
        self._bind_manifest()

    def _bind_manifest(self) -> None:
        marker = self.root / self._MARKER
        try:
            if self.root.is_symlink():
                raise ShardCacheError("cache root must not be a symlink")
            self.root.mkdir(parents=True, exist_ok=True)
            if not marker.exists() and not marker.is_symlink():
                if any(self.root.iterdir()):
                    raise ShardCacheError(
                        "nonempty cache root has no manifest ownership marker"
                    )
                try:
                    write_dataset_manifest(self.manifest, marker)
                except DatasetManifestExistsError:
                    pass
            if marker.is_symlink() or not marker.is_file():
                raise ShardCacheError("cache manifest marker must be a regular file")
            if parse_dataset_manifest_bytes(marker.read_bytes()) != self.manifest:
                raise ShardCacheError("cache root belongs to a different manifest")
        except (OSError, DatasetManifestError):
            raise ShardCacheError(
                "cache manifest ownership could not be verified"
            ) from None

    def _path(self, shard_path: str) -> Path:
        return self.root / shard_path

    def _file_owner(self, relative_path: str) -> str | None:
        if relative_path in self._shard_paths:
            return relative_path
        if relative_path.endswith(".partial.assembling"):
            candidate = relative_path.removesuffix(".partial.assembling")
        elif relative_path.endswith(".partial"):
            candidate = relative_path.removesuffix(".partial")
        else:
            candidate = relative_path.rpartition(".partial.range-")[0]
        return candidate if candidate in self._shard_paths else None

    def _disk_usage(self) -> tuple[dict[str, int], int]:
        """Snapshot every payload file; unrecognized files still consume quota."""

        usage = dict.fromkeys(self._shard_paths, 0)
        other = 0
        try:
            for path in self.root.rglob("*"):
                if path == self.root / self._MARKER:
                    continue
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    # An active transfer may have just atomically renamed a file.
                    continue
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ShardCacheError("cache entries must be regular files")
                owner = self._file_owner(path.relative_to(self.root).as_posix())
                if owner is None:
                    other += info.st_size
                else:
                    usage[owner] += info.st_size
        except OSError:
            raise ShardCacheError("cache usage could not be inspected") from None
        return usage, other

    def _usage_bytes(self) -> int:
        usage, other = self._disk_usage()
        return sum(usage.values()) + other

    def usage_bytes(self) -> int:
        with self._lock:
            return self._usage_bytes()

    def _accounted_bytes(
        self, usage: dict[str, int], other: int, reservations: dict[str, int]
    ) -> int:
        # Partial bytes are already inside the peak reservation. Adding both would
        # reject valid concurrent downloads as their files grow on disk.
        return other + sum(
            max(size, reservations.get(path, 0)) for path, size in usage.items()
        )

    def _evict_for(
        self, shard_path: str, *, protected_paths: frozenset[str]
    ) -> tuple[tuple[str, ...], int]:
        shard = self.manifest.shard(shard_path)
        usage, other = self._disk_usage()
        destination = self._path(shard_path)
        if destination.is_file():
            if destination.stat().st_size != shard.bytes:
                raise ShardCacheError("cached shard differs from manifest")
            peak = usage[shard_path]
        else:
            # The transfer helper discards obsolete ranges and stale assemblies
            # before writing. Matching ranges + prefix fit in one shard; assembly
            # needs one more. Existing artifacts are never ignored at admission.
            peak = max(2 * shard.bytes, usage[shard_path])
        if peak > self.quota.high_bytes:
            raise ShardCacheError(
                "shard transfer peak exceeds the cache high watermark"
            )
        reservations = {**self._reservations, shard_path: peak}
        projected = self._accounted_bytes(usage, other, reservations)
        if projected <= self.quota.high_bytes:
            return (), peak

        target = max(self.quota.low_bytes, sum(reservations.values()))
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
                info = path.stat()
                candidates.append(
                    (info.st_mtime_ns, candidate.path, path, info.st_size)
                )
        evicted: list[str] = []
        for _, relative_path, path, size in sorted(candidates):
            if projected <= target:
                break
            path.unlink()
            projected -= size
            evicted.append(relative_path)
        if projected > self.quota.high_bytes:
            raise ShardCacheError("cache quota cannot be met without protected shards")
        return tuple(evicted), peak

    def fetch(
        self,
        shard_path: str,
        *,
        protected_paths: frozenset[str] = frozenset(),
        progress: DownloadProgress | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> CachedShard:
        self.manifest.shard(shard_path)
        with self._lock:
            if shard_path in self._reservations:
                raise ShardCacheError("shard download is already in flight")
            evicted, peak = self._evict_for(shard_path, protected_paths=protected_paths)
            self._reservations[shard_path] = peak
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
                disk_usage, other = self._disk_usage()
                usage = sum(disk_usage.values()) + other
                other_reservations = {
                    path: value
                    for path, value in self._reservations.items()
                    if path != shard_path
                }
                if (
                    self._accounted_bytes(disk_usage, other, other_reservations)
                    > self.quota.high_bytes
                ):
                    raise ShardCacheError("cache usage exceeds the high watermark")
        finally:
            with self._lock:
                self._reservations.pop(shard_path, None)
        return CachedShard(fetched=fetched, evicted_paths=evicted, usage_bytes=usage)

    def resumable_partials(self) -> tuple[tuple[str, int], ...]:
        """List retained partials, ranges and assemblies, including orphan ranges."""

        found: list[tuple[str, int]] = []
        with self._lock:
            if self._reservations:
                raise ShardCacheError(
                    "partial inspection requires no in-flight downloads"
                )
            # Validate the whole tree first, including symlinked directories.
            self._disk_usage()
            try:
                for path in sorted(self.root.rglob("*")):
                    relative = path.relative_to(self.root).as_posix()
                    owner = self._file_owner(relative)
                    if owner is not None and relative != owner and path.is_file():
                        found.append((owner, path.stat().st_size))
            except OSError:
                raise ShardCacheError("cache partial inspection failed") from None
        return tuple(found)
