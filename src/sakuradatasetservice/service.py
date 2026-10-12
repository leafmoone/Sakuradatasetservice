"""Local shard cache service used by the training process."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import random
import secrets
import socket
import stat
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

from sakuradatasetservice.cache import CachedShard, ShardCache, ShardCacheError
from sakuradatasetservice.manifest import DatasetManifest
from sakuradatasetservice.modelscope import DatasetTransientError, DatasetTransportError
from sakuradatasetservice.service_protocol import (
    MAX_SERVICE_FRAME_BYTES,
    SERVICE_PROTOCOL_VERSION,
    DataServiceProtocolError,
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
    canonical_json_bytes,
    parse_frame,
)
from sakuradatasetservice.validation import (
    ValidationSelection,
    validate_selection_manifest,
)

_QUEUE_SCHEMA_VERSION = 2

_CLIENT_DISCONNECTED_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ENOTCONN,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)

# Leases retired by an epoch rollover are kept (bounded) so that the ranks'
# late ACKs — their workers may still be streaming the retired shards behind
# a full prefetch queue — are accepted idempotently instead of raising.
_RETIRED_LEASE_CAP = 16384


class DataServiceError(RuntimeError):
    pass


class _ServiceStopping(Exception):
    pass


def _log(message: str) -> None:
    print(f"[data-server] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class DataServiceLimits:
    download_concurrency: int
    verified_shard_lookahead: int
    lease_channel_capacity: int
    ack_channel_capacity: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.download_concurrency,
                self.verified_shard_lookahead,
                self.lease_channel_capacity,
                self.ack_channel_capacity,
            )
        ):
            raise ValueError("data service limits must be positive integers")


@dataclass(frozen=True, slots=True)
class DataServiceStats:
    in_flight_downloads: int
    ready_shards: int
    outstanding_leases: int


@dataclass(frozen=True, slots=True)
class _Row:
    path: str
    status: Literal["pending", "active", "completed"]


@dataclass(frozen=True, slots=True)
class _RetiredLease:
    """A lease retired by an epoch-boundary force rollover, still awaiting
    the rank's late ACK so it can be accepted idempotently."""

    path: str
    state_revision: int
    worker_id: int


@dataclass(frozen=True, slots=True)
class _QueueState:
    cycle: int
    rows: tuple[_Row, ...]

    @property
    def completed_epochs(self) -> int:
        return self.cycle

    @property
    def current_epoch(self) -> int:
        return self.completed_epochs + 1


class _QueueStore:
    def __init__(
        self,
        path: Path,
        manifest: DatasetManifest,
        selection: ValidationSelection,
    ) -> None:
        validate_selection_manifest(selection, manifest)
        self.path = path
        excluded = frozenset(selection.shard_paths)
        self.paths = frozenset(
            item.path for item in manifest.shards if item.path not in excluded
        )
        if not self.paths:
            raise DataServiceError("validation selection leaves no training shards")

    def new(self, cycle: int) -> _QueueState:
        # Canonical input order: self.paths is a frozenset whose
        # iteration order is per-process (string hash
        # randomization). The shuffle must start from
        # sorted(paths) — the only input arrangement that is
        # bit-identical across processes, restarts, and experiment
        # arms — otherwise two fresh cycle-0 runs consume different
        # sample sequences despite identical manifest and seed.
        paths = sorted(self.paths)
        # Deterministic per-cycle order: the cross-cycle lookahead in
        # _schedule_lookahead pre-downloads the NEXT cycle's leading shards
        # while the current cycle still drains. Those pre-downloads only
        # match the order the real epoch roll produces when every cycle's
        # shuffle is a pure function of its cycle number, so the boundary
        # finds a warm pipeline instead of a cold-start download gap that
        # starves the DDP ranks and trips the 300s NCCL watchdog.
        seed = (0x9E3779B97F4A7C15 * (cycle + 1)) & 0xFFFFFFFFFFFFFFFF
        random.Random(seed).shuffle(paths)
        return _QueueState(cycle, tuple(_Row(path, "pending") for path in paths))

    def _validate(self, state: _QueueState) -> None:
        paths = tuple(row.path for row in state.rows)
        if (
            type(state.cycle) is not int
            or state.cycle < 0
            or frozenset(paths) != self.paths
            or len(paths) != len(set(paths))
            or any(
                row.status not in {"pending", "active", "completed"}
                for row in state.rows
            )
        ):
            raise DataServiceError("data queue state is invalid")

    def load(self) -> _QueueState | None:
        if self.path.is_symlink():
            raise DataServiceError("data queue state may not be a symlink")
        if not self.path.exists():
            return None
        try:
            document = json.loads(self.path.read_bytes())
            if not isinstance(document, dict):
                raise TypeError
            raw = cast(dict[str, Any], document)
            schema_version = raw.get("schema_version")
            if schema_version == 1:
                if set(raw) != {"cycle", "rows", "schema_version"}:
                    raise ValueError
            elif schema_version == _QUEUE_SCHEMA_VERSION:
                if set(raw) != {
                    "completed_epochs",
                    "current_epoch",
                    "cycle",
                    "rows",
                    "schema_version",
                }:
                    raise ValueError
            else:
                raise ValueError
            cycle = raw["cycle"]
            if type(cycle) is not int:
                raise TypeError
            if schema_version == _QUEUE_SCHEMA_VERSION and (
                type(raw["completed_epochs"]) is not int
                or type(raw["current_epoch"]) is not int
                or raw["completed_epochs"] != cycle
                or raw["current_epoch"] != cycle + 1
            ):
                raise ValueError
            raw_rows = raw["rows"]
            if not isinstance(raw_rows, list):
                raise TypeError
            rows: list[_Row] = []
            for item in cast(list[object], raw_rows):
                if not isinstance(item, dict):
                    raise TypeError
                item_mapping = cast(dict[object, object], item)
                if set(item_mapping) != {"path", "status"}:
                    raise TypeError
                row = cast(dict[str, object], item_mapping)
                status = row["status"]
                rows.append(
                    _Row(
                        path=cast(str, row["path"]),
                        status=(
                            "pending"
                            if status == "active"
                            else cast(Literal["pending", "active", "completed"], status)
                        ),
                    )
                )
            state = _QueueState(cycle, tuple(rows))
            self._validate(state)
            return state
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            raise DataServiceError("data queue state could not be loaded") from None

    def save(self, state: _QueueState) -> None:
        self._validate(state)
        payload = canonical_json_bytes(
            {
                "completed_epochs": state.completed_epochs,
                "current_epoch": state.current_epoch,
                "cycle": state.cycle,
                "rows": [
                    {"path": row.path, "status": row.status} for row in state.rows
                ],
                "schema_version": _QUEUE_SCHEMA_VERSION,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            raise DataServiceError("data queue state could not be saved") from None
        finally:
            temporary.unlink(missing_ok=True)


class DataSupplyService:
    """Own the cache and issue one local shard lease per data worker."""

    def __init__(
        self,
        manifest: DatasetManifest,
        validation_selection: ValidationSelection,
        cache: ShardCache,
        mainset_path: Path,
        ownership_lock_path: Path,
        identity: DataServiceSessionIdentity,
        limits: DataServiceLimits,
    ) -> None:
        if manifest.dataset_id != identity.dataset_id:
            raise DataServiceError("service dataset differs from session")
        if (
            min(
                limits.lease_channel_capacity,
                limits.ack_channel_capacity,
                limits.verified_shard_lookahead,
            )
            < identity.worker_count
        ):
            raise DataServiceError(
                "data service capacities are smaller than worker count"
            )
        if not ownership_lock_path.is_absolute() or ownership_lock_path.is_symlink():
            raise DataServiceError("service ownership lock path is invalid")
        self.manifest = manifest
        self.cache = cache
        self.identity = identity
        self.limits = limits
        self.store = _QueueStore(mainset_path, manifest, validation_selection)
        self.ownership_lock_path = ownership_lock_path
        self._state: _QueueState | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=limits.download_concurrency,
            thread_name_prefix="sakuradatasetservice-data-fetch",
        )
        self._futures: dict[str, Future[CachedShard]] = {}
        self._ready: dict[str, CachedShard] = {}
        self._outstanding: dict[str, ShardLeaseDescriptor] = {}
        self._worker_leases: dict[int, str] = {}
        self._retired_leases: OrderedDict[str, _RetiredLease] = OrderedDict()
        self._revision = 0
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self._ownership_handle: BinaryIO | None = None
        self._external_stop: threading.Event | None = None
        self._shutdown = threading.Event()

    @property
    def state(self) -> _QueueState:
        if self._state is None:
            raise DataServiceError("data queue is not loaded")
        return self._state

    def start(self, stop_event: threading.Event | None = None) -> None:
        with self._lock:
            if self._started or self._closed:
                raise DataServiceError("data service lifecycle is invalid")
            self.store.path.parent.mkdir(parents=True, exist_ok=True)
            self.cache.root.mkdir(parents=True, exist_ok=True)
            try:
                self.ownership_lock_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self.ownership_lock_path,
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                )
                handle = os.fdopen(descriptor, "a+b")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                raise DataServiceError("another data service owns the cache") from None
            self._ownership_handle = handle
            self._external_stop = stop_event
            resumable = self.cache.resumable_partials()
            if resumable:
                resumable_mib = sum(size for _, size in resumable) / 1024**2
                _log(
                    f"恢复未完成下载: {len({path for path, _ in resumable})} 个分片, "
                    f"已保留 {resumable_mib:.0f} MiB"
                )
            state = self.store.load()
            if state is None:
                state = self.store.new(0)
            self.store.save(state)
            self._state = state
            self._started = True
            _log(
                f"训练分片队列: epoch={state.current_epoch}, "
                f"completed_epochs={state.completed_epochs}, cycle={state.cycle}, "
                f"shards={len(state.rows)}, "
                f"workers={self.identity.worker_count}"
            )
            _log(
                f"下载并发: shards={self.limits.download_concurrency}, "
                f"lookahead={self.limits.verified_shard_lookahead}"
            )
            self._schedule_lookahead()

    def _cancelled(self) -> bool:
        return self._shutdown.is_set() or (
            self._external_stop is not None and self._external_stop.is_set()
        )

    def _require_running(self) -> None:
        if not self._started or self._closed:
            raise DataServiceError("data service is not running")

    def _pending_paths(self) -> tuple[str, ...]:
        leased = {item.record.path for item in self._outstanding.values()}
        return tuple(
            row.path
            for row in self.state.rows
            if row.status == "pending" and row.path not in leased
        )

    def _protected_paths(self) -> frozenset[str]:
        return frozenset(
            {
                *self._ready,
                *self._futures,
                *(item.record.path for item in self._outstanding.values()),
            }
        )

    def _schedule_lookahead(self) -> None:
        occupied = len(self._ready) + len(self._futures)
        candidates = list(self._pending_paths())
        if len(candidates) < self.limits.verified_shard_lookahead:
            # Cross-cycle lookahead: once the current cycle is about to drain,
            # start downloading the NEXT cycle's leading shards (deterministic
            # per-cycle order) so the epoch boundary stays warm instead of a
            # cold-start download gap that starves the DDP ranks.
            seen = set(candidates)
            for row in self.store.new(self.state.cycle + 1).rows:
                if row.path in seen or occupied >= self.limits.verified_shard_lookahead:
                    break
                seen.add(row.path)
                candidates.append(row.path)
        for path in candidates:
            if occupied >= self.limits.verified_shard_lookahead:
                break
            if path in self._ready or path in self._futures:
                continue
            _log(f"预取分片: {path}")
            self._futures[path] = self._executor.submit(
                self.cache.fetch,
                path,
                protected_paths=self._protected_paths(),
                progress=self._download_progress(path),
                cancelled=self._cancelled,
            )
            occupied += 1

    @staticmethod
    def _download_progress(path: str) -> Callable[[int, int, float, float], None]:
        def report(
            downloaded: int,
            total: int,
            elapsed: float,
            bytes_per_second: float,
        ) -> None:
            mib = 1024 * 1024
            percent = 100.0 * downloaded / total
            if elapsed == 0.0:
                _log(
                    f"断点续传: {path} {downloaded / mib:.0f}/{total / mib:.0f} MiB "
                    f"({percent:.1f}%)"
                )
                return
            _log(
                f"下载进度: {path} {downloaded / mib:.0f}/{total / mib:.0f} MiB "
                f"({percent:.1f}%), {bytes_per_second / mib:.1f} MiB/s"
            )

        return report

    def _collect_completed(self) -> None:
        for path, future in tuple(self._futures.items()):
            if not future.done():
                continue
            self._futures.pop(path, None)
            try:
                result = future.result()
            except DatasetTransientError as error:
                if self._cancelled():
                    raise _ServiceStopping from None
                _log(f"分片下载暂时失败，保留进度并重试: {path}: {error}")
                continue
            except (DatasetTransportError, OSError, ShardCacheError) as error:
                if self._cancelled():
                    raise _ServiceStopping from None
                raise DataServiceError(
                    f"data shard could not be prepared: {path}: {error}"
                ) from None
            self._ready[path] = result
            source = "缓存" if result.fetched.cache_hit else "下载"
            _log(f"分片就绪({source}): {path}")

    def _wait_for_ready_count(self, count: int) -> None:
        announced = False
        while len(self._ready) < count:
            if self._cancelled():
                raise _ServiceStopping
            self._collect_completed()
            if len(self._ready) >= count:
                return
            self._schedule_lookahead()
            if not self._futures:
                raise DataServiceError("data service has no shard to prepare")
            if not announced:
                _log(f"等待任意 {count} 个分片就绪")
                announced = True
            wait(
                tuple(self._futures.values()),
                timeout=0.25,
                return_when=FIRST_COMPLETED,
            )

    def _take_ready(self) -> tuple[str, CachedShard]:
        self._wait_for_ready_count(1)
        for path in self._pending_paths():
            cached = self._ready.pop(path, None)
            if cached is not None:
                return path, cached
        raise DataServiceError("ready shard is absent from the pending queue")

    def wait_until_ready(self) -> bool:
        with self._lock:
            self._require_running()
            if not self._pending_paths():
                # A process starting exactly at an epoch boundary must not
                # fail fast while un-ACKed leases block the rollover; give
                # the boundary one force-rollover chance, same as lease().
                self._force_roll_over_if_wedge()
            pending_count = len(self._pending_paths())
            if pending_count <= 0:
                raise DataServiceError("data service has no remaining training shard")
            ready_count = min(self.identity.worker_count, pending_count)
            if ready_count < self.identity.worker_count:
                _log(
                    f"epoch 尾部仅剩 {ready_count} 个分片; "
                    f"允许少于 {self.identity.worker_count} 个 worker 的启动屏障"
                )
            try:
                self._wait_for_ready_count(ready_count)
                return True
            except _ServiceStopping:
                return False

    def _state_with_status(
        self, path: str, status: Literal["pending", "active", "completed"]
    ) -> _QueueState:
        matches = [row for row in self.state.rows if row.path == path]
        if len(matches) != 1:
            raise DataServiceError("leased shard is absent from queue")
        return replace(
            self.state,
            rows=tuple(
                replace(row, status=status) if row.path == path else row
                for row in self.state.rows
            ),
        )

    def _commit_state(self, state: _QueueState) -> None:
        self.store.save(state)
        self._state = state
        self._revision += 1

    def _replace_status(
        self, path: str, status: Literal["pending", "active", "completed"]
    ) -> None:
        self._commit_state(self._state_with_status(path, status))

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        with self._lock:
            self._require_running()
            if (
                type(worker_id) is not int
                or not 0 <= worker_id < self.identity.worker_count
            ):
                raise DataServiceError("worker identity is invalid")
            existing_lease_id = self._worker_leases.get(worker_id)
            if existing_lease_id is not None:
                descriptor = self._outstanding.get(existing_lease_id)
                if descriptor is None:
                    raise DataServiceError("worker lease state is inconsistent")
                _log(f"worker={worker_id} 继续分片: {descriptor.record.path}")
                return descriptor
            if len(self._outstanding) >= self.limits.lease_channel_capacity:
                raise DataServiceError("lease capacity is exhausted")
            if not self._pending_paths():
                self._force_roll_over_if_wedge()
            if not self._pending_paths():
                return None
            path, cached = self._take_ready()
            self._replace_status(path, "active")
            lease_id = secrets.token_urlsafe(16)
            descriptor = ShardLeaseDescriptor(
                lease_id=lease_id,
                worker_id=worker_id,
                cycle_index=self.state.cycle,
                state_revision=self._revision,
                record=self.manifest.shard(path),
                local_path=cached.fetched.path,
            )
            self._outstanding[lease_id] = descriptor
            self._worker_leases[worker_id] = lease_id
            _log(f"worker={worker_id} 加载分片: {path}")
            self._schedule_lookahead()
            return descriptor

    def acknowledge(self, lease_id: str, worker_id: int, state_revision: int) -> None:
        with self._lock:
            self._require_running()
            descriptor = self._outstanding.get(lease_id)
            if descriptor is None:
                retired = self._retired_leases.get(lease_id)
                if (
                    retired is not None
                    and retired.state_revision == state_revision
                    and retired.worker_id == worker_id
                ):
                    # The epoch-boundary force rollover already completed this
                    # lease; the rank's late ACK is safe to swallow.
                    _log(
                        f"worker={worker_id} 完成分片(幂等): {retired.path} "
                        f"(租约已在 epoch 边界 rollover 中退休，重复 ACK 忽略)"
                    )
                    return
                raise DataServiceError("ACK does not match an active lease")
            if (
                descriptor.worker_id != worker_id
                or descriptor.state_revision != state_revision
                or self._worker_leases.get(worker_id) != lease_id
            ):
                raise DataServiceError("ACK does not match an active lease")
            self._complete_lease(descriptor)

    def _complete_lease(self, descriptor: ShardLeaseDescriptor) -> None:
        """Mark a lease's shard completed (rollover-aware). Caller holds _lock."""
        completed_state = self._state_with_status(descriptor.record.path, "completed")
        completed_epoch: int | None = None
        if all(row.status == "completed" for row in completed_state.rows):
            completed_epoch = completed_state.current_epoch
            next_state = self.store.new(completed_state.cycle + 1)
            if next_state.completed_epochs != completed_epoch:
                raise DataServiceError("data epoch rollover identity is invalid")
            self._commit_state(next_state)
        else:
            self._commit_state(completed_state)
        del self._outstanding[descriptor.lease_id]
        del self._worker_leases[descriptor.worker_id]
        _log(f"worker={descriptor.worker_id} 完成分片: {descriptor.record.path}")
        if completed_epoch is not None:
            _log(
                f"数据 epoch 完成: epoch={completed_epoch}, "
                f"completed_epochs={self.state.completed_epochs}, "
                f"shards={len(self.state.rows)}; "
                f"已重建 epoch={self.state.current_epoch} 队列"
            )
        self._schedule_lookahead()

    def _force_roll_over_if_wedge(self) -> None:
        """Force-complete the epoch when the queue has drained but leases are
        still outstanding.

        At an epoch boundary the ranks can otherwise wait circularly: the rank
        whose shards are all done starves in its drain loop waiting for the
        rollover, while the rank holding the last un-ACKed leases sits in the
        cross-rank allreduce and can no longer process the completions (and
        therefore ACK the leases) that would trigger the rollover.  Completing
        the epoch here — retiring the still-outstanding leases so their late
        ACKs become no-ops — is lossless for a cyclic dataset: the shard data
        already buffered inside a rank is still consumed in order, and those
        shards are re-served (and fully consumed) in the next cycle.  This
        must stay a lease-time check rather than a time threshold: a healthy
        lease outlives the NCCL watchdog by an order of magnitude
        (~26 min per shard vs 300 s), so no timeout value could both avoid
        force-completing healthy leases and fire before the watchdog.
        Caller holds _lock.
        """
        if not self._outstanding:
            return
        completed_epoch = self.state.current_epoch
        retired = 0
        for descriptor in list(self._outstanding.values()):
            self._retire_lease(descriptor)
            retired += 1
        next_state = self.store.new(self.state.cycle + 1)
        if next_state.completed_epochs != completed_epoch:
            raise DataServiceError("data epoch rollover identity is invalid")
        self._commit_state(next_state)
        _log(
            f"epoch 边界强制完成: epoch={completed_epoch}, "
            f"退休未完成租约={retired}, "
            f"completed_epochs={self.state.completed_epochs}, "
            f"shards={len(self.state.rows)}; "
            f"已重建 epoch={self.state.current_epoch} 队列"
        )
        self._schedule_lookahead()

    def _retire_lease(self, descriptor: ShardLeaseDescriptor) -> None:
        """Move an outstanding lease to the retired set so the rank's late
        ACK (its worker may still be streaming the shard behind a full
        prefetch queue) is accepted as an idempotent no-op. Caller holds
        _lock."""
        self._retired_leases[descriptor.lease_id] = _RetiredLease(
            path=descriptor.record.path,
            state_revision=descriptor.state_revision,
            worker_id=descriptor.worker_id,
        )
        while len(self._retired_leases) > _RETIRED_LEASE_CAP:
            self._retired_leases.popitem(last=False)
        del self._outstanding[descriptor.lease_id]
        del self._worker_leases[descriptor.worker_id]

    @property
    def done(self) -> bool:
        return False

    @property
    def stats(self) -> DataServiceStats:
        with self._lock:
            return DataServiceStats(
                in_flight_downloads=sum(
                    not future.done() for future in self._futures.values()
                ),
                ready_shards=len(self._ready),
                outstanding_leases=len(self._outstanding),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown.set()
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._ownership_handle is not None:
                try:
                    fcntl.flock(self._ownership_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self._ownership_handle.close()
                    self._ownership_handle = None


def _recv_frame(connection: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while len(chunks) <= MAX_SERVICE_FRAME_BYTES:
        block = connection.recv(min(4096, MAX_SERVICE_FRAME_BYTES + 1 - len(chunks)))
        if not block:
            break
        chunks.extend(block)
        if chunks.endswith(b"\n"):
            break
    return parse_frame(bytes(chunks))


def _response(payload: dict[str, object]) -> bytes:
    framed = canonical_json_bytes(payload)
    if len(framed) > MAX_SERVICE_FRAME_BYTES:
        raise DataServiceError("service response exceeds frame limit")
    return framed


def _client_disconnected(error: OSError) -> bool:
    return (
        isinstance(
            error,
            (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
                TimeoutError,
            ),
        )
        or error.errno in _CLIENT_DISCONNECTED_ERRNOS
    )


class DataServiceServer:
    def __init__(
        self,
        service: DataSupplyService,
        socket_path: Path,
        *,
        request_timeout_seconds: float,
    ) -> None:
        if (
            not socket_path.is_absolute()
            or type(request_timeout_seconds) is not float
            or request_timeout_seconds <= 0.0
        ):
            raise ValueError("data service socket settings are invalid")
        self.service = service
        self.socket_path = socket_path
        self.request_timeout_seconds = request_timeout_seconds

    def _dispatch(self, request: dict[str, Any]) -> dict[str, object]:
        operation = request.get("op")
        if operation == "health":
            if set(request) != {"op", "protocol_version", "worker_count"}:
                raise DataServiceError("health request fields are invalid")
            if (
                request["protocol_version"] != SERVICE_PROTOCOL_VERSION
                or request["worker_count"] != self.service.identity.worker_count
            ):
                raise DataServiceError("client worker topology differs from service")
            return {
                "done": self.service.done,
                "ok": True,
                "protocol_version": SERVICE_PROTOCOL_VERSION,
                "session_identity": self.service.identity.as_dict(),
            }
        if operation == "lease":
            if set(request) != {"op", "session_id", "worker_id"}:
                raise DataServiceError("lease request fields are invalid")
            if request["session_id"] != self.service.identity.session_id:
                raise DataServiceError("data service session changed")
            descriptor = self.service.lease(request["worker_id"])
            return {
                "done": descriptor is None,
                "lease": None if descriptor is None else descriptor.as_dict(),
                "ok": True,
            }
        if operation == "ack":
            if set(request) != {
                "lease_id",
                "op",
                "session_id",
                "state_revision",
                "worker_id",
            }:
                raise DataServiceError("ACK request fields are invalid")
            if request["session_id"] != self.service.identity.session_id:
                raise DataServiceError("data service session changed")
            self.service.acknowledge(
                request["lease_id"], request["worker_id"], request["state_revision"]
            )
            return {"ok": True}
        raise DataServiceError("service operation is invalid")

    def _handle_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(self.request_timeout_seconds)
            request = _recv_frame(connection)
        except OSError as error:
            if not _client_disconnected(error):
                raise
            _log(f"客户端连接在读取请求时断开，忽略当前连接: {error}")
            return

        response: dict[str, object]
        try:
            response = self._dispatch(request)
        except (
            DataServiceError,
            DataServiceProtocolError,
            TypeError,
            ValueError,
        ) as error:
            _log(f"请求失败: {error}")
            response = {
                "error": str(error),
                "error_type": type(error).__name__,
                "ok": False,
            }

        try:
            connection.sendall(_response(response))
        except OSError as error:
            if not _client_disconnected(error):
                raise
            _log(f"客户端连接在发送响应时断开，忽略当前连接: {error}")

    def serve(
        self,
        stop_event: threading.Event,
        *,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        listener: socket.socket | None = None
        bound_inode: int | None = None
        try:
            self.service.start(stop_event)
            if self.socket_path.exists() or self.socket_path.is_symlink():
                if self.socket_path.is_symlink() or not stat.S_ISSOCK(
                    self.socket_path.stat().st_mode
                ):
                    raise DataServiceError("data service socket path is occupied")
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.1)
                    probe.connect(str(self.socket_path))
                except OSError:
                    self.socket_path.unlink(missing_ok=True)
                else:
                    raise DataServiceError("another data service owns the socket")
                finally:
                    probe.close()
            # Readiness contract: the socket file must not exist
            # until the warmup barrier passes. Consumers
            # (training_stack.sh) treat socket existence as
            # "protocol servable"; creating it before
            # wait_until_ready() lets the trainer connect into an
            # empty backlog while the accept loop still waits for
            # ready shards, and the client's health timeout
            # expires.
            if not self.service.wait_until_ready() or stop_event.is_set():
                return
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
            bound_inode = self.socket_path.stat().st_ino
            os.chmod(self.socket_path, 0o600)
            listener.listen(self.service.limits.ack_channel_capacity)
            listener.settimeout(0.2)
            if ready_callback is not None:
                ready_callback()
            while not stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    self._handle_connection(connection)
        finally:
            if listener is not None:
                listener.close()
            try:
                if (
                    bound_inode is not None
                    and self.socket_path.exists()
                    and self.socket_path.stat().st_ino == bound_inode
                ):
                    self.socket_path.unlink()
            finally:
                self.service.close()


__all__ = [
    "DataServiceError",
    "DataServiceLimits",
    "DataServiceServer",
    "DataServiceStats",
    "DataSupplyService",
]
