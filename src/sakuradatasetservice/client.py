"""Lightweight trainer-side client for an already-running data service."""

from __future__ import annotations

import socket
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from sakuradatasetservice.service_protocol import (
    MAX_SERVICE_FRAME_BYTES,
    SERVICE_PROTOCOL_VERSION,
    DataServiceProtocolError,
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
    canonical_json_bytes,
    parse_frame,
)

# A single round-trip can transiently fail while the data service is busy
# (startup window, cache I/O bursts from concurrent shard downloads). Retry
# transport-level failures with a bounded backoff; protocol-level rejections
# (invalid frame, session mismatch) still fail immediately.
_TRANSPORT_ATTEMPTS = 3
_TRANSPORT_RETRY_BASE_SECONDS = 2.0


class DataServiceUnavailable(RuntimeError):
    """The configured service is absent or rejects the exact session identity."""


def _recv(connection: socket.socket) -> dict[str, Any]:
    body = bytearray()
    while len(body) <= MAX_SERVICE_FRAME_BYTES:
        block = connection.recv(min(4096, MAX_SERVICE_FRAME_BYTES + 1 - len(body)))
        if not block:
            break
        body.extend(block)
        if body.endswith(b"\n"):
            break
    try:
        return parse_frame(bytes(body))
    except DataServiceProtocolError:
        raise DataServiceUnavailable("data service returned an invalid frame") from None


class DataServiceClient:
    """Transmit bounded lease/ACK messages without owning cache or state."""

    def __init__(
        self,
        socket_path: Path,
        *,
        worker_count: int,
        request_timeout_seconds: float,
    ) -> None:
        if (
            not socket_path.is_absolute()
            or type(worker_count) is not int
            or worker_count <= 0
            or type(request_timeout_seconds) is not float
            or request_timeout_seconds <= 0.0
        ):
            raise ValueError("data service client settings are invalid")
        self.socket_path = socket_path
        self.worker_count = worker_count
        self.request_timeout_seconds = request_timeout_seconds
        self.identity, _ = self._health_identity()

    def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        frame = canonical_json_bytes(payload)
        if len(frame) > MAX_SERVICE_FRAME_BYTES:
            raise DataServiceUnavailable("data service request exceeds the frame bound")
        response = self._request_transport(frame)
        if response.get("ok") is not True:
            reason = response.get("error")
            detail = reason if isinstance(reason, str) and reason else "unknown error"
            raise DataServiceUnavailable(f"data service rejected the request: {detail}")
        return response

    def _request_transport(self, frame: bytes) -> dict[str, Any]:
        last_error: OSError | None = None
        for attempt in range(_TRANSPORT_ATTEMPTS):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.request_timeout_seconds)
            try:
                connection.connect(str(self.socket_path))
                connection.sendall(frame)
                response = _recv(connection)
            except (OSError, TimeoutError) as error:
                last_error = error
                if attempt + 1 < _TRANSPORT_ATTEMPTS:
                    time.sleep(_TRANSPORT_RETRY_BASE_SECONDS * (2**attempt))
                continue
            finally:
                connection.close()
            return response
        raise DataServiceUnavailable("data service is unavailable") from last_error

    def _health_identity(self) -> tuple[DataServiceSessionIdentity, bool]:
        response = self._request(
            {
                "op": "health",
                "protocol_version": SERVICE_PROTOCOL_VERSION,
                "worker_count": self.worker_count,
            }
        )
        if set(response) != {
            "done",
            "ok",
            "protocol_version",
            "session_identity",
        } or (
            response["protocol_version"] != SERVICE_PROTOCOL_VERSION
            or type(response["done"]) is not bool
        ):
            raise DataServiceUnavailable("data service health identity is invalid")
        try:
            identity = DataServiceSessionIdentity.from_dict(
                response["session_identity"]
            )
        except DataServiceProtocolError:
            raise DataServiceUnavailable(
                "data service health identity is invalid"
            ) from None
        if identity.worker_count != self.worker_count:
            raise DataServiceUnavailable("data service health identity is invalid")
        return identity, response["done"]

    def health(self) -> bool:
        identity, done = self._health_identity()
        if identity != self.identity:
            raise DataServiceUnavailable("data service session changed")
        return done

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        response = self._request(
            {
                "op": "lease",
                "session_id": self.identity.session_id,
                "worker_id": worker_id,
            }
        )
        if (
            set(response) != {"done", "lease", "ok"}
            or type(response["done"]) is not bool
        ):
            raise DataServiceUnavailable("data service lease response is invalid")
        if response["done"]:
            if response["lease"] is not None:
                raise DataServiceUnavailable("completed service returned a lease")
            return None
        try:
            descriptor = ShardLeaseDescriptor.from_dict(response["lease"])
        except DataServiceProtocolError:
            raise DataServiceUnavailable("data service lease is invalid") from None
        if descriptor.worker_id != worker_id or not descriptor.local_path.is_file():
            raise DataServiceUnavailable("data service lease identity is invalid")
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        response = self._request(
            {
                "lease_id": descriptor.lease_id,
                "op": "ack",
                "session_id": self.identity.session_id,
                "state_revision": descriptor.state_revision,
                "worker_id": descriptor.worker_id,
            }
        )
        if set(response) != {"ok"}:
            raise DataServiceUnavailable("data service ACK response is invalid")


class RankedDataServiceClient:
    """Expose one rank's local worker IDs over a shared global service session."""

    def __init__(
        self,
        delegate: DataServiceClient,
        *,
        rank: int,
        workers_per_rank: int,
        world_size: int,
    ) -> None:
        if (
            type(rank) is not int
            or type(workers_per_rank) is not int
            or type(world_size) is not int
            or not 0 <= rank < world_size
            or workers_per_rank <= 0
            or delegate.identity.worker_count != workers_per_rank * world_size
        ):
            raise ValueError("ranked data service topology is invalid")
        self._delegate = delegate
        self._offset = rank * workers_per_rank
        self._workers_per_rank = workers_per_rank
        self._leases: dict[str, ShardLeaseDescriptor] = {}
        self.identity = DataServiceSessionIdentity(
            dataset_id=delegate.identity.dataset_id,
            worker_count=workers_per_rank,
            session_id=delegate.identity.session_id,
        )

    def health(self) -> bool:
        return self._delegate.health()

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if type(worker_id) is not int or not 0 <= worker_id < self._workers_per_rank:
            raise DataServiceUnavailable("rank-local worker ID is invalid")
        descriptor = self._delegate.lease(self._offset + worker_id)
        if descriptor is None:
            return None
        self._leases[descriptor.lease_id] = descriptor
        return replace(descriptor, worker_id=worker_id)

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        original = self._leases.pop(descriptor.lease_id, None)
        if (
            original is None
            or replace(original, worker_id=descriptor.worker_id) != descriptor
        ):
            raise DataServiceUnavailable("rank-local lease identity changed")
        self._delegate.acknowledge(original)


__all__ = [
    "DataServiceClient",
    "DataServiceUnavailable",
    "RankedDataServiceClient",
]
