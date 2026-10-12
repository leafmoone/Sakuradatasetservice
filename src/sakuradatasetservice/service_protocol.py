"""Small JSON protocol shared by the local data service and trainer."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from sakuradatasetservice.manifest import ShardRecord

SERVICE_PROTOCOL_VERSION = 4
MAX_SERVICE_FRAME_BYTES = 16 * 1024 * 1024


class DataServiceProtocolError(RuntimeError):
    pass


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class DataServiceSessionIdentity:
    dataset_id: str
    worker_count: int
    session_id: str = field(default_factory=lambda: secrets.token_urlsafe(12))

    def __post_init__(self) -> None:
        _nonempty(self.dataset_id, "dataset_id")
        _nonempty(self.session_id, "session_id")
        if type(self.worker_count) is not int or self.worker_count <= 0:
            raise ValueError("data service worker count is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "worker_count": self.worker_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> DataServiceSessionIdentity:
        document = _mapping(value, "session identity")
        _exact_keys(
            document, {"dataset_id", "session_id", "worker_count"}, "session identity"
        )
        try:
            return cls(
                dataset_id=cast(str, document["dataset_id"]),
                session_id=cast(str, document["session_id"]),
                worker_count=cast(int, document["worker_count"]),
            )
        except (TypeError, ValueError):
            raise DataServiceProtocolError("session identity is invalid") from None


@dataclass(frozen=True, slots=True)
class ShardLeaseDescriptor:
    lease_id: str
    worker_id: int
    cycle_index: int
    state_revision: int
    record: ShardRecord
    local_path: Path

    def __post_init__(self) -> None:
        if (
            type(self.lease_id) is not str
            or not self.lease_id
            or type(self.worker_id) is not int
            or self.worker_id < 0
            or type(self.cycle_index) is not int
            or self.cycle_index < 0
            or type(self.state_revision) is not int
            or self.state_revision < 0
            or not self.local_path.is_absolute()
        ):
            raise ValueError("shard lease descriptor is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "lease_id": self.lease_id,
            "local_path": str(self.local_path),
            "record": {"bytes": self.record.bytes, "path": self.record.path},
            "state_revision": self.state_revision,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ShardLeaseDescriptor:
        document = _mapping(value, "lease")
        _exact_keys(
            document,
            {
                "cycle_index",
                "lease_id",
                "local_path",
                "record",
                "state_revision",
                "worker_id",
            },
            "lease",
        )
        record = _mapping(document["record"], "lease record")
        _exact_keys(record, {"bytes", "path"}, "lease record")
        try:
            return cls(
                lease_id=cast(str, document["lease_id"]),
                worker_id=cast(int, document["worker_id"]),
                cycle_index=cast(int, document["cycle_index"]),
                state_revision=cast(int, document["state_revision"]),
                local_path=Path(cast(str, document["local_path"])),
                record=ShardRecord(
                    path=cast(str, record["path"]), bytes=cast(int, record["bytes"])
                ),
            )
        except (TypeError, ValueError):
            raise DataServiceProtocolError("lease is invalid") from None


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataServiceProtocolError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _exact_keys(document: dict[str, Any], expected: set[str], name: str) -> None:
    if set(document) != expected:
        raise DataServiceProtocolError(f"{name} has unknown or missing fields")


def parse_frame(payload: bytes) -> dict[str, Any]:
    if (
        not payload
        or len(payload) > MAX_SERVICE_FRAME_BYTES
        or not payload.endswith(b"\n")
    ):
        raise DataServiceProtocolError("service frame size or terminator is invalid")
    try:
        return _mapping(json.loads(payload), "service frame")
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DataServiceProtocolError("service frame is invalid JSON") from None


__all__ = [
    "MAX_SERVICE_FRAME_BYTES",
    "SERVICE_PROTOCOL_VERSION",
    "DataServiceProtocolError",
    "DataServiceSessionIdentity",
    "ShardLeaseDescriptor",
    "canonical_json_bytes",
    "parse_frame",
]
