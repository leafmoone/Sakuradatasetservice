"""Small local inventory for the training WebDataset shards."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from sakuradatasetservice.config import DataSourceConfig

RepoIdPattern = r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$"
RevisionPattern = r"^[A-Za-z0-9._-]+$"

NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=512)]
PositiveInt = Annotated[int, Field(gt=0)]


def _tuple_from_json(value: object) -> object:
    return tuple(cast(list[object], value)) if type(value) is list else value


class DatasetManifestError(ValueError):
    pass


class DatasetManifestPublicationError(DatasetManifestError):
    pass


class DatasetManifestExistsError(DatasetManifestPublicationError):
    pass


class RemoteManifestBuildError(DatasetManifestError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def is_safe_shard_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and value == value.strip()
        and "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and value.casefold().endswith((".tar", ".tar.gz", ".tgz"))
    )


class DatasetSourceIdentity(StrictModel):
    """Corpus source identity — fully config-driven ([data.source]).

    ``repo_id`` is a ModelScope ``owner/name`` repository identifier and
    ``revision`` a branch or tag.  Values are validated structurally
    (shape only) and never matched against a hardcoded list, so pointing a
    new corpus repository in the config requires no code change."""

    repo_id: Annotated[
        str,
        StringConstraints(min_length=3, max_length=200, pattern=RepoIdPattern),
    ]
    revision: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100, pattern=RevisionPattern),
    ]

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id_shape(cls, value: str) -> str:
        owner, name = value.split("/", 1)
        if owner in {".", ".."} or name in {".", ".."}:
            raise ValueError("repo_id parts must not be . or ..")
        return value


class ShardRecord(StrictModel):
    """The path and expected byte size needed to fetch one shard."""

    path: NonEmpty
    bytes: PositiveInt

    @model_validator(mode="after")
    def validate_path(self) -> ShardRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class RemoteShardRecord(StrictModel):
    path: NonEmpty
    bytes: PositiveInt

    @model_validator(mode="after")
    def validate_path(self) -> RemoteShardRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class ManifestAggregates(StrictModel):
    shards: PositiveInt
    bytes: PositiveInt


class DatasetManifest(StrictModel):
    schema_version: Literal[3]
    dataset_id: NonEmpty
    source: DatasetSourceIdentity
    shards: Annotated[tuple[ShardRecord, ...], BeforeValidator(_tuple_from_json)]
    aggregates: ManifestAggregates

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        paths = tuple(item.path for item in self.shards)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("dataset shard paths must be sorted and unique")
        expected = ManifestAggregates(
            shards=len(self.shards), bytes=sum(item.bytes for item in self.shards)
        )
        if self.aggregates != expected:
            raise ValueError("dataset manifest aggregates do not match shard records")
        return self

    @classmethod
    def from_shards(
        cls,
        source: DatasetSourceIdentity,
        shards: tuple[ShardRecord, ...],
    ) -> DatasetManifest:
        ordered = tuple(sorted(shards, key=lambda item: item.path))
        return cls(
            schema_version=3,
            dataset_id=f"{source.repo_id}@{source.revision}",
            source=source,
            shards=ordered,
            aggregates=ManifestAggregates(
                shards=len(ordered), bytes=sum(item.bytes for item in ordered)
            ),
        )

    def shard(self, path: str) -> ShardRecord:
        for shard in self.shards:
            if shard.path == path:
                return shard
        raise DatasetManifestError(f"unknown dataset shard: {path}")


def canonical_manifest_bytes(manifest: DatasetManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def parse_dataset_manifest_bytes(payload: bytes) -> DatasetManifest:
    try:
        parsed: object = json.loads(payload)
        return DatasetManifest.model_validate(parsed, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        raise DatasetManifestError("dataset manifest is invalid") from None


def source_identity(source: DataSourceConfig) -> DatasetSourceIdentity:
    try:
        return DatasetSourceIdentity(repo_id=source.repo_id, revision=source.revision)
    except ValidationError:
        raise DatasetManifestError("dataset source config is invalid") from None


def build_dataset_manifest(
    source: DatasetSourceIdentity,
    remote_shards: tuple[RemoteShardRecord, ...],
) -> DatasetManifest:
    if not remote_shards or len({item.path for item in remote_shards}) != len(
        remote_shards
    ):
        raise RemoteManifestBuildError(
            "remote inventory is empty or has duplicate paths"
        )
    return DatasetManifest.from_shards(
        source,
        tuple(ShardRecord(path=item.path, bytes=item.bytes) for item in remote_shards),
    )


def load_dataset_manifest(path: Path, source: DataSourceConfig) -> DatasetManifest:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        manifest = parse_dataset_manifest_bytes(path.read_bytes())
    except OSError:
        raise DatasetManifestError("dataset manifest could not be read") from None
    if manifest.source != source_identity(source):
        raise DatasetManifestError("dataset manifest source does not match config")
    return manifest


def write_dataset_manifest(manifest: DatasetManifest, destination: Path) -> str:
    payload = canonical_manifest_bytes(manifest)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise DatasetManifestExistsError(
                "dataset manifest destination already exists"
            ) from None
        temporary.unlink()
        temporary = None
    except DatasetManifestExistsError:
        raise
    except OSError:
        raise DatasetManifestPublicationError(
            "dataset manifest could not be written"
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest.dataset_id


__all__ = [
    "DatasetManifest",
    "DatasetManifestError",
    "DatasetManifestExistsError",
    "DatasetManifestPublicationError",
    "DatasetSourceIdentity",
    "ManifestAggregates",
    "RemoteManifestBuildError",
    "RemoteShardRecord",
    "ShardRecord",
    "build_dataset_manifest",
    "canonical_manifest_bytes",
    "is_safe_shard_path",
    "load_dataset_manifest",
    "parse_dataset_manifest_bytes",
    "source_identity",
    "write_dataset_manifest",
]
