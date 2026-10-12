"""Validation shard selection and local availability checks."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from sakuradatasetservice.manifest import DatasetManifest, ShardRecord
from sakuradatasetservice.modelscope import (
    FetchedShard,
)
from sakuradatasetservice.providers import DatasetProvider

VALIDATION_SELECTION_SEED = 44


class ValidationSelectionError(ValueError):
    pass


class ValidationSelectionExistsError(ValidationSelectionError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    selection_id: str
    dataset_id: str
    seed: int
    shards: tuple[ShardRecord, ...]

    def __post_init__(self) -> None:
        paths = self.shard_paths
        if (
            not self.selection_id
            or not self.dataset_id
            or type(self.seed) is not int
            or not self.shards
            or len(paths) != len(set(paths))
        ):
            raise ValidationSelectionError("validation shard selection is invalid")

    @property
    def shard_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.shards)


@dataclass(frozen=True, slots=True)
class PreparedValidationShards:
    selection: ValidationSelection
    root: Path
    files: tuple[FetchedShard, ...]


def canonical_validation_selection_bytes(selection: ValidationSelection) -> bytes:
    payload = {
        "dataset_id": selection.dataset_id,
        "schema_version": 2,
        "seed": selection.seed,
        "selection_id": selection.selection_id,
        "shards": [
            {"bytes": item.bytes, "path": item.path} for item in selection.shards
        ],
    }
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def parse_validation_selection(payload: bytes) -> ValidationSelection:
    try:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise TypeError
        document = cast(dict[str, object], raw)
        if (
            set(document)
            != {
                "dataset_id",
                "schema_version",
                "seed",
                "selection_id",
                "shards",
            }
            or document["schema_version"] != 2
        ):
            raise ValueError
        raw_shards = document["shards"]
        if not isinstance(raw_shards, list):
            raise TypeError
        shards = tuple(
            ShardRecord(path=cast(str, item["path"]), bytes=cast(int, item["bytes"]))
            for item in cast(list[dict[str, object]], raw_shards)
        )
        dataset_id = document["dataset_id"]
        if not isinstance(dataset_id, str):
            raise TypeError
        return ValidationSelection(
            selection_id=cast(str, document["selection_id"]),
            dataset_id=dataset_id,
            seed=cast(int, document["seed"]),
            shards=shards,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValidationSelectionError("validation selection is invalid") from None


def select_validation_shards(
    manifest: DatasetManifest,
    *,
    count: int,
    seed: int = VALIDATION_SELECTION_SEED,
) -> ValidationSelection:
    if type(count) is not int or count <= 0 or count >= len(manifest.shards):
        raise ValidationSelectionError("validation shard count is invalid")
    selected = tuple(random.Random(seed).sample(list(manifest.shards), count))
    return ValidationSelection(
        selection_id="validation",
        dataset_id=manifest.dataset_id,
        seed=seed,
        shards=selected,
    )


def create_validation_selection(
    manifest: DatasetManifest,
    *,
    shard_paths: tuple[str, ...],
    seed: int = VALIDATION_SELECTION_SEED,
    selection_id: str = "validation",
) -> ValidationSelection:
    try:
        shards = tuple(manifest.shard(path) for path in shard_paths)
    except ValueError as error:
        raise ValidationSelectionError(str(error)) from None
    return ValidationSelection(selection_id, manifest.dataset_id, seed, shards)


def validate_selection_manifest(
    selection: ValidationSelection, manifest: DatasetManifest
) -> None:
    if selection.dataset_id != manifest.dataset_id:
        raise ValidationSelectionError("validation selection differs from dataset")
    for item in selection.shards:
        if manifest.shard(item.path).bytes != item.bytes:
            raise ValidationSelectionError("validation selection differs from dataset")
    if len(selection.shards) >= len(manifest.shards):
        raise ValidationSelectionError("validation selection leaves no training shard")


def load_validation_selection(path: Path) -> ValidationSelection:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        selection = parse_validation_selection(path.read_bytes())
    except OSError:
        raise ValidationSelectionError(
            "validation selection could not be read"
        ) from None
    return replace(selection, selection_id=path.parent.name)


def write_validation_selection(
    selection: ValidationSelection, destination: Path
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValidationSelectionExistsError("validation selection already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_validation_selection_bytes(selection))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_validation_selection(
    manifest: DatasetManifest,
    path: Path,
    *,
    expected_shard_count: int,
) -> ValidationSelection:
    if path.exists() or path.is_symlink():
        selection = load_validation_selection(path)
    else:
        selection = select_validation_shards(
            manifest, count=expected_shard_count, seed=VALIDATION_SELECTION_SEED
        )
        write_validation_selection(selection, path)
    validate_selection_manifest(selection, manifest)
    if len(selection.shards) != expected_shard_count:
        raise ValidationSelectionError("validation shard count differs from config")
    return selection


def _validation_path(root: Path, record: ShardRecord) -> Path:
    return root / record.path


def prepare_validation_shards(
    transport: DatasetProvider,
    manifest: DatasetManifest,
    selection: ValidationSelection,
    root: Path,
) -> PreparedValidationShards:
    validate_selection_manifest(selection, manifest)
    files: list[FetchedShard] = []
    for record in selection.shards:
        destination = _validation_path(root, record)
        if destination.is_file() and destination.stat().st_size == record.bytes:
            files.append(FetchedShard(destination, record.path, record.bytes, True))
            continue
        temporary_root = root / ".downloads"
        fetched = transport.fetch_shard(manifest, record.path, temporary_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(fetched.path, destination)
        files.append(FetchedShard(destination, record.path, record.bytes, False))
    return PreparedValidationShards(selection, root, tuple(files))


def require_published_validation_shards(
    manifest: DatasetManifest,
    selection: ValidationSelection,
    root: Path,
) -> PreparedValidationShards:
    validate_selection_manifest(selection, manifest)
    files: list[FetchedShard] = []
    for record in selection.shards:
        path = _validation_path(root, record)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.bytes
        ):
            raise ValidationSelectionError(
                f"validation shard is unavailable: {record.path}"
            )
        files.append(FetchedShard(path, record.path, record.bytes, True))
    return PreparedValidationShards(selection, root, tuple(files))


__all__ = [
    "VALIDATION_SELECTION_SEED",
    "PreparedValidationShards",
    "ValidationSelection",
    "ValidationSelectionError",
    "ValidationSelectionExistsError",
    "canonical_validation_selection_bytes",
    "create_validation_selection",
    "ensure_validation_selection",
    "load_validation_selection",
    "parse_validation_selection",
    "prepare_validation_shards",
    "require_published_validation_shards",
    "select_validation_shards",
    "validate_selection_manifest",
    "write_validation_selection",
]
