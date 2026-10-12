"""Small HTTPS client for the configured ModelScope WebDataset branch."""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import ssl
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit

from sakuradatasetservice.config import (
    ConfigurationError,
    DataSourceConfig,
    DataTransportConfig,
    resolve_secret,
)
from sakuradatasetservice.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetManifestExistsError,
    DatasetSourceIdentity,
    RemoteShardRecord,
    ShardRecord,
    build_dataset_manifest,
    is_safe_shard_path,
    load_dataset_manifest,
    source_identity,
    write_dataset_manifest,
)

MODELSCOPE_DATASET_HOST = "modelscope.cn"
MODELSCOPE_TOKEN_ENVIRONMENT = "MODELSCOPE_API_TOKEN"
_HTTPS_PORT = 443
_MAX_REDIRECTS = 5
_LISTING_PAGE_SIZE = 1000
_LISTING_MAX_PAGES = 100_000
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class DatasetTransportError(RuntimeError):
    """ModelScope listing or download failed."""


class DatasetTransientError(DatasetTransportError):
    """A temporary ModelScope failure that can resume from partial bytes."""


class DatasetAuthenticationError(DatasetTransportError):
    """ModelScope authentication failed."""


class ShardIntegrityError(DatasetTransportError):
    """A shard does not match the dataset manifest."""


class _RetryableRequestError(Exception):
    pass


class _RangeUnsupportedError(DatasetTransportError):
    pass


@dataclass(frozen=True)
class FetchedShard:
    path: Path
    relative_path: str
    bytes: int
    cache_hit: bool


@dataclass(frozen=True)
class _Target:
    host: str
    request_target: str
    authenticated: bool


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


DownloadProgress = Callable[[int, int, float, float], None]

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _repo_path(source: DatasetSourceIdentity) -> str:
    owner, name = source.repo_id.split("/", 1)
    return f"{quote(owner, safe='')}/{quote(name, safe='')}"


class ModelScopeDatasetTransport:
    """Download one shard at a time without pre-downloading the dataset."""

    def __init__(self, token: str, policy: DataTransportConfig) -> None:
        if not token or "\r" in token or "\n" in token:
            raise DatasetAuthenticationError("ModelScope token is missing or invalid")
        self._token = token
        self._policy = policy

    def fetch_manifest(self, source: DatasetSourceIdentity) -> DatasetManifest:
        return build_remote_dataset_manifest(self, source)

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

    @classmethod
    def from_token_environment(
        cls, token_environment_name: str, policy: DataTransportConfig
    ) -> ModelScopeDatasetTransport:
        if token_environment_name != MODELSCOPE_TOKEN_ENVIRONMENT:
            raise DatasetAuthenticationError("unexpected ModelScope token variable")
        try:
            token = resolve_secret(token_environment_name).get_secret_value()
        except ConfigurationError:
            raise DatasetAuthenticationError("ModelScope token is missing") from None
        return cls(token, policy)

    def _headers(
        self, target: _Target, extra_headers: dict[str, str] | None = None
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json, application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "Sakuradatasetservice/1",
        }
        if target.authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
            headers["Cookie"] = f"m_session_id={self._token}"
        if extra_headers is not None:
            headers.update(extra_headers)
        return headers

    @property
    def stream_chunk_bytes(self) -> int:
        return self._policy.stream_chunk_bytes

    @property
    def streams_per_shard(self) -> int:
        return self._policy.streams_per_shard

    def _redirect(self, current: _Target, location: str) -> _Target:
        parsed = urlsplit(
            urljoin(f"https://{current.host}{current.request_target}", location)
        )
        host = parsed.hostname.casefold() if parsed.hostname else ""
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, _HTTPS_PORT)
        ):
            raise DatasetTransportError("ModelScope returned an unsafe redirect")
        request_target = parsed.path or "/"
        if parsed.query:
            request_target += f"?{parsed.query}"
        return _Target(
            host=host,
            request_target=request_target,
            authenticated=current.authenticated and host == MODELSCOPE_DATASET_HOST,
        )

    def _open(
        self,
        initial: _Target,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[http.client.HTTPResponse, Any]:
        target = initial
        for redirects in range(_MAX_REDIRECTS + 1):
            connection: http.client.HTTPSConnection | None = None
            try:
                proxy_raw = os.environ.get("https_proxy") or os.environ.get(
                    "HTTPS_PROXY"
                )
                proxy = urlsplit(proxy_raw) if proxy_raw else None
                if (
                    proxy is not None
                    and proxy.scheme in {"http", "https"}
                    and proxy.hostname
                ):
                    proxy_headers: dict[str, str] = {}
                    if proxy.username is not None:
                        credentials = (
                            f"{unquote(proxy.username)}:{unquote(proxy.password or '')}"
                        )
                        encoded = base64.b64encode(credentials.encode("utf-8")).decode(
                            "ascii"
                        )
                        proxy_headers["Proxy-Authorization"] = f"Basic {encoded}"
                    connection = http.client.HTTPSConnection(
                        proxy.hostname,
                        proxy.port or (443 if proxy.scheme == "https" else 80),
                        timeout=self._policy.connect_timeout_seconds,
                        context=ssl.create_default_context(),
                    )
                    connection.set_tunnel(
                        target.host, _HTTPS_PORT, headers=proxy_headers
                    )
                else:
                    connection = http.client.HTTPSConnection(
                        target.host,
                        _HTTPS_PORT,
                        timeout=self._policy.connect_timeout_seconds,
                        context=ssl.create_default_context(),
                    )
                connection.request(
                    "GET",
                    target.request_target,
                    headers=self._headers(target, extra_headers),
                )
                if connection.sock is not None:
                    connection.sock.settimeout(self._policy.read_timeout_seconds)
                response = connection.getresponse()
            except (OSError, TimeoutError, http.client.HTTPException):
                if connection is not None:
                    connection.close()
                raise _RetryableRequestError from None
            if response.status not in _REDIRECT_STATUSES:
                return response, connection
            location = response.getheader("Location")
            response.close()
            connection.close()
            if location is None or redirects == _MAX_REDIRECTS:
                raise DatasetTransportError("ModelScope redirect limit exceeded")
            target = self._redirect(target, location)
        raise DatasetTransportError("ModelScope redirect limit exceeded")

    def _with_retries(
        self,
        target: _Target,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[http.client.HTTPResponse, Any]:
        for attempt in range(self._policy.max_retries + 1):
            try:
                response, connection = self._open(target, extra_headers)
            except _RetryableRequestError:
                if attempt == self._policy.max_retries:
                    raise DatasetTransientError("ModelScope request failed") from None
            else:
                if response.status in (401, 403):
                    response.close()
                    connection.close()
                    raise DatasetAuthenticationError("ModelScope authentication failed")
                if response.status not in _RETRYABLE_STATUSES:
                    return response, connection
                response.close()
                connection.close()
                if attempt == self._policy.max_retries:
                    raise DatasetTransientError("ModelScope retry limit exceeded")
            time.sleep(self._policy.retry_backoff_seconds)
        raise DatasetTransientError("ModelScope retry limit exceeded")

    def list_files(
        self, source: DatasetSourceIdentity
    ) -> tuple[RemoteShardRecord, ...]:
        """List all WebDataset shards for the configured branch selector."""

        files: list[RemoteShardRecord] = []
        for page in range(1, _LISTING_MAX_PAGES + 1):
            query = urlencode(
                {
                    "Revision": source.revision,
                    "Recursive": "True",
                    "PageNumber": page,
                    "PageSize": _LISTING_PAGE_SIZE,
                }
            )
            target = _Target(
                MODELSCOPE_DATASET_HOST,
                f"/api/v1/datasets/{_repo_path(source)}/repo/tree?{query}",
                True,
            )
            response, connection = self._with_retries(target)
            try:
                if response.status != 200:
                    raise DatasetTransportError("ModelScope listing failed")
                payload = response.read()
                document = cast(dict[str, Any], json.loads(payload))
                data = cast(dict[str, Any], document["Data"])
                entries = cast(object, data["Files"])
                if not isinstance(entries, list):
                    raise TypeError
                if not entries:
                    return tuple(files)
                for raw_entry in cast(list[object], entries):
                    if not isinstance(raw_entry, dict):
                        raise TypeError
                    entry = cast(dict[str, object], raw_entry)
                    if entry.get("Type") in {"tree", "directory"}:
                        continue
                    path = entry["Path"]
                    size = entry["Size"]
                    if not isinstance(path, str) or type(size) is not int:
                        raise TypeError
                    if not path.casefold().endswith((".tar", ".tar.gz", ".tgz")):
                        continue
                    if not is_safe_shard_path(path):
                        raise TypeError
                    files.append(
                        RemoteShardRecord(
                            path=path,
                            bytes=size,
                        )
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise DatasetTransportError("ModelScope listing is invalid") from None
            finally:
                response.close()
                connection.close()
        raise DatasetTransportError("ModelScope listing exceeded configured page limit")

    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: _Writer,
        *,
        start_offset: int = 0,
        end_offset: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Stream one full file or byte range, resuming failed transfers."""

        end = shard.bytes - 1 if end_offset is None else end_offset
        if (
            type(start_offset) is not int
            or type(end) is not int
            or start_offset < 0
            or end < start_offset
            or end >= shard.bytes
        ):
            raise DatasetTransportError("dataset shard byte range is invalid")

        query = urlencode(
            {"Revision": manifest.source.revision, "FilePath": shard.path}
        )
        target = _Target(
            MODELSCOPE_DATASET_HOST,
            f"/api/v1/datasets/{_repo_path(manifest.source)}/repo?{query}",
            True,
        )
        current = start_offset
        transfer_failures = 0
        while current <= end:
            if cancelled is not None and cancelled():
                raise DatasetTransportError("dataset shard download was cancelled")
            ranged = current > 0 or end < shard.bytes - 1
            headers = {"Range": f"bytes={current}-{end}"} if ranged else None
            response, connection = self._with_retries(target, headers)
            transfer_failed = False
            try:
                if response.status == 404:
                    raise _RetryableRequestError
                if ranged and response.status == 200:
                    raise _RangeUnsupportedError(
                        "ModelScope download endpoint ignored the byte range"
                    )
                if response.status not in ({206} if ranged else {200, 206}):
                    raise DatasetTransportError("dataset shard download failed")
                if response.status == 206:
                    match = _CONTENT_RANGE.fullmatch(
                        response.getheader("Content-Range") or ""
                    )
                    if (
                        match is None
                        or int(match.group(1)) != current
                        or int(match.group(2)) != end
                        or int(match.group(3)) != shard.bytes
                    ):
                        raise DatasetTransportError(
                            "ModelScope returned an invalid byte range"
                        )
                if response.getheader("Content-Encoding") not in (None, "identity"):
                    raise DatasetTransportError(
                        "compressed HTTP transfer is not supported"
                    )
                while current <= end:
                    if cancelled is not None and cancelled():
                        raise DatasetTransportError(
                            "dataset shard download was cancelled"
                        )
                    chunk = response.read(
                        min(self._policy.stream_chunk_bytes, end - current + 1)
                    )
                    if not chunk:
                        break
                    if output.write(chunk) != len(chunk):
                        raise DatasetTransportError(
                            "dataset shard write was incomplete"
                        )
                    current += len(chunk)
            except (
                OSError,
                TimeoutError,
                http.client.HTTPException,
                _RetryableRequestError,
            ):
                transfer_failed = True
            finally:
                response.close()
                connection.close()
            if current > end:
                return
            if cancelled is not None and cancelled():
                raise DatasetTransportError("dataset shard download was cancelled")
            if transfer_failures >= self._policy.max_retries:
                reason = "failed" if transfer_failed else "ended early"
                raise DatasetTransientError(
                    f"dataset shard transfer {reason}"
                ) from None
            transfer_failures += 1
            time.sleep(self._policy.retry_backoff_seconds)


def validate_remote_manifest(
    transport: ModelScopeDatasetTransport, manifest: DatasetManifest
) -> None:
    """Require the configured shard paths and sizes to match the remote listing."""

    expected = {(item.path, item.bytes) for item in manifest.shards}
    remote_files = transport.list_files(manifest.source)
    observed = {(item.path, item.bytes) for item in remote_files}
    if len(remote_files) != len(manifest.shards) or observed != expected:
        raise ShardIntegrityError("remote dataset listing differs from manifest")


def build_remote_dataset_manifest(
    transport: ModelScopeDatasetTransport,
    source: DatasetSourceIdentity,
) -> DatasetManifest:
    """Build v2 only from facts returned by the upstream listing."""

    return build_dataset_manifest(source, transport.list_files(source))


def ensure_dataset_manifest(
    transport: ModelScopeDatasetTransport,
    path: Path,
    source: DataSourceConfig,
    *,
    initialize_if_missing: bool,
    refresh_existing: bool,
) -> DatasetManifest:
    """Initialize once when absent; otherwise load the operational snapshot locally."""

    if type(initialize_if_missing) is not bool or type(refresh_existing) is not bool:
        raise DatasetManifestError("dataset manifest policy is invalid")
    if refresh_existing:
        raise DatasetManifestError("automatic dataset manifest refresh is prohibited")
    if path.exists() or path.is_symlink():
        return load_dataset_manifest(path, source)
    if not initialize_if_missing:
        raise DatasetManifestError(
            "dataset manifest is absent and initialization is disabled"
        )

    manifest = build_remote_dataset_manifest(transport, source_identity(source))
    try:
        write_dataset_manifest(manifest, path)
    except DatasetManifestExistsError:
        # A concurrent service won no-clobber publication. Its strict local manifest
        # becomes the operational snapshot; mutable master is not relisted on restart.
        manifest = load_dataset_manifest(path, source)
    return manifest


def _verify_existing(path: Path, shard: ShardRecord) -> bool:
    return path.is_file() and path.stat().st_size == shard.bytes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _TransferProgress:
    def __init__(
        self,
        downloaded: int,
        total: int,
        callback: DownloadProgress | None,
    ) -> None:
        self.downloaded = downloaded
        self.total = total
        self.callback = callback
        self.started = time.monotonic()
        self.last_report = self.started
        self.session_bytes = 0
        self.lock = threading.Lock()
        if downloaded and callback is not None:
            callback(downloaded, total, 0.0, 0.0)

    def add(self, size: int) -> None:
        with self.lock:
            self.downloaded += size
            self.session_bytes += size
            if self.downloaded > self.total:
                raise ShardIntegrityError("downloaded shard is larger than manifest")
            now = time.monotonic()
            if self.callback is not None and (
                now - self.last_report >= 5.0 or self.downloaded == self.total
            ):
                elapsed = max(now - self.started, 1e-6)
                self.callback(
                    self.downloaded,
                    self.total,
                    elapsed,
                    self.session_bytes / elapsed,
                )
                self.last_report = now


def _download_ranges(start: int, end: int, count: int) -> tuple[tuple[int, int], ...]:
    remaining = end - start + 1
    workers = min(count, remaining)
    part_size = (remaining + workers - 1) // workers
    return tuple(
        (offset, min(offset + part_size - 1, end))
        for offset in range(start, end + 1, part_size)
    )


def _range_path(partial: Path, start: int, end: int) -> Path:
    return partial.with_name(f"{partial.name}.range-{start:012d}-{end:012d}")


def _require_regular_partial(path: Path, *, maximum_bytes: int) -> int:
    if path.is_symlink():
        raise ShardIntegrityError("cache partial must not be a symlink")
    if not path.exists():
        return 0
    if not path.is_file():
        raise ShardIntegrityError("cache partial must be a regular file")
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ShardIntegrityError("cache partial is larger than its byte range")
    return size


class ShardTransport(Protocol):
    @property
    def stream_chunk_bytes(self) -> int: ...
    @property
    def streams_per_shard(self) -> int: ...
    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: _Writer,
        *,
        start_offset: int = 0,
        end_offset: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None: ...


def fetch_dataset_shard(
    transport: ShardTransport,
    manifest: DatasetManifest,
    shard_path: str,
    cache_root: Path,
    *,
    progress: DownloadProgress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FetchedShard:
    """Resume and parallelize one shard before atomically publishing it."""

    shard = manifest.shard(shard_path)
    destination = cache_root / shard.path
    partial = destination.with_name(f"{destination.name}.partial")
    assembly = destination.with_name(f"{destination.name}.partial.assembling")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify_existing(destination, shard):
        return FetchedShard(
            destination,
            shard.path,
            shard.bytes,
            True,
        )
    if destination.exists():
        raise ShardIntegrityError("cached shard differs from manifest")
    prefix_bytes = _require_regular_partial(partial, maximum_bytes=shard.bytes)
    if prefix_bytes < shard.bytes:
        partial.touch(exist_ok=True)
        ranges = _download_ranges(
            prefix_bytes,
            shard.bytes - 1,
            transport.streams_per_shard,
        )
        parts = tuple(
            (start, end, _range_path(partial, start, end)) for start, end in ranges
        )
        existing_part_bytes = 0
        for start, end, path in parts:
            existing_part_bytes += _require_regular_partial(
                path,
                maximum_bytes=end - start + 1,
            )
            path.touch(exist_ok=True)
        tracker = _TransferProgress(
            prefix_bytes + existing_part_bytes,
            shard.bytes,
            progress,
        )

        def download_part(start: int, end: int, path: Path) -> None:
            existing = path.stat().st_size
            if existing == end - start + 1:
                return
            with path.open("ab") as handle:

                class _PartWriter:
                    def write(self, payload: bytes) -> int:
                        written = handle.write(payload)
                        tracker.add(written)
                        return written

                transport.download(
                    manifest,
                    shard,
                    _PartWriter(),
                    start_offset=start + existing,
                    end_offset=end,
                    cancelled=cancelled,
                )
                handle.flush()
                os.fsync(handle.fileno())

        with ThreadPoolExecutor(
            max_workers=len(parts),
            thread_name_prefix="sakuradatasetservice-shard-range",
        ) as executor:
            futures = [
                executor.submit(download_part, start, end, path)
                for start, end, path in parts
            ]
            for future in futures:
                future.result()

        if cancelled is not None and cancelled():
            raise DatasetTransportError("dataset shard download was cancelled")
        if any(path.stat().st_size != end - start + 1 for start, end, path in parts):
            raise DatasetTransientError("dataset shard range download ended early")
        if assembly.is_symlink() or (assembly.exists() and not assembly.is_file()):
            raise ShardIntegrityError("cache assembly path is invalid")
        try:
            with assembly.open("wb") as output:
                for source in (partial, *(path for _, _, path in parts)):
                    if cancelled is not None and cancelled():
                        raise DatasetTransportError(
                            "dataset shard download was cancelled"
                        )
                    if not source.exists():
                        continue
                    with source.open("rb") as handle:
                        while chunk := handle.read(16 * transport.stream_chunk_bytes):
                            output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if assembly.stat().st_size != shard.bytes:
                raise ShardIntegrityError("assembled shard size differs from manifest")
            os.replace(assembly, partial)
            _fsync_directory(partial.parent)
        except Exception:
            assembly.unlink(missing_ok=True)
            raise
        for _, _, path in parts:
            path.unlink(missing_ok=True)
        _fsync_directory(partial.parent)

    if partial.stat().st_size != shard.bytes:
        raise ShardIntegrityError("downloaded shard size differs from manifest")
    os.replace(partial, destination)
    _fsync_directory(destination.parent)
    return FetchedShard(
        destination,
        shard.path,
        shard.bytes,
        False,
    )
