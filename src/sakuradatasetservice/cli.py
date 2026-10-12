"""Minimal inventory and same-host service commands."""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from .cache import CacheQuota, ShardCache
from .config import DataTransportConfig
from .manifest import (
    DatasetSourceIdentity,
    parse_dataset_manifest_bytes,
    write_dataset_manifest,
)
from .modelscope import ModelScopeDatasetTransport
from .providers import DatasetProvider, LocalDatasetProvider
from .service import DataServiceLimits, DataServiceServer, DataSupplyService
from .service_protocol import DataServiceSessionIdentity


def _provider(args: argparse.Namespace) -> DatasetProvider:
    if args.local_root is not None:
        return LocalDatasetProvider(args.local_root)
    return ModelScopeDatasetTransport.from_token_environment(
        "MODELSCOPE_API_TOKEN", DataTransportConfig()
    )


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--local-root", type=Path, help="trusted directory containing .tar shards"
    )
    source.add_argument(
        "--modelscope",
        action="store_true",
        help="use MODELSCOPE_API_TOKEN from the environment",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Same-host pluggable dataset shard delivery (Linux)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser(
        "manifest", help="create a new inventory; never overwrite an existing file"
    )
    _source_arguments(inventory)
    inventory.add_argument(
        "--repo-id", required=True, help="logical owner/name source namespace"
    )
    inventory.add_argument("--revision", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser(
        "serve", help="serve a saved inventory over a local Unix socket"
    )
    _source_arguments(serve)
    serve.add_argument("--manifest", type=Path, required=True)
    serve.add_argument("--cache-root", type=Path, required=True)
    serve.add_argument("--state-dir", type=Path, required=True)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--download-concurrency", type=int, default=1)
    serve.add_argument("--cache-low-bytes", type=int, default=0)
    serve.add_argument("--cache-high-bytes", type=int, required=True)
    serve.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="manifest shard path to exclude; repeatable",
    )
    serve.add_argument(
        "--repeat",
        action="store_true",
        help="cycle available unleased shards indefinitely",
    )
    args = parser.parse_args(argv)
    provider = _provider(args)
    if args.command == "manifest":
        manifest = provider.fetch_manifest(
            DatasetSourceIdentity(repo_id=args.repo_id, revision=args.revision)
        )
        write_dataset_manifest(manifest, args.output)
        print(f"Created manifest: {args.output} ({len(manifest.shards)} shards)")
        return
    manifest = parse_dataset_manifest_bytes(args.manifest.read_bytes())
    state = args.state_dir.resolve()
    cache = ShardCache(
        args.cache_root,
        manifest,
        provider,
        CacheQuota(args.cache_low_bytes, args.cache_high_bytes),
    )
    service = DataSupplyService(
        manifest,
        cache,
        state / "queue.json",
        state / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, args.workers),
        DataServiceLimits(
            args.download_concurrency, args.workers, args.workers, args.workers
        ),
        repeat=args.repeat,
        excluded_paths=frozenset(args.exclude),
    )
    server = DataServiceServer(
        service, state / "data.sock", request_timeout_seconds=30.0
    )
    stop = threading.Event()
    previous = {
        number: signal.signal(number, lambda *_: stop.set())
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        server.serve(
            stop,
            ready_callback=lambda: print(f"Ready: {server.socket_path}", flush=True),
        )
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
