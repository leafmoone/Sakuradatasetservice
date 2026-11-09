"""Multiple distinct shards per folder, with durable scheduling identity."""

from collections import Counter
from pathlib import PurePosixPath
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from sakuradatasetservice import (
    CacheQuota,
    DataServiceClient,
    DataServiceLimits,
    DataServiceServer,
    DataServiceSessionIdentity,
    DatasetSourceIdentity,
    DataSupplyService,
    LocalDatasetProvider,
    ShardCache,
    cli,
)
from sakuradatasetservice.manifest import (
    DatasetManifest,
    ShardRecord,
    write_dataset_manifest,
)
from sakuradatasetservice.service import DataServiceError, _QueueStore


def test_five_per_folder_without_replacement_and_resume(tmp_path):
    paths = [f"many/{i}.tar" for i in range(12)] + [f"few/{i}.tar" for i in range(3)]
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="local/test", revision="v1"),
        tuple(ShardRecord(path=p, bytes=1) for p in paths),
    )

    def store(count):
        return _QueueStore(
            tmp_path / "queue.json",
            manifest,
            frozenset(),
            sampling="balance",
            seed=42,
            per_folder=count,
        )

    queue = store(5)
    first = queue.new(0)
    selected = [row.path for row in first.rows]
    assert Counter(PurePosixPath(p).parent.as_posix() for p in selected) == {
        "many": 5,
        "few": 3,
    }
    assert len(selected) == len(set(selected))
    assert first == store(5).new(0)
    assert queue.new(1) != first
    queue.save(first)
    assert store(5).load() == first
    with pytest.raises(DataServiceError):
        store(4).load()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_rejects_invalid_per_folder(tmp_path, value):
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="local/test", revision="v1"),
        (ShardRecord(path="a/one.tar", bytes=1),),
    )
    with pytest.raises(ValueError, match="per_folder"):
        _QueueStore(
            tmp_path / "q", manifest, frozenset(), sampling="balance", per_folder=value
        )


def test_per_folder_requires_balance(tmp_path):
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="local/test", revision="v1"),
        (ShardRecord(path="a/one.tar", bytes=1),),
    )
    with pytest.raises(ValueError, match="balance"):
        _QueueStore(tmp_path / "q", manifest, frozenset(), per_folder=5)


def test_cli_passes_five_per_folder(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.tar").write_bytes(b"a")
    manifest = LocalDatasetProvider(source).fetch_manifest(
        DatasetSourceIdentity(repo_id="local/test", revision="v1")
    )
    manifest_path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    constructor = Mock()
    monkeypatch.setattr(cli, "DataSupplyService", constructor)
    monkeypatch.setattr(cli, "DataServiceServer", Mock())
    monkeypatch.setattr(cli.signal, "signal", Mock())
    cli.main(
        [
            "serve",
            "--local-root",
            str(source),
            "--manifest",
            str(manifest_path),
            "--cache-root",
            str(tmp_path / "cache"),
            "--state-dir",
            str(tmp_path / "state"),
            "--cache-high-bytes",
            "1024",
            "--sampling",
            "balance",
            "--per-folder",
            "5",
        ]
    )
    assert constructor.call_args.kwargs["per_folder"] == 5


def test_service_delivers_five_per_folder_over_socket(tmp_path):
    source = tmp_path / "source"
    for folder, count in (("many", 8), ("few", 2)):
        (source / folder).mkdir(parents=True)
        for i in range(count):
            (source / folder / f"{i}.tar").write_bytes(f"{folder}/{i}".encode())
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/test", revision="v1")
    )
    cache = ShardCache(
        tmp_path / "cache", manifest, provider, CacheQuota(0, 1024 * 1024)
    )
    service = DataSupplyService(
        manifest,
        cache,
        tmp_path / "queue.json",
        tmp_path / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, 1),
        DataServiceLimits(1, 1, 1, 1),
        sampling="balance",
        seed=42,
        per_folder=5,
    )
    server = DataServiceServer(
        service, tmp_path / "data.sock", request_timeout_seconds=3.0
    )
    stop, ready = Event(), Event()
    thread = Thread(
        target=server.serve, args=(stop,), kwargs={"ready_callback": ready.set}
    )
    thread.start()
    try:
        assert ready.wait(5)
        client = DataServiceClient(
            tmp_path / "data.sock", worker_count=1, request_timeout_seconds=3.0
        )
        received = []
        for lease in client.iter_leases(worker_id=0):
            with lease as shard:
                received.append(shard.local_path.read_bytes().decode())
        assert len(set(received)) == 7
        assert Counter(item.split("/")[0] for item in received) == {"many": 5, "few": 2}
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
