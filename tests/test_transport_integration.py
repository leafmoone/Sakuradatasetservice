"""Same-host socket integration with real tiny tar shards and no network."""

import io
import socket
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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
)
from sakuradatasetservice.client import DataServiceUnavailable
from sakuradatasetservice.service import DataServiceError
from sakuradatasetservice.service_protocol import canonical_json_bytes, parse_frame


@pytest.fixture(params=["dispatch", "unix_socket"])
def transport(request):
    if request.param == "unix_socket":
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except PermissionError:
            pytest.skip(
                "BLOCKED: this execution environment denies AF_UNIX sockets (EPERM)"
            )
        else:
            probe.close()
    return request.param


def make_service(tmp_path, *, repeat=False, workers=1):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    for index in range(2):
        path = source / f"{index}.tar"
        if not path.exists():
            with tarfile.open(path, "w") as tar:
                payload = f"sample-{index}".encode()
                member = tarfile.TarInfo(f"{index}.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/socket", revision="v1")
    )
    cache = ShardCache(
        tmp_path / "cache", manifest, provider, CacheQuota(0, 1024 * 1024)
    )
    return DataSupplyService(
        manifest,
        cache,
        tmp_path / "queue.json",
        tmp_path / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, workers),
        DataServiceLimits(1, workers, workers, workers),
        repeat=repeat,
    )


@contextmanager
def serving(tmp_path, transport, *, repeat=False, workers=1):
    service = make_service(tmp_path, repeat=repeat, workers=workers)
    socket_path = tmp_path / "data.sock"
    server = DataServiceServer(service, socket_path, request_timeout_seconds=2.0)
    if transport == "dispatch":
        # Exercise the real JSON protocol/client/service without claiming OS
        # socket coverage when this environment blocks socket creation.
        class FramedClient(DataServiceClient):
            def _request_transport(self, frame):
                try:
                    response = server._dispatch(parse_frame(frame))
                except DataServiceError as error:
                    response = {"ok": False, "error": str(error)}
                return parse_frame(canonical_json_bytes(response))

        try:
            service.start()
            assert service.wait_until_ready()
            yield (
                service,
                FramedClient(
                    socket_path, worker_count=workers, request_timeout_seconds=2.0
                ),
            )
        finally:
            service.close()
        return
    stop = threading.Event()
    ready = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(server.serve, stop, ready_callback=ready.set)
        try:
            assert ready.wait(5), "socket server did not become ready"
            client = DataServiceClient(
                socket_path, worker_count=workers, request_timeout_seconds=2.0
            )
            yield service, client
        finally:
            stop.set()
            future.result(timeout=5)
    assert not socket_path.exists()


def test_once_real_tar_context_and_cleanup(tmp_path, transport):
    with serving(tmp_path, transport) as (service, client):
        samples = []
        for lease in client.iter_leases(0):
            with lease as shard, tarfile.open(shard.local_path) as tar:
                member = tar.next()
                samples.append(tar.extractfile(member).read().decode())
        assert sorted(samples) == ["sample-0", "sample-1"]
        assert client.health()
        assert service.done


def test_repeat_shortage_and_slow_lease(tmp_path, transport):
    with serving(tmp_path, transport, repeat=True, workers=3) as (service, client):
        slow = client.lease(0)
        fast = client.lease(1)
        assert client.lease(2) is None
        assert not client.health()
        for _ in range(3):
            client.acknowledge(fast)
            client.acknowledge(fast)
            fast = client.lease(1)
            assert fast.record.path != slow.record.path
            assert slow.local_path.is_file()
        client.acknowledge(slow)
        available = client.lease(2)
        assert available.record.path == slow.record.path
        assert not service.done


def test_restart_rejects_old_client_and_reissues(tmp_path, transport):
    with serving(tmp_path, transport) as (_, old_client):
        old_lease = old_client.lease(0)
    with serving(tmp_path, transport) as (_, new_client):
        if transport == "dispatch":
            old_client._request_transport = new_client._request_transport
        with pytest.raises(DataServiceUnavailable, match="session changed"):
            old_client.health()
        with pytest.raises(DataServiceUnavailable, match="session changed"):
            old_client.acknowledge(old_lease)
        new_lease = new_client.lease(0)
        assert new_lease.record.path == old_lease.record.path
        assert new_lease.lease_id != old_lease.lease_id
        new_client.acknowledge(new_lease)
