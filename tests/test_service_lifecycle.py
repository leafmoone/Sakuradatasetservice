from concurrent.futures import Future
from dataclasses import replace

import pytest

from sakuradatasetservice import (
    CacheQuota,
    DataServiceLimits,
    DataServiceSessionIdentity,
    DatasetSourceIdentity,
    DataSupplyService,
    LocalDatasetProvider,
    ShardCache,
    ShardLease,
)
from sakuradatasetservice.service import DataServiceError


def make_service(
    tmp_path, *, repeat=False, count=3, workers=2, excluded_paths=frozenset()
):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    for index in range(count):
        (source / f"{index}.tar").write_bytes(bytes([index + 1]) * 10)
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/service", revision="v1")
    )
    cache = ShardCache(tmp_path / "cache", manifest, provider, CacheQuota(0, 1000))
    return DataSupplyService(
        manifest,
        cache,
        tmp_path / "queue.json",
        tmp_path / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, workers),
        DataServiceLimits(1, workers, workers, workers),
        repeat=repeat,
        excluded_paths=excluded_paths,
    )


def ack(service, descriptor):
    service.acknowledge(
        descriptor.lease_id, descriptor.worker_id, descriptor.state_revision
    )


def test_once_and_optional_exclusions(tmp_path):
    service = make_service(tmp_path, excluded_paths=frozenset({"0.tar"}))
    try:
        service.start()
        seen = []
        while (lease := service.lease(0)) is not None:
            seen.append(lease.record.path)
            ack(service, lease)
            ack(service, lease)
        assert set(seen) == {"1.tar", "2.tar"}
        assert len(seen) == 2
        assert service.done
    finally:
        service.close()


def test_current_ready_intersection(tmp_path):
    service = make_service(tmp_path, count=2, workers=1)
    try:
        service.start()
        first = service.lease(0)
        assert first is not None
        ack(service, first)
        # Simulate a future-cycle cache hit arriving before current work.
        for future in service._futures.values():
            future.result()
        service._futures.clear()
        current = service._pending_paths()[0]
        service._ready = {first.record.path: service.cache.fetch(first.record.path)}
        pending = Future()
        pending.set_result(service.cache.fetch(current))
        service._futures[current] = pending
        second = service.lease(0)
        assert second is not None and second.record.path == current
    finally:
        service.close()


def test_rollover_retains_slow_lease_and_late_ack(tmp_path):
    service = make_service(tmp_path, repeat=True, count=2)
    try:
        service.start()
        slow = service.lease(0)
        fast = service.lease(1)
        assert slow is not None and fast is not None
        ack(service, fast)
        newer = service.lease(1)
        assert newer is not None and newer.cycle_index == 1
        assert newer.record.path != slow.record.path
        assert slow.lease_id in service._outstanding
        assert slow.record.path in service._protected_paths()
        assert slow.local_path.read_bytes()
        assert service.lease(0) == slow
        before = service.state
        ack(service, slow)
        assert service.state == before
        ack(service, slow)
        assert not service.done
        ack(service, newer)
        # Once the held path is released it returns to eligibility.
        following = service.lease(0)
        assert following.record.path == slow.record.path
    finally:
        service.close()


def test_ack_window_bounded_and_validates_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("sakuradatasetservice.service._RETIRED_LEASE_CAP", 2)
    service = make_service(tmp_path, repeat=True, workers=1)
    try:
        service.start()
        leases = []
        for _ in range(4):
            lease = service.lease(0)
            assert lease is not None
            leases.append(lease)
            ack(service, lease)
        assert len(service._retired_leases) == 2
        ack(service, leases[-1])
        with pytest.raises(DataServiceError):
            ack(service, replace(leases[-1], state_revision=999999))
        with pytest.raises(DataServiceError):
            ack(service, leases[0])
    finally:
        service.close()


def test_restart_reissues_unacknowledged_work(tmp_path):
    service = make_service(tmp_path, count=1, workers=1)
    service.start()
    lease = service.lease(0)
    service.close()
    restarted = make_service(tmp_path, count=1, workers=1)
    try:
        restarted.start()
        replacement = restarted.lease(0)
        assert replacement.record == lease.record
        assert restarted.identity.session_id != service.identity.session_id
        assert replacement.lease_id != lease.lease_id
        with pytest.raises(DataServiceError):
            ack(restarted, lease)
    finally:
        restarted.close()


def test_lease_context_ack_only_success(tmp_path):
    service = make_service(tmp_path, count=1, workers=1)

    class Client:
        calls = 0

        def acknowledge(self, descriptor):
            self.calls += 1

    client = Client()
    try:
        service.start()
        descriptor = service.lease(0)
        with pytest.raises(ValueError), ShardLease(client, descriptor):
            raise ValueError("consumer failed")
        assert client.calls == 0
        lease = ShardLease(client, descriptor)
        with lease as item:
            assert item == descriptor
        lease.acknowledge()
        assert client.calls == 1
    finally:
        service.close()


def test_ranked_ack_failure_keeps_retry_mapping(tmp_path):
    from sakuradatasetservice import RankedDataServiceClient

    service = make_service(tmp_path, count=1, workers=1)
    try:
        service.start()
        descriptor = service.lease(0)

        class Delegate:
            identity = service.identity
            calls = 0

            def lease(self, worker_id):
                return descriptor

            def acknowledge(self, lease):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("ACK response lost")

        delegate = Delegate()
        client = RankedDataServiceClient(
            delegate, rank=0, workers_per_rank=1, world_size=1
        )
        lease = client.lease(0)
        with pytest.raises(OSError):
            client.acknowledge(lease)
        client.acknowledge(lease)
        client.acknowledge(lease)
        assert delegate.calls == 3
    finally:
        service.close()


def test_completed_once_queue_stays_completed_on_restart(tmp_path):
    service = make_service(tmp_path, count=1, workers=1)
    service.start()
    ack(service, service.lease(0))
    service.close()
    restarted = make_service(tmp_path, count=1, workers=1)
    try:
        restarted.start()
        assert restarted.wait_until_ready()
        assert restarted.done
        assert restarted.lease(0) is None
    finally:
        restarted.close()


def test_queue_rejects_changed_manifest(tmp_path):
    service = make_service(tmp_path, count=1, workers=1)
    service.start()
    service.close()
    # The queue carries source and inventory, not merely matching shard paths.
    service.store.manifest = service.manifest.model_copy(
        update={"source": DatasetSourceIdentity(repo_id="local/other", revision="v1")}
    )
    with pytest.raises(DataServiceError):
        service.store.load()


def test_repeat_all_paths_held_is_temporary_without_cycle_spin(tmp_path):
    service = make_service(tmp_path, repeat=True, count=2, workers=3)
    try:
        service.start()
        slow = service.lease(0)
        fast = service.lease(1)
        cycle = service.state.cycle
        for _ in range(3):
            assert service.lease(2) is None
            assert service.state.cycle == cycle
            assert not service.done
        for _ in range(4):
            ack(service, fast)
            fast = service.lease(1)
            assert fast.record.path != slow.record.path
            assert slow.record.path in service._protected_paths()
        ack(service, slow)
        available = service.lease(0)
        assert available.record.path == slow.record.path
    finally:
        service.close()


def test_iterator_polls_temporary_unavailability_and_can_cancel(monkeypatch):
    from sakuradatasetservice import DataServiceClient

    sleeps = []
    monkeypatch.setattr("sakuradatasetservice.client.time.sleep", sleeps.append)

    class Client:
        calls = 0

        def acquire(self, worker_id):
            self.calls += 1

        def health(self):
            return False

    client = Client()
    iterator = DataServiceClient.iter_leases(
        client, 0, poll_interval_seconds=0.1, cancelled=lambda: len(sleeps) == 2
    )
    assert list(iterator) == []
    assert client.calls == 2
    assert sleeps == [0.1, 0.1]


def test_iterator_waits_until_global_done(monkeypatch):
    from sakuradatasetservice import DataServiceClient

    sleeps = []
    monkeypatch.setattr("sakuradatasetservice.client.time.sleep", sleeps.append)

    class Client:
        checks = 0

        def acquire(self, worker_id):
            return None

        def health(self):
            self.checks += 1
            return self.checks == 2

    client = Client()
    assert list(DataServiceClient.iter_leases(client, 0)) == []
    assert client.checks == 2
    assert sleeps == [0.05]
