"""Folder sampling without changing once/repeat allocation and cache contracts."""
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import PurePosixPath

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
    RankedDataServiceClient,
    ShardCache,
)
from sakuradatasetservice.manifest import DatasetManifest, ShardRecord
from sakuradatasetservice.service import DataServiceError, _QueueStore
from sakuradatasetservice.service_protocol import canonical_json_bytes

PATHS = ("large/a.tar", "large/b.tar", "large/c.tar", "small/one.tar", "holdout/only.tar")
EXCLUDED = frozenset({"holdout/only.tar"})


def inputs(paths=PATHS):
    return DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="local/balance", revision="fixed"),
        tuple(ShardRecord(path=p, bytes=4) for p in paths),
    )


def store(tmp_path, *, paths=PATHS, excluded=EXCLUDED, seed=44, sampling="balance"):
    return _QueueStore(tmp_path / "queue.json", inputs(paths), excluded,
                       sampling=sampling, seed=seed)


def paths_of(state):
    return tuple(row.path for row in state.rows)


def make_service(tmp_path, *, paths=PATHS, excluded=EXCLUDED, repeat=False,
                 workers=2, seed=44, sampling="balance", high=2**20):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    for p in paths:
        destination = source / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with tarfile.open(destination, "w") as tar:
                payload = p.encode()
                member = tarfile.TarInfo("sample.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(DatasetSourceIdentity(repo_id="local/balance", revision="fixed"))
    cache = ShardCache(tmp_path / "cache", manifest, provider, CacheQuota(0, high))
    return DataSupplyService(
        manifest, cache, tmp_path / "queue.json", tmp_path / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, workers),
        DataServiceLimits(1, workers, workers, workers),
        excluded_paths=excluded, repeat=repeat, sampling=sampling, seed=seed,
    )


def ack(service, lease):
    service.acknowledge(lease.lease_id, lease.worker_id, lease.state_revision)


@pytest.mark.parametrize("seed", [-1, True, False, 1.0, "44", None])
def test_api_rejects_bad_seed_before_creating_state(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        make_service(tmp_path, seed=seed)
    assert not (tmp_path / "queue.json").exists()


@pytest.mark.parametrize("sampling", ["balanced", "Balance", "", None, 1, True, ["balance"]])
def test_api_rejects_bad_sampling(tmp_path, sampling):
    with pytest.raises(ValueError, match="sampling"):
        make_service(tmp_path, sampling=sampling)


def test_exclusions_uneven_folders_and_uniform_choice(tmp_path):
    excluded = EXCLUDED | {"large/c.tar"}
    candidate = store(tmp_path, excluded=excluded)
    counts = Counter()
    for cycle in range(600):
        chosen = paths_of(candidate.new(cycle))
        assert len(chosen) == 2
        assert {PurePosixPath(p).parent.as_posix() for p in chosen} == {"large", "small"}
        assert set(chosen).isdisjoint(excluded)
        counts.update(chosen)
    assert counts["small/one.tar"] == 600
    assert 240 < counts["large/a.tar"] < 360
    assert counts["large/a.tar"] + counts["large/b.tar"] == 600
    assert len(candidate.paths) == 3


@pytest.mark.parametrize("paths,expected", [
    (("a.tar", "b.tar"), {"."}),
    (("one/a.tar", "one/b.tar"), {"one"}),
    (("a.tar", "b.tar", "one/single.tar"), {".", "one"}),
])
def test_root_single_folder_and_singleton(tmp_path, paths, expected):
    candidate = store(tmp_path, paths=paths, excluded=frozenset())
    for cycle in range(5):
        chosen = paths_of(candidate.new(cycle))
        assert len(chosen) == len(expected)
        assert {PurePosixPath(p).parent.as_posix() for p in chosen} == expected


def test_all_excluded_or_absent_path_rejected(tmp_path):
    with pytest.raises(DataServiceError, match="no available"):
        store(tmp_path, excluded=frozenset(PATHS))
    with pytest.raises(DataServiceError, match="absent"):
        store(tmp_path, excluded=frozenset({"unknown.tar"}))


def test_same_cycle_seed_restart_and_lookahead_no_rng_side_effects(tmp_path):
    candidate = store(tmp_path)
    rng = random.getstate()
    preview = candidate.new(1)
    assert preview == store(tmp_path).new(1)
    for cycle in range(10):
        candidate.new(cycle)
    assert candidate.new(1) == preview
    assert random.getstate() == rng
    assert not candidate.path.exists()
    assert any(candidate.new(c) != store(tmp_path, seed=45).new(c) for c in range(8))
    assert len({paths_of(candidate.new(c)) for c in range(10)}) > 1


def test_order_matches_shared_balance_algorithm(tmp_path):
    # This sorted-input/local-RNG algorithm is shared with SakuraMoon08ed12c.
    candidate = store(tmp_path)
    for seed in (0, 44, 123):
        current = store(tmp_path, seed=seed)
        for cycle in range(8):
            groups = {}
            for path in sorted(candidate.paths):
                groups.setdefault(PurePosixPath(path).parent.as_posix(), []).append(path)
            rng = random.Random(f"balance:{seed}:{cycle}")
            selected = [rng.choice(group) for _, group in sorted(groups.items())]
            rng.shuffle(selected)
            assert paths_of(current.new(cycle)) == tuple(selected)


def test_hash_seed_process_independence(tmp_path):
    snippet = '''
from pathlib import Path
from sakuradatasetservice import DatasetSourceIdentity
from sakuradatasetservice.manifest import DatasetManifest, ShardRecord
from sakuradatasetservice.service import _QueueStore
m = DatasetManifest.from_shards(DatasetSourceIdentity(repo_id="local/test", revision="v1"),
    tuple(ShardRecord(path=p, bytes=1) for p in ("a/x.tar", "a/y.tar", "b/z.tar", "root.tar")))
s = _QueueStore(Path("unused"), m, frozenset(), sampling="balance", seed=44)
print([[r.path for r in s.new(c).rows] for c in range(5)])
'''
    # Preserve pytest's source path when the package has not been installed.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    results = [subprocess.run([sys.executable, "-c", snippet], cwd=tmp_path,
               env=dict(env, PYTHONHASHSEED=seed), text=True, capture_output=True, check=True).stdout
               for seed in ("1", "2", "12345")]
    assert results[0] == results[1] == results[2]


def test_schema4_restore_active_completed_and_no_sidecar(tmp_path):
    candidate = store(tmp_path)
    state = candidate.new(3)
    active = replace(state.rows[0], status="active")
    completed = replace(state.rows[1], status="completed")
    candidate.save(replace(state, rows=(active, completed)))
    raw = json.loads(candidate.path.read_bytes())
    assert set(raw) == {"cycle", "rows", "manifest", "schema_version", "sampling", "seed", "excluded_paths"}
    assert raw["schema_version"] == 4
    assert raw["manifest"] == candidate.manifest.model_dump(mode="json")
    assert raw["sampling"] == "balance" and raw["seed"] == 44
    assert raw["excluded_paths"] == sorted(EXCLUDED)
    assert store(tmp_path).load().rows == (replace(active, status="pending"), completed)
    assert tuple(tmp_path.iterdir()) == (candidate.path,)


@pytest.mark.parametrize("singletons", [False, True])
def test_resume_rejects_mode_seed_and_exact_exclusion_changes(tmp_path, singletons):
    paths = ("a/one.tar", "b/one.tar", "excluded.tar") if singletons else PATHS
    excluded = frozenset({"excluded.tar"}) if singletons else EXCLUDED
    candidate = store(tmp_path, paths=paths, excluded=excluded)
    candidate.save(candidate.new(0))
    before = candidate.path.read_bytes()
    for mode, seed in (("shuffle", 44), ("balance", 45)):
        with pytest.raises(DataServiceError, match="could not be loaded"):
            store(tmp_path, paths=paths, excluded=excluded, sampling=mode, seed=seed).load()
        assert candidate.path.read_bytes() == before
    # Exclude an unchosen tar: cycle0 membership/order can still coincide, but
    # the exact excluded set must not silently alter future-cycle candidates.
    if not singletons:
        extra = next(p for p in candidate.paths if p not in paths_of(candidate.new(0)))
        with pytest.raises(DataServiceError, match="could not be loaded"):
            store(tmp_path, excluded=excluded | {extra}).load()


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 45), ("sampling", "shuffle"),
    ("schema_version", 4.0), ("excluded_paths", []), ("cycle", True)])
def test_bad_queue_metadata_rejected(tmp_path, field, value):
    candidate = store(tmp_path)
    candidate.save(candidate.new(0))
    raw = json.loads(candidate.path.read_bytes())
    raw[field] = value
    candidate.path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(DataServiceError, match="could not be loaded"):
        candidate.load()


def test_exact_cycle_selection_order_status_and_full_manifest_binding(tmp_path):
    candidate = store(tmp_path)
    state = candidate.new(0)
    for rows in (state.rows[::-1], state.rows[:-1],
                 (replace(state.rows[0], status="bad"), *state.rows[1:])):
        with pytest.raises(DataServiceError, match="invalid"):
            candidate.save(replace(state, rows=rows))
    candidate.save(state)
    raw = json.loads(candidate.path.read_bytes())
    unselected = next(r for r in raw["manifest"]["shards"] if r["path"] not in paths_of(state))
    unselected["bytes"] += 1
    candidate.path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(DataServiceError, match="could not be loaded"):
        candidate.load()


def test_shuffle_legacy_schema3_sequence_and_resume_unchanged(tmp_path):
    candidate = store(tmp_path, sampling="shuffle", seed=999)
    expected = sorted(candidate.paths)
    random.Random((0x9E3779B97F4A7C15 * 4) & 0xFFFFFFFFFFFFFFFF).shuffle(expected)
    assert paths_of(candidate.new(3)) == tuple(expected)
    assert candidate.new(3) == store(tmp_path, sampling="shuffle", seed=0).new(3)
    state = candidate.new(3)
    # Legacy shuffle restore accepts arbitrary valid order, as before.
    state = replace(state, rows=tuple(replace(r, status="active") for r in state.rows[::-1]))
    candidate.save(state)
    raw = json.loads(candidate.path.read_bytes())
    assert set(raw) == {"manifest", "cycle", "rows", "schema_version"}
    assert raw["schema_version"] == 3
    assert paths_of(candidate.load()) == paths_of(state)
    assert all(r.status == "pending" for r in candidate.load().rows)
    with pytest.raises(DataServiceError, match="could not be loaded"):
        store(tmp_path).load()


def test_once_one_per_folder_requires_all_acks_and_stays_done_after_restart(tmp_path):
    service = make_service(tmp_path)
    try:
        service.start()
        selected = set(paths_of(service.state))
        first, second = service.lease(0), service.lease(1)
        assert {first.record.path, second.record.path} == selected
        assert service.lease(0) is first
        assert not service.done
        ack(service, first)
        assert service.lease(0) is None and not service.done
        ack(service, second)
        assert service.done and service.state.cycle == 0
    finally:
        service.close()
    restarted = make_service(tmp_path)
    try:
        restarted.start()
        assert restarted.done and restarted.wait_until_ready()
        assert restarted.lease(0) is None
    finally:
        restarted.close()


def test_restart_preserves_completed_replays_active_with_new_session(tmp_path):
    service = make_service(tmp_path)
    service.start()
    first, slow = service.lease(0), service.lease(1)
    ack(service, first)
    partial = service.cache.root / "large/a.tar.partial"
    partial.write_bytes(b"progress")
    service.close()
    restarted = make_service(tmp_path)
    try:
        restarted.start()
        assert restarted.identity.session_id != service.identity.session_id
        assert tuple(r.path for r in restarted.state.rows if r.status == "completed") == (first.record.path,)
        replacement = restarted.lease(0)
        assert replacement.record.path == slow.record.path
        assert replacement.lease_id != slow.lease_id
        assert partial.read_bytes() == b"progress"
        with pytest.raises(DataServiceError, match="ACK"):
            ack(restarted, slow)
        ack(restarted, replacement)
        assert restarted.done
    finally:
        restarted.close()


def test_repeat_real_multi_cycle_selection_and_no_duplicate_active(tmp_path):
    service = make_service(tmp_path, repeat=True, workers=1)
    try:
        service.start()
        for cycle in range(12):
            selected = paths_of(service.store.new(cycle))
            seen = []
            for _ in selected:
                lease = service.lease(0)
                assert lease.cycle_index == cycle
                assert service.lease(0) is lease
                seen.append(lease.record.path)
                ack(service, lease)
            assert set(seen) == set(selected)
            assert service.state.cycle == cycle + 1 and not service.done
    finally:
        service.close()


def test_slow_old_cycle_lease_is_pinned_and_ack_only_releases_it(tmp_path):
    service = make_service(tmp_path, repeat=True)
    try:
        service.start()
        slow = service.lease(0)
        fast = service.lease(1)
        ack(service, fast)
        newer = service.lease(1)
        assert newer.cycle_index == 1 and newer.record.path != slow.record.path
        assert service.lease(0) is slow
        assert slow.record.path in service._protected_paths()
        assert slow.local_path.is_file()
        before = service.state
        ack(service, slow)
        assert service.state == before
        ack(service, slow)  # Existing ACK retry window remains idempotent.
        assert slow.lease_id not in service._outstanding
        ack(service, newer)
    finally:
        service.close()


def test_next_selected_all_held_does_not_skip_or_spin_cycle(tmp_path):
    # One folder, several candidates: the old all-candidate availability check
    # incorrectly advanced even when the NEXT selected tar was already held.
    paths = tuple(f"one/{i}.tar" for i in range(6))
    seed = next(s for s in range(100) if paths_of(store(tmp_path, paths=paths,
        excluded=frozenset(), seed=s).new(0)) == paths_of(store(tmp_path, paths=paths,
        excluded=frozenset(), seed=s).new(1)))
    service = make_service(tmp_path, paths=paths, excluded=frozenset(), repeat=True,
                           workers=2, seed=seed)
    try:
        service.start()
        slow = service.lease(0)
        assert service.store.paths - {slow.record.path}, "other catalog candidates are free"
        before, revision = service.state, service._revision
        for _ in range(8):
            assert service.lease(1) is None
            assert not service.done
            assert service.state == before and service._revision == revision
        ack(service, slow)
        replacement = service.lease(1)
        assert replacement.cycle_index == 1 and replacement.record.path == slow.record.path
    finally:
        service.close()


def test_older_cycle_held_selection_blocks_next_without_redrawing(tmp_path):
    paths = tuple(f"one/{i}.tar" for i in range(6))
    def selections(seed):
        candidate = store(tmp_path, paths=paths, excluded=frozenset(), seed=seed)
        return [paths_of(candidate.new(c))[0] for c in range(3)]
    seed = next(s for s in range(1000)
                if selections(s)[0] == selections(s)[2] != selections(s)[1])
    service = make_service(tmp_path, paths=paths, excluded=frozenset(), repeat=True,
                           workers=3, seed=seed)
    try:
        service.start()
        slow = service.lease(0)
        fast = service.lease(1)
        assert slow.cycle_index == 0 and fast.cycle_index == 1
        assert fast.record.path != slow.record.path
        before, revision = service.state, service._revision
        for _ in range(5):
            assert service.lease(2) is None
            assert service.state == before and service._revision == revision
            assert not service.done
        ack(service, slow)
        assert service.state == before, "old-cycle ACK cannot complete current selection"
        following = service.lease(2)
        assert following.cycle_index == 2 and following.record.path == slow.record.path
        assert fast.lease_id in service._outstanding
        assert fast.record.path in service._protected_paths()
        assert len({lease.record.path for lease in service._outstanding.values()}) == 2
    finally:
        service.close()


def test_fast_ack_cannot_advance_into_all_held_selection_and_skip_it(tmp_path):
    paths = tuple(f"one/{i}.tar" for i in range(6))
    def selections(seed):
        candidate = store(tmp_path, paths=paths, excluded=frozenset(), seed=seed)
        return [paths_of(candidate.new(c))[0] for c in range(4)]
    seed = next(s for s in range(1000)
                if selections(s)[0] == selections(s)[2] != selections(s)[1]
                and selections(s)[3] != selections(s)[0])
    service = make_service(tmp_path, paths=paths, excluded=frozenset(), repeat=True,
                           workers=3, seed=seed)
    try:
        service.start()
        slow = service.lease(0)
        fast = service.lease(1)
        assert slow.cycle_index == 0 and fast.cycle_index == 1
        # ACK completes cycle1, but cycle2 selects ONLY the still-held cycle0 A.
        ack(service, fast)
        assert service.state.cycle == 1
        assert all(row.status == "completed" for row in service.state.rows)
        before, revision = service.state, service._revision
        for _ in range(5):
            assert service.lease(1) is None
            assert service.lease(2) is None
            assert service.state == before and service._revision == revision
            assert not service.done
            assert slow.record.path in service._protected_paths()
        ack(service, slow)
        assert service.state == before
        following = service.lease(1)
        assert following.cycle_index == 2
        assert following.record.path == slow.record.path
    finally:
        service.close()


def test_exclusion_binding_rejects_even_when_cycle_selected_order_coincides(tmp_path):
    for seed in range(100):
        candidate = store(tmp_path, seed=seed)
        selected = paths_of(candidate.new(0))
        extra = next(p for p in sorted(candidate.paths) if p not in selected)
        changed = store(tmp_path, seed=seed, excluded=EXCLUDED | {extra})
        if paths_of(changed.new(0)) == selected:
            break
    else:
        pytest.fail("test needs a coincident selection with changed exclusions")
    candidate.save(candidate.new(0))
    assert paths_of(changed.new(0)) == selected
    with pytest.raises(DataServiceError, match="could not be loaded"):
        changed.load()


def test_nonselected_old_candidate_cache_evicted_with_full_quota_and_namespace(tmp_path):
    # One cached tar + 2x transfer peak would be 30720, one byte too high.
    service = make_service(tmp_path, high=30719)
    try:
        selected = set(paths_of(service.store.new(0)))
        previous = next(p for p in service.store.paths if p not in selected)
        target = next(p for p in selected if p.startswith("large/"))
        assert service.cache.manifest is service.manifest
        service.cache.fetch(previous)
        result = service.cache.fetch(target)
        assert previous in result.evicted_paths
        assert result.fetched.path.is_file()
        assert service.cache.usage_bytes() == 10240
        assert service.cache.quota.high_bytes == 30719
    finally:
        service.close()


def test_real_unix_socket_tiny_tars_ranked_clients_balance_repeat(tmp_path):
    service = make_service(tmp_path, repeat=True)
    server = DataServiceServer(service, tmp_path / "balance.sock", request_timeout_seconds=3.0)
    stop, ready = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(server.serve, stop, ready_callback=ready.set)
        try:
            assert ready.wait(5), "balance server did not start"
            delegate = DataServiceClient(tmp_path / "balance.sock", worker_count=2,
                                         request_timeout_seconds=3.0)
            clients = tuple(RankedDataServiceClient(delegate, rank=rank, workers_per_rank=1,
                                                     world_size=2) for rank in range(2))
            for cycle in range(3):
                leases = [client.lease(0) for client in clients]
                assert {lease.record.path for lease in leases} == set(paths_of(service.store.new(cycle)))
                for client, lease in zip(clients, leases):
                    assert lease.cycle_index == cycle
                    with tarfile.open(lease.local_path) as tar:
                        assert tar.extractfile(tar.next()).read().decode() == lease.record.path
                    client.acknowledge(lease)
                assert not delegate.health()
        finally:
            stop.set()
            future.result(timeout=5)
    assert not (tmp_path / "balance.sock").exists()
    assert service.store.load() == service.state


def test_lookahead_future_only_ready_does_not_starve_current(tmp_path):
    service = make_service(tmp_path, repeat=True, workers=1)
    try:
        service.start()
        first = service.lease(0)
        ack(service, first)
        # Finish background tasks before replacing their test harness state.
        for future in service._futures.values():
            future.result()
        service._futures.clear()
        next_only = next(p for p in service.store.paths if p not in paths_of(service.state))
        service._ready = {next_only: service.cache.fetch(next_only)}
        current = service.lease(0)
        assert current.record.path in paths_of(service.state)
        assert current.record.path != next_only
    finally:
        service.close()
