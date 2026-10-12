import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sakuradatasetservice import (
    CacheQuota,
    DatasetSourceIdentity,
    LocalDatasetProvider,
    ShardCache,
)
from sakuradatasetservice.cache import ShardCacheError
from sakuradatasetservice.manifest import DatasetManifest, ShardRecord
from sakuradatasetservice.modelscope import DatasetTransientError, ShardIntegrityError


def make_cache(tmp_path, *, size=8, count=2, high=32, low=0, provider_type=None):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(count):
        (source / f"{index}.tar").write_bytes(bytes(range(size)))
    provider = (provider_type or LocalDatasetProvider)(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/corpus", revision="v1")
    )
    cache = ShardCache(tmp_path / "cache", manifest, provider, CacheQuota(low, high))
    return cache, provider


def payload_bytes(root):
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.name != ShardCache._MARKER
    )


def test_peak_reservation_rejects_before_download(tmp_path):
    cache, _ = make_cache(tmp_path, high=15)
    with pytest.raises(ShardCacheError, match="transfer peak"):
        cache.fetch("0.tar")
    assert payload_bytes(cache.root) == 0
    assert cache._reservations == {}


def test_exact_two_shard_size_quota_and_cache_hit(tmp_path):
    cache, _ = make_cache(tmp_path, high=16, count=1)
    result = cache.fetch("0.tar")
    assert result.fetched.path.read_bytes() == bytes(range(8))
    assert result.usage_bytes == 8
    assert cache.fetch("0.tar").fetched.cache_hit
    # A verified cache hit needs no second copy.
    smaller = ShardCache(cache.root, cache.manifest, cache.transport, CacheQuota(0, 8))
    assert smaller.fetch("0.tar").fetched.cache_hit


def test_relative_cache_root_publishes_absolute_paths(tmp_path, monkeypatch):
    cache, provider = make_cache(tmp_path, count=1)
    monkeypatch.chdir(tmp_path)
    relative = ShardCache(Path("relative"), cache.manifest, provider, cache.quota)
    fetched = relative.fetch("0.tar").fetched
    assert relative.root == tmp_path / "relative"
    assert fetched.path.is_absolute()


def test_partials_assemblies_and_unknown_files_consume_quota(tmp_path):
    cache, _ = make_cache(tmp_path)
    (cache.root / "0.tar.partial").write_bytes(b"abc")
    (cache.root / "0.tar.partial.range-000000000003-000000000007").write_bytes(b"de")
    (cache.root / "0.tar.partial.assembling").write_bytes(b"abcd")
    (cache.root / "1.tar").write_bytes(bytes(range(8)))
    (cache.root / "unexpected").write_bytes(b"extra")
    assert cache.usage_bytes() == 22
    assert sum(size for _, size in cache.resumable_partials()) == 9


def test_orphan_ranges_are_counted_without_prefix_file(tmp_path):
    cache, _ = make_cache(tmp_path)
    (cache.root / "0.tar.partial.range-000000000000-000000000007").write_bytes(b"abc")
    assert cache.usage_bytes() == 3
    assert cache.resumable_partials() == (("0.tar", 3),)


def test_concurrent_range_assembly_never_exceeds_exact_quota(tmp_path, monkeypatch):
    class TwoRangeProvider(LocalDatasetProvider):
        streams_per_shard = 2
        stream_chunk_bytes = 2

    cache, _ = make_cache(tmp_path, provider_type=TwoRangeProvider)
    # Include matching ranges, obsolete ranges and a stale assembly from a crash.
    (cache.root / "0.tar.partial").write_bytes(bytes(range(3)))
    (cache.root / "0.tar.partial.range-000000000003-000000000005").write_bytes(b"\x03")
    (cache.root / "0.tar.partial.range-000000000000-000000000003").write_bytes(b"old!")
    (cache.root / "0.tar.partial.assembling").write_bytes(b"junk")
    (cache.root / "1.tar.partial").write_bytes(bytes(range(2)))
    (cache.root / "1.tar.partial.range-000000000002-000000000004").write_bytes(
        b"\x02\x03"
    )
    first_assembly_ready = threading.Event()
    both_assemblies_ready = threading.Barrier(2)
    observed = []
    original_replace = os.replace

    def inspect_replace(source, destination):
        if str(source).endswith(".partial.assembling"):
            if Path(source).name.startswith("0.tar"):
                first_assembly_ready.set()
            both_assemblies_ready.wait(timeout=10)
            # Keep both full assembly copies present until both inspect the peak.
            observed.append(payload_bytes(cache.root))
            both_assemblies_ready.wait(timeout=10)
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", inspect_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cache.fetch, "0.tar")
        assert first_assembly_ready.wait(timeout=10)
        # The first reservation already contains 16 bytes on disk. The second
        # admission must not count those bytes again on top of its reservation.
        second = executor.submit(cache.fetch, "1.tar")
        for result in (first.result(timeout=15), second.result(timeout=15)):
            assert result.fetched.path.read_bytes() == bytes(range(8))
    assert observed == [32, 32]
    assert cache.usage_bytes() == 16
    assert cache.resumable_partials() == ()
    assert cache._reservations == {}


def test_protected_shard_blocks_peak_but_lru_can_be_evicted(tmp_path):
    cache, _ = make_cache(tmp_path, high=23)
    cache.fetch("0.tar")
    with pytest.raises(ShardCacheError, match="protected"):
        cache.fetch("1.tar", protected_paths=frozenset({"0.tar"}))
    assert (cache.root / "0.tar").is_file()
    result = cache.fetch("1.tar")
    assert result.evicted_paths == ("0.tar",)
    assert cache.usage_bytes() == 8


def test_eviction_obeys_lru_and_low_watermark(tmp_path):
    cache, _ = make_cache(tmp_path, count=3, high=31, low=24)
    cache.fetch("0.tar")
    cache.fetch("1.tar")
    os.utime(cache.root / "0.tar", ns=(1, 1))
    os.utime(cache.root / "1.tar", ns=(2, 2))
    result = cache.fetch("2.tar")
    assert result.evicted_paths == ("0.tar",)
    assert (cache.root / "1.tar").is_file()
    assert result.usage_bytes == 16


def test_failed_transfer_keeps_partial_bytes_and_releases_reservation(tmp_path):
    class InterruptedProvider(LocalDatasetProvider):
        interrupted = True

        def download(self, manifest, shard, output, **kwargs):
            if self.interrupted:
                output.write(bytes(range(2)))
                raise DatasetTransientError("test interruption")
            return super().download(manifest, shard, output, **kwargs)

    cache, provider = make_cache(tmp_path, high=17, provider_type=InterruptedProvider)
    with pytest.raises(DatasetTransientError, match="interruption"):
        cache.fetch("0.tar")
    assert cache._reservations == {}
    assert cache.usage_bytes() == 2
    with pytest.raises(ShardCacheError, match="protected"):
        cache.fetch("1.tar")
    provider.interrupted = False
    assert cache.fetch("0.tar").fetched.path.read_bytes() == bytes(range(8))
    assert cache.usage_bytes() == 8


def test_range_writer_rejects_oversized_write_before_disk_growth(tmp_path):
    class OversizedProvider(LocalDatasetProvider):
        streams_per_shard = 2

        def download(self, manifest, shard, output, **kwargs):
            output.write(b"x" * shard.bytes)

    cache, _ = make_cache(tmp_path, count=1, high=16, provider_type=OversizedProvider)
    with pytest.raises(ShardIntegrityError, match="range is larger"):
        cache.fetch("0.tar")
    assert cache.usage_bytes() == 0
    assert cache._reservations == {}


def test_complete_prefix_cleans_stale_ranges_without_redownload(tmp_path):
    class NoDownloadProvider(LocalDatasetProvider):
        def download(self, *args, **kwargs):
            pytest.fail("a completed prefix should be published without downloading")

    cache, _ = make_cache(tmp_path, count=1, provider_type=NoDownloadProvider)
    (cache.root / "0.tar.partial").write_bytes(bytes(range(8)))
    (cache.root / "0.tar.partial.range-000000000000-000000000007").write_bytes(b"old")
    (cache.root / "0.tar.partial.assembling").write_bytes(b"junk")
    assert cache.fetch("0.tar").fetched.path.read_bytes() == bytes(range(8))
    assert cache.resumable_partials() == ()
    assert cache.usage_bytes() == 8


@pytest.mark.parametrize("change", ["source", "revision", "inventory", "size"])
def test_cache_marker_rejects_other_source_or_manifest(tmp_path, change):
    cache, provider = make_cache(tmp_path)
    original = cache.manifest
    source = original.source
    shards = original.shards
    if change == "source":
        source = DatasetSourceIdentity(repo_id="another/corpus", revision="v1")
    elif change == "revision":
        source = DatasetSourceIdentity(repo_id="local/corpus", revision="v2")
    elif change == "inventory":
        shards = (ShardRecord(path="other.tar", bytes=8),)
    else:
        shards = (ShardRecord(path="0.tar", bytes=9), original.shards[1])
    different = DatasetManifest.from_shards(source, shards)
    with pytest.raises(ShardCacheError, match="different manifest"):
        ShardCache(cache.root, different, provider, cache.quota)
    marker = json.loads((cache.root / ShardCache._MARKER).read_text())
    assert marker == original.model_dump(mode="json")
    assert "hash" not in (cache.root / ShardCache._MARKER).read_text()
    assert ShardCache(cache.root, original, provider, cache.quota).manifest == original


def test_unmarked_nonempty_root_is_not_silently_adopted(tmp_path):
    cache, provider = make_cache(tmp_path)
    root = tmp_path / "unbound"
    root.mkdir()
    existing = root / "0.tar"
    existing.write_bytes(b"old data")
    with pytest.raises(ShardCacheError, match="no manifest ownership marker"):
        ShardCache(root, cache.manifest, provider, cache.quota)
    assert existing.read_bytes() == b"old data"
    assert not (root / ShardCache._MARKER).exists()


def test_symlinked_cache_entries_are_rejected(tmp_path):
    cache, _ = make_cache(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "0.tar").write_bytes(b"untouched")
    (cache.root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ShardCacheError, match="regular files"):
        cache.fetch("0.tar")
    assert (outside / "0.tar").read_bytes() == b"untouched"
