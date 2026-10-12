import subprocess
import sys
from pathlib import Path

from sakuradatasetservice import (
    CacheQuota,
    DatasetSourceIdentity,
    LocalDatasetProvider,
    ShardCache,
)


def test_local_provider_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.tar").write_bytes(b"local shard")
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/test", revision="v1")
    )
    cache = ShardCache(tmp_path / "cache", manifest, provider, CacheQuota(0, 100))
    fetched = cache.fetch("a.tar").fetched
    assert fetched.path.read_bytes() == b"local shard"
    assert not fetched.cache_hit
    assert cache.fetch("a.tar").fetched.cache_hit


def test_package_import_has_no_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sakuradatasetservice; assert not any(x == 'torch' or x.startswith(('torch.', 'sakuramoon')) for x in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_source_has_no_training_imports():
    root = Path(__file__).parents[1] / "src"
    for path in root.rglob("*.py"):
        assert "sakuramoon" not in path.read_text().lower(), path


def test_service_accepts_two_method_provider(tmp_path):
    from sakuradatasetservice import (
        DataServiceLimits,
        DataServiceSessionIdentity,
        DataSupplyService,
    )
    from sakuradatasetservice.validation import ValidationSelection

    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.tar", "b.tar"):
        (source / name).write_bytes(b"shard")
    local = LocalDatasetProvider(source)

    class MinimalProvider:
        def fetch_manifest(self, source):
            return local.fetch_manifest(source)

        def fetch_shard(self, *args, **kwargs):
            return local.fetch_shard(*args, **kwargs)

    provider = MinimalProvider()
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/test", revision="v1")
    )
    selection = ValidationSelection(
        "test", manifest.dataset_id, 44, (manifest.shards[0],)
    )
    cache = ShardCache(tmp_path / "cache", manifest, provider, CacheQuota(0, 100))
    service = DataSupplyService(
        manifest,
        selection,
        cache,
        tmp_path / "queue.json",
        tmp_path / "owner.lock",
        DataServiceSessionIdentity(manifest.dataset_id, 1),
        DataServiceLimits(1, 1, 1, 1),
    )
    try:
        service.start()
        assert service.wait_until_ready()
    finally:
        service.close()
