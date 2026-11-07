"""CLI sampler options validate before provider IO and reach the actual service."""
import json
import os
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from sakuradatasetservice import (
    DataServiceClient,
    DatasetSourceIdentity,
    LocalDatasetProvider,
    cli,
)
from sakuradatasetservice.manifest import write_dataset_manifest


@pytest.mark.parametrize("options,mode,seed,repeat", [
    ([], "shuffle", 0, False),
    (["--sampling", "balance", "--seed", "44", "--repeat"], "balance", 44, True),
    (["--sampling", "shuffle", "--seed", "99"], "shuffle", 99, False),
])
def test_cli_wires_default_and_explicit_sampling(tmp_path, monkeypatch, options, mode, seed, repeat):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.tar").write_bytes(b"test")
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(DatasetSourceIdentity(repo_id="local/cli", revision="v1"))
    path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest, path)
    constructor = Mock()
    server = Mock()
    monkeypatch.setattr(cli, "DataSupplyService", constructor)
    monkeypatch.setattr(cli, "DataServiceServer", Mock(return_value=server))
    monkeypatch.setattr(cli.signal, "signal", Mock())
    cli.main(["serve", "--local-root", str(source), "--manifest", str(path),
              "--cache-root", str(tmp_path / "cache"), "--state-dir", str(tmp_path / "state"),
              "--cache-high-bytes", "1024", *options])
    assert constructor.call_args.kwargs == {
        "repeat": repeat, "excluded_paths": frozenset(), "sampling": mode, "seed": seed,
    }
    server.serve.assert_called_once()


@pytest.mark.parametrize("options", [
    ["--sampling", "balanced"], ["--sampling", "Balance"],
    ["--seed", "-1"], ["--seed", "True"], ["--seed", "1.0"],
])
def test_cli_invalid_sampler_rejected_before_provider_or_cache(tmp_path, monkeypatch, options):
    provider = Mock(side_effect=AssertionError("provider must not be consulted"))
    monkeypatch.setattr(cli, "_provider", provider)
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--local-root", str(tmp_path / "missing"),
                  "--manifest", str(tmp_path / "missing.json"),
                  "--cache-root", str(tmp_path / "cache"), "--state-dir", str(tmp_path / "state"),
                  "--cache-high-bytes", "1024", *options])
    assert error.value.code == 2
    provider.assert_not_called()
    assert not (tmp_path / "cache").exists()


def test_real_cli_process_balance_once_and_sigterm(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a/one.tar", "a/two.tar", "b/one.tar"):
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(DatasetSourceIdentity(repo_id="local/cli", revision="v1"))
    manifest_path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    state = tmp_path / "state"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    with subprocess.Popen([
        sys.executable, "-m", "sakuradatasetservice", "serve", "--local-root", str(source),
        "--manifest", str(manifest_path), "--cache-root", str(tmp_path / "cache"),
        "--state-dir", str(state), "--cache-high-bytes", "1024", "--workers", "1",
        "--sampling", "balance", "--seed", "44",
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            deadline = time.monotonic() + 10
            while not (state / "data.sock").exists() and process.poll() is None:
                assert time.monotonic() < deadline, "CLI never became ready"
                time.sleep(0.02)
            assert process.poll() is None
            client = DataServiceClient(state / "data.sock", worker_count=1,
                                       request_timeout_seconds=3.0)
            paths = []
            while (lease := client.lease(0)) is not None:
                paths.append(lease.record.path)
                assert lease.local_path.read_bytes().decode() == lease.record.path
                client.acknowledge(lease)
            assert len(paths) == 2 and {p.split("/")[0] for p in paths} == {"a", "b"}
            assert client.health()
        finally:
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
    assert not (state / "data.sock").exists()
    raw = json.loads((state / "queue.json").read_bytes())
    assert raw["sampling"] == "balance" and raw["seed"] == 44
    assert raw["schema_version"] == 4
    assert all(r["status"] == "completed" for r in raw["rows"])
