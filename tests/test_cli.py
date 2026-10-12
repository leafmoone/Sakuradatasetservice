import subprocess
import sys

from sakuradatasetservice.manifest import parse_dataset_manifest_bytes


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "sakuradatasetservice", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "manifest" in result.stdout and "serve" in result.stdout
    assert run_cli("serve", "--help").returncode == 0


def test_cli_local_manifest_no_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.tar").write_bytes(b"test bytes")
    output = tmp_path / "manifest.json"
    args = (
        "manifest",
        "--local-root",
        source,
        "--repo-id",
        "local/cli",
        "--revision",
        "v1",
        "--output",
        output,
    )
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    manifest = parse_dataset_manifest_bytes(output.read_bytes())
    assert manifest.source.repo_id == "local/cli"
    assert manifest.shards[0].bytes == 10
    before = output.read_bytes()
    assert run_cli(*args).returncode != 0
    assert output.read_bytes() == before
