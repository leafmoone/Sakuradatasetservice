# Sakuradatasetservice

Independent, pluggable shard delivery extracted from SakuraMoon. Import as
`sakuradatasetservice`. Python 3.11+, Linux, same-host Unix sockets. The only runtime
package dependency is Pydantic; no Torch, model or training imports.

This library owns inventory, byte-size verification, local caching and shard leases.
The consumer owns tar/sample decoding, captions, tokenizers, image augmentation,
collation, training and distributed coordination. SakuraMoon is not modified.

## Install and test

```sh
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
pyright
python -m build --wheel
```

No public package publication or license grant is implied. This private repository's
license is unspecified.

## Minimal command line

```sh
sakura-dataset-service manifest --local-root my-shards --repo-id local/my-corpus --revision v1 --output manifest.json
sakura-dataset-service serve --local-root my-shards --manifest manifest.json --cache-root cache-v1 --state-dir service-state --cache-high-bytes 4294967296
```

`python -m sakuradatasetservice` is equivalent. Replace `--local-root` with
`--modelscope` for the included remote provider. `serve` defaults to once mode;
`--repeat` enables cyclic supply, `--workers` sets consumer topology and repeated
`--exclude` flags omit explicit shard paths. Stop the service with SIGINT/SIGTERM.
The service stays available after once-mode completion so clients can read completion
and retry recent ACKs. Manifest creation refuses to overwrite an existing file.

## Provider boundary

A `DatasetProvider` supplies just two operations:

- `fetch_manifest(source) -> DatasetManifest`
- `fetch_shard(manifest, shard_path, cache_root, *, progress, cancelled) -> FetchedShard`

Included providers are `LocalDatasetProvider` and `ModelScopeDatasetTransport`.
There is no plugin registry or dependency on a specific training framework.
`DatasetSourceIdentity(repo_id="owner/name", revision="v1")` is the logical namespace,
also used for local sources; it does not make local data a ModelScope repository.

```python
from pathlib import Path
from sakuradatasetservice import (
    CacheQuota, DatasetSourceIdentity, LocalDatasetProvider, ShardCache,
)

provider = LocalDatasetProvider(Path("my-shards"))
manifest = provider.fetch_manifest(
    DatasetSourceIdentity(repo_id="local/my-corpus", revision="v1")
)
cache = ShardCache(Path("cache-v1"), manifest, provider,
                   CacheQuota(low_bytes=2 * 1024**3, high_bytes=4 * 1024**3))
result = cache.fetch(manifest.shards[0].path)
print(result.fetched.path)
```

For ModelScope, construct `ModelScopeDatasetTransport` with `DataTransportConfig`.
`from_token_environment("MODELSCOPE_API_TOKEN", policy)` reads the existing process
environment; never put tokens in source or manifests. Manifest helpers support local
snapshots and explicit initialization. Existing snapshots are not silently refreshed.
Remote network/authentication behavior must be tested against your own permitted account.

Custom providers are trusted code. They must publish the expected path and exact size,
honor cancellation, avoid following untrusted paths, and stay within a two-shard-size
peak transfer footprint. The included transfer helper handles byte ranges, resumption
and atomic publication. Size verification detects truncation, not same-size corruption.

## Service and client

Keep the cache directory dedicated to one exact manifest. Its readable ownership marker
rejects a different source, revision or inventory, even when shard names overlap. A
nonempty unmarked directory is rejected; use a fresh directory. Queue and ownership-lock
files belong outside the cache. Use one service and one cache owner per cache directory.

```python
import threading
from pathlib import Path
from sakuradatasetservice import (
    DataServiceLimits, DataServiceServer, DataServiceSessionIdentity, DataSupplyService,
)

state = Path("service-state").resolve()
service = DataSupplyService(
    manifest, cache, state / "queue.json", state / "owner.lock",
    DataServiceSessionIdentity(manifest.dataset_id, worker_count=2),
    DataServiceLimits(download_concurrency=2, verified_shard_lookahead=2,
                      lease_channel_capacity=2, ack_channel_capacity=2),
    excluded_paths=frozenset(),  # optional, e.g. validation shards
    repeat=False,
)
server = DataServiceServer(service, state / "data.sock", request_timeout_seconds=30.0)
stop = threading.Event()
# Run in a dedicated process or thread. Set stop to shut down cleanly.
server.serve(stop)
```

In another same-host process, after the service is ready:

```python
import threading
from pathlib import Path
from sakuradatasetservice import DataServiceClient

state = Path("service-state").resolve()
stop = threading.Event()  # set from the application when cancellation is needed
client = DataServiceClient(state / "data.sock", worker_count=2,
                           request_timeout_seconds=30.0)
for lease in client.iter_leases(worker_id=0, cancelled=stop.is_set):
    with lease as shard:
        with shard.local_path.open("rb") as reader:
            consume(reader)  # the application's decoder/processing function
```

Close the reader before leaving the lease context. Successful exit ACKs the lease;
an exception does not. Explicit `client.lease(worker_id)` plus
`client.acknowledge(descriptor)` is also supported. `acquire()` returns a `ShardLease`
context. Breaking an iterator does not ACK unfinished work. An unACKed lease is returned
again to that worker; there is no automatic lease timeout or abandon/skip operation.

Each concurrent consumer needs a distinct worker ID. `RankedDataServiceClient` maps
rank-local IDs to the global worker topology, without importing a distributed runtime.
The socket and shard paths must be on the same host and accessible to the same trusted
user. Socket permissions are 0600. This is not a cross-host/network service.

## Once, repeat and failure semantics

- `repeat=False` is the default. Each non-excluded shard is allocated once in the
  persisted queue. `health()` returns true only when all work is ACKed.
- `repeat=True` supplies available shards cyclically. An actively leased path is never
  reissued to another worker. A free worker may start another allocation cycle while a
  slower worker still holds an earlier lease. Held paths remain pinned until ACK.
  These are allocation cycles, **not strict corpus epochs or completed-epoch counts**.
  After ACK, a held path becomes eligible again.
- `lease()`/`acquire()` returning `None` can mean temporarily no available path.
  Check `health()` to distinguish completion. `iter_leases()` polls at a positive,
  configurable interval until globally done or cancelled; repeat mode does not end
  on a temporary shortage. If all paths are held, consumers must keep processing and
  ACKing them. The library does not promise universal DDP deadlock freedom.
- Successful ACK retries are idempotent within a bounded 16,384-entry, process-local
  window. Older or mismatched ACKs are rejected. Lost responses can therefore be retried,
  but ACKs are not durably deduplicated across service restarts.
- Restart creates a fresh session. Persisted current-cycle active rows become pending;
  completed rows remain completed. Old clients/leases are rejected and must reconnect.
  Older-cycle outstanding leases are not durably journaled; repeat mode supplies those
  paths again in later allocation cycles. There is no exactly-once processing guarantee.
- State binds the exact manifest and excluded path set. Changing either requires a new
  state file and dedicated cache. A fully consumed once queue stays consumed on restart.
- Cache quotas include shard data, ranges, partials and assembly files. Admission reserves
  up to **2× each downloading shard's size**, not just its final size, with concurrent
  reservations accounted once. Ownership metadata and filesystem allocation overhead are
  excluded. Leave real disk headroom. Protected leases cannot be evicted; an insufficient
  quota fails explicitly rather than deleting active data.
- On consumer failure, retry that worker's outstanding lease or restart the service to
  requeue current work. On transient download failure, resumable bytes are retained. On
  shutdown, downloads are cancelled and the socket/ownership lock are released.

The `validation` helpers remain optional. No fixed validation shard count or training
transform policy is imposed by the service.

## Verification scope

Core tests cover provider boundaries, exact cache ownership, concurrent transfer peaks,
lease/readiness/ACK behavior, restart state, real tiny tar consumption through JSON
client/server dispatch, and local CLI inventory creation. Wheel installation/import is
checked in a fresh environment without Torch installed.

Actual Unix-socket acceptance is **not yet verified** in the current execution environment:
its OS socket creation returns EPERM, including the permitted elevated test run. The
three real socket integration cases are explicitly skipped with a BLOCKED reason there;
in-process dispatch tests are separate and do not substitute for socket coverage.
Run the suite on the intended Linux host before production use. No live ModelScope
account/download, production-scale throughput or distributed training test is claimed.
