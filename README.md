# Sakuradatasetservice

Independent, pluggable shard delivery extracted from SakuraMoon. Import as
`sakuradatasetservice`. Python 3.11+, Linux, same-host Unix sockets; no Torch,
model, tokenizer, image transform or training dependencies.

The provider boundary is `DatasetProvider.fetch_manifest()` and `fetch_shard()`.
Included implementations are `ModelScopeDatasetTransport` and `LocalDatasetProvider`.
Keep each corpus/revision in its own cache directory. Consumers own sample decoding,
caption/tokenizer/image augmentation, batching and distributed training.

Development: `python -m pip install -e '.[dev]'`, then `python -m pytest`.
This initial extraction establishes package boundaries. Lease lifecycle and quota
regressions are being hardened before production use. No public package publication
or license grant is implied; the repository is private and its license is unspecified.
