"""Independent dataset delivery. Training transforms belong to the consumer."""

from .cache import CacheQuota, ShardCache
from .client import DataServiceClient, RankedDataServiceClient, ShardLease
from .config import DataSourceConfig, DataTransportConfig
from .manifest import DatasetManifest, DatasetSourceIdentity, ShardRecord
from .modelscope import ModelScopeDatasetTransport
from .providers import DatasetProvider, LocalDatasetProvider
from .service import DataServiceLimits, DataServiceServer, DataSupplyService
from .service_protocol import DataServiceSessionIdentity, ShardLeaseDescriptor

__all__ = [
    "CacheQuota",
    "DataServiceClient",
    "DataServiceLimits",
    "DataServiceServer",
    "DataServiceSessionIdentity",
    "DataSourceConfig",
    "DataSupplyService",
    "DataTransportConfig",
    "DatasetManifest",
    "DatasetProvider",
    "DatasetSourceIdentity",
    "LocalDatasetProvider",
    "ModelScopeDatasetTransport",
    "RankedDataServiceClient",
    "ShardCache",
    "ShardLease",
    "ShardLeaseDescriptor",
    "ShardRecord",
]
