"""Small independent transport configuration; no training configuration imports."""

from __future__ import annotations

import os
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints


class ConfigurationError(ValueError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DataSourceConfig(StrictModel):
    repo_id: Annotated[str, StringConstraints(min_length=1)]
    revision: Annotated[str, StringConstraints(min_length=1)]


class DataTransportConfig(StrictModel):
    connect_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    read_timeout_seconds: Annotated[float, Field(gt=0)] = 60.0
    max_retries: Annotated[int, Field(ge=0)] = 3
    retry_backoff_seconds: Annotated[float, Field(ge=0)] = 1.0
    stream_chunk_bytes: Annotated[int, Field(ge=65536, le=16777216)] = 1048576
    streams_per_shard: Annotated[int, Field(gt=0)] = 1


def resolve_secret(name: str) -> SecretStr:
    value = os.environ.get(name)
    if not value:
        raise ConfigurationError("required token environment variable is absent")
    return SecretStr(value)
