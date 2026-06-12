from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogConfig:
    level: int = logging.INFO
    log_file: str | None = None
    third_party_level: int = logging.WARNING
    fmt: str = "[%(levelname)s] %(name)s - %(message)s"


DEFAULT_LOG_CONFIG: LogConfig = LogConfig()
