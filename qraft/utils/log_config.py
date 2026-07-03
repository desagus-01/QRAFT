from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogConfig:
    level: int = logging.INFO
    console_level: int | None = None
    file_level: int | None = None
    log_file: str | None = None
    third_party_level: int = logging.WARNING
    fmt: str = "[%(levelname)s] %(name)s - %(message)s"
    include_context: bool = True


DEFAULT_LOG_CONFIG: LogConfig = LogConfig()
