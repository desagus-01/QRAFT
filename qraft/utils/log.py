from __future__ import annotations

import logging
import logging.handlers
import sys
from collections.abc import Mapping
from typing import Any

from qraft.utils.log_config import DEFAULT_LOG_CONFIG, LogConfig

# Third-party loggers known to be chatty during model fitting.
_NOISY_LOGGERS = (
    "arch",
    "statsmodels",
    "matplotlib",
    "cvxpy",
)

_INCLUDE_CONTEXT = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    /,
    **context: Any,
) -> None:
    if not logger.isEnabledFor(level):
        return

    extra = {"qraft_event": event, "qraft_context": context}
    if context and _INCLUDE_CONTEXT:
        suffix = " ".join(f"{key}={value}" for key, value in context.items())
        logger.log(level, "%s | %s", message, suffix, extra=extra)
        return

    logger.log(level, "%s", message, extra=extra)


def debug_event(
    logger: logging.Logger,
    event: str,
    message: str,
    /,
    **context: Any,
) -> None:
    log_event(logger, logging.DEBUG, event, message, **context)


def info_event(
    logger: logging.Logger,
    event: str,
    message: str,
    /,
    **context: Any,
) -> None:
    log_event(logger, logging.INFO, event, message, **context)


def warning_event(
    logger: logging.Logger,
    event: str,
    message: str,
    /,
    **context: Any,
) -> None:
    log_event(logger, logging.WARNING, event, message, **context)


def error_event(
    logger: logging.Logger,
    event: str,
    message: str,
    /,
    **context: Any,
) -> None:
    log_event(logger, logging.ERROR, event, message, **context)


def setup_logging(cfg: LogConfig | None = None) -> None:
    global _INCLUDE_CONTEXT

    if cfg is None:
        cfg = DEFAULT_LOG_CONFIG
    _INCLUDE_CONTEXT = cfg.include_context

    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(cfg.level)

    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(cfg.console_level or cfg.level)
    console_handler.setFormatter(logging.Formatter(cfg.fmt))
    root.addHandler(console_handler)

    if cfg.log_file is not None:
        file_fmt = "%(asctime)s " + cfg.fmt
        file_handler = logging.handlers.RotatingFileHandler(
            cfg.log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(cfg.file_level or cfg.level)
        file_handler.setFormatter(logging.Formatter(file_fmt))
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(cfg.third_party_level)


def event_context(record: logging.LogRecord) -> Mapping[str, Any]:
    return getattr(record, "qraft_context", {})
