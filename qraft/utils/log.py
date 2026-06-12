from __future__ import annotations

import logging
import logging.handlers
import sys

from qraft.utils.log_config import DEFAULT_LOG_CONFIG, LogConfig

# Third-party loggers known to be chatty during model fitting.
_NOISY_LOGGERS = (
    "arch",
    "statsmodels",
    "matplotlib",
    "cvxpy",
)


def setup_logging(cfg: LogConfig | None = None) -> None:
    if cfg is None:
        cfg = DEFAULT_LOG_CONFIG

    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(cfg.level)

    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(cfg.level)
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
        file_handler.setLevel(cfg.level)
        file_handler.setFormatter(logging.Formatter(file_fmt))
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(cfg.third_party_level)
