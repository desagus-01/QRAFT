from __future__ import annotations

import logging
import logging.handlers
import sys

from qraft.forecast.config import LogConfig

# Third-party loggers known to be chatty during model fitting.
_NOISY_LOGGERS = (
    "arch",
    "statsmodels",
    "matplotlib",
    "cvxpy",
)


def setup_logging(cfg: LogConfig | None = None) -> None:
    """Configure the root logger from a ``LogConfig``.

    Safe to call multiple times -- existing handlers are cleared first so
    repeated calls in notebooks do not duplicate output.

    Parameters
    ----------
    cfg :
        Configuration object. If ``None``, :data:`policy.DEFAULT_LOG_CONFIG`
        is used.
    """
    if cfg is None:
        from qraft.forecast.config import DEFAULT_LOG_CONFIG

        cfg = DEFAULT_LOG_CONFIG

    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(cfg.level)

    # Clear handlers added by previous calls or rogue basicConfig() calls
    # that fired at import time before setup_logging() was invoked.
    root.handlers.clear()

    # Console handler clean format, no timestamps, for interactive use.
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
