from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

from qraft.utils.log import info_event, log_event, setup_logging
from qraft.utils.log_config import DEFAULT_LOG_CONFIG, LogConfig


def test_log_event_adds_event_and_context(caplog) -> None:
    logger = logging.getLogger("qraft.tests.logging")

    with caplog.at_level(logging.INFO, logger="qraft.tests.logging"):
        info_event(
            logger,
            "test.event",
            "Test event happened",
            asset="AAPL",
            count=2,
        )

    record = caplog.records[0]
    assert record.qraft_event == "test.event"
    assert record.qraft_context == {"asset": "AAPL", "count": 2}
    assert record.message == "Test event happened | asset=AAPL count=2"


def test_log_event_respects_level(caplog) -> None:
    logger = logging.getLogger("qraft.tests.logging.filtered")

    with caplog.at_level(logging.INFO, logger="qraft.tests.logging.filtered"):
        log_event(logger, logging.DEBUG, "test.debug", "Hidden event")

    assert not caplog.records


def test_setup_logging_can_hide_context() -> None:
    setup_logging(LogConfig(include_context=False))
    logger = logging.getLogger("qraft.tests.logging.no_context")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    try:
        info_event(logger, "test.event", "Test event happened", asset="AAPL")
        assert stream.getvalue().strip() == "Test event happened"
    finally:
        logger.removeHandler(handler)
        setup_logging(DEFAULT_LOG_CONFIG)


def test_setup_logging_supports_file_level(tmp_path: Path) -> None:
    log_file = tmp_path / "qraft.log"
    setup_logging(
        LogConfig(
            level=logging.DEBUG,
            console_level=logging.WARNING,
            file_level=logging.DEBUG,
            log_file=str(log_file),
        )
    )
    logger = logging.getLogger("qraft.tests.logging.file")

    info_event(logger, "test.file", "File event", count=1)

    assert "File event | count=1" in log_file.read_text()
    setup_logging(DEFAULT_LOG_CONFIG)
