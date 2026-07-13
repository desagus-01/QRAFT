from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from qraft.core.cadence import infer_periods_per_year, resolve_periods_per_year


def test_daily_cadence_infers_trading_periods_per_year() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]

    assert infer_periods_per_year(dates) == 252.0


def test_daily_cadence_accepts_configured_trading_periods_per_year() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]

    assert resolve_periods_per_year(dates, 252.0) == 252.0


def test_weekly_cadence_still_uses_calendar_spacing() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=7 * i) for i in range(10)]

    assert infer_periods_per_year(dates) == pytest.approx(365.25 / 7.0)
