from statsmodels.stats.multitest import multipletests


def multiple_tests_rejected(
    p_values: list[float], significance_level: float = 0.05
) -> list[bool]:
    """
    Adjusts p-values for multiple tests then compares to alpha to determine rejection.

    The ``significance_level`` default of 0.05 mirrors ``IIDConfig.significance_level``.
    Prefer passing ``cfg.significance_level`` explicitly from a config object in new code.
    """
    return multipletests(p_values, alpha=significance_level, method="holm-sidak")[
        0
    ].tolist()
