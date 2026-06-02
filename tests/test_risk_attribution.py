import numpy as np
import polars as pl

from forecast.scenarios.panel import ScenarioPanel
from risk.measures import var
from risk.risk_attribution import get_var_data


def test_get_var_data_filters_by_measures_var_cutoff() -> None:
    panel = ScenarioPanel(
        values=pl.DataFrame(
            {
                "factor": [1.0, 2.0, 3.0, 4.0],
                "loss": [10.0, 20.0, 30.0, 40.0],
            }
        ),
        dates=None,
        prob=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    cutoff = float(
        var(
            panel.values["loss"].to_numpy(),
            prob=panel.prob,
            method="empirical",
            alpha=0.25,
            distribution_type="loss",
        )
    )

    tail = get_var_data(panel, alpha=0.25)

    assert cutoff == 40.0
    assert tail["loss"].to_list() == [40.0]
    assert tail["prob"].to_list() == [0.4]
