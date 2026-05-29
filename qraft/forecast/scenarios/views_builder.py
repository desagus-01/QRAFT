from dataclasses import dataclass, field
from typing import Self

import numpy as np
import polars as pl
from forecast.scenarios.types import (
    ConstraintSignLike,
    CorrInfo,
    View,
)
from polars import DataFrame
from utils.helpers import indicator_quantile_marginal


@dataclass
class ViewBuilder:
    """
    Holds all possible types of views and helps to concat them.
    """

    data: DataFrame
    views: list[View] = field(default_factory=list)

    def _match_mean_ref(self, views: list[View]) -> list[View]:
        means = {}
        pending_std = {}

        for v in views:
            rd = v.risk_driver

            if v.type == "mean":
                means[rd] = v.views_target
                # attach mean to any std views we saw earlier for this driver
                for sv in pending_std.pop(rd, []):
                    sv.mean_ref = v.views_target

            elif v.type == "std":
                if rd in means:
                    v.mean_ref = means[rd]
                else:
                    pending_std.setdefault(rd, []).append(v)

        return views

    def build(self) -> list[View]:
        """
        Returns all created views
        """
        return self._match_mean_ref(self.views)

    # TODO: Review the below, think math might be wrong
    def quantile(self, quant: float, quant_prob: float) -> Self:
        if quant_prob > quant:
            raise ValueError(
                f"Your target prob of {quant_prob}, must be smaller or equal to your current quant of {quant}!"
            )

        quant_ind = indicator_quantile_marginal(self.data, quant)

        self.views.append(
            View(
                type="quantile",
                risk_driver=self.data.columns[0],
                data=quant_ind.select(pl.col("quant_ind")).to_numpy().T,
                views_target=np.array(quant_prob),
                sign_type="equal_less",
            )
        )

        return self

    def corr(
        self,
        corr_targets: list[CorrInfo],
        sign_type: list[ConstraintSignLike],
    ) -> Self:
        self.views.extend(
            [
                View(
                    type="corr",
                    risk_driver=corr_info.asset_pair,
                    data=self.data[list(corr_info.asset_pair)].to_numpy().T,
                    views_target=np.array(corr_info.corr),
                    sign_type=sign_type[i],
                )
                for i, corr_info in enumerate(corr_targets)
            ]
        )
        return self

    def mean(
        self,
        target_means: dict[str, float],
        sign_type: list[ConstraintSignLike],
    ) -> Self:
        """
        Builds the constraints based on each asset's targeted mean.
        """
        # Checks we have equal amounts of self.data, constraints, and targets
        if not (len(target_means.keys()) == len(sign_type)):
            raise ValueError(
                "All inputs must have the same length as the number of views."
            )

        self.views.extend(
            [
                View(
                    type="mean",
                    risk_driver=key,
                    data=self.data[key].to_numpy().T,
                    views_target=np.array(target_means[key]),
                    sign_type=sign_type[i],
                )
                for i, key in enumerate(target_means.keys())
            ]
        )

        return self

    def std(
        self,
        target_std: dict[str, float],
        sign_type: list[ConstraintSignLike],
    ) -> Self:
        # Checks we have equal amounts of self.data, constraints, and targets
        if not (len(target_std.keys()) == len(sign_type)):
            raise ValueError(
                "All inputs must have the same length as the number of views."
            )
        self.views.extend(
            [
                View(
                    type="std",
                    risk_driver=key,
                    data=self.data[key].to_numpy().T,
                    views_target=np.array(target_std[key]),
                    sign_type=sign_type[i],
                    mean_ref=None,
                )
                for i, key in enumerate(target_std.keys())
            ]
        )
        return self

    def ranking(
        self,
        asset_ranking: list[str],
    ) -> Self:
        """
        Build view based on which assets should have a higher expected return (ie A >= B >= C ...).
        """

        assets_to_iterate = range(len(asset_ranking) - 1)

        comparisons = [
            (asset_ranking[i], asset_ranking[i + 1]) for i in assets_to_iterate
        ]
        self.views.extend(
            [
                View(
                    type="sorting",
                    risk_driver=comparisons[asset],
                    data=self.data[comparisons[asset]].to_numpy().T,
                    views_target=None,
                    sign_type="equal_greater",
                )
                for asset in assets_to_iterate
            ]
        )
        return self

    def marginal(self, current_marginal: str, target_marginal: str) -> Self:
        rd_np = self.data.select(target_marginal).to_numpy()

        # Add a mean view
        _ = self.mean(
            {current_marginal: rd_np.mean()},
            sign_type=["equal"],
        )

        # Add a std view
        _ = self.std(
            {current_marginal: rd_np.std()},
            sign_type=["equal"],
            mean_ref=rd_np.mean(),
        )

        return self
