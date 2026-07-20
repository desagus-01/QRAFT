# QRAFT

> [!WARNING]
> This package is still in alpha so please expect breaking changes.
> Research and educational use only. Not financial or investment advice.

**Quantitative Risk, Allocation, and Forecasting Toolkit**

QRAFT is an advanced end-to-end portfolio construction toolkit aimed at practitioners and quants who wish to emphasize scenarios based on their market views throughout the allocation process.

The package is heavily inspired by the teachings of Attilio Meucci for the 'core' of views using entropy pooling and Stephen Boyd and the cvxportfolio library for policy creation and evaluation.

## What QRAFT Does

There are 5 core pillars for this project:

1. Views
2. Forecasting
3. Policies
4. Policy Evaluation/Backtest
5. Risk Management

I'll briefly explain each below (I will make docs at one point with more detail on each!).

Please have a look at the `examples/` directory for clear use cases (but also read the below so that they make sense!). A bit of setup is repeated in the examples below so that each section can be read on its own.

For a first pass through the whole workflow, start with `examples/quickstart.py`. It builds a synthetic market, adds a view, runs forecasts, creates a preset MPO allocation, looks at risk, and then runs a simple backtest with allocation plots.

### Data

At the moment, QRAFT is mainly built around clean daily price data. In practice this means a table with a `date` column, columns for the assets you can trade, and optionally columns for extra market/factor series you want to use for views, forecasting, or risk attribution.

The price data should be strictly positive and should not contain NaNs or infinite values for the assets/factors you include in the universe. Users define what is tradable and what is a factor through `AssetUniverse`, and wrap the data into `MarketData` using `MarketData.from_prices(...)`.

After this point, most of the package works from the same `MarketData` object: views are attached with `market.with_views(...)`, forecasts are run with `Forecaster`, policies use it through `Allocation`, and backtests use it through `Backtest`.

For examples, `qraft.utils.example_data.synthetic_market_example()` creates a small made-up market with synthetic assets (`NOVA`, `ORION`, `TERRA`, `LUMEN`, `CYPHER`) and synthetic factors (`STRESS_INDEX`, `GROWTH_PULSE`, `RATE_WAVE`).

---

### Views

Probably the biggest differentiation between this project and other portfolio construction packages out there. 'Views' refers to the ability of the investor to add their **market** views to the whole investment process explicitly and in a formal manner. This is done through entropy pooling, formalized by Attilio Meucci in "Fully Flexible Views: Theory and Practice" (Meucci, A., 2008, Risk, 21(10), 97-102; SSRN 1213325) and the companion "Historical Scenarios with Fully Flexible Probabilities" (Meucci, A., 2010, GARP Risk Professional; SSRN 1696802). In short, the package allows investors to input their future market expectations, infer updated discrete `scenario probabilities` from them, and simulate futures deriving from these views.

The main public object is `Views`, which combines one or more view specifications such as `MeanView`, `StdView`, `CorrView`, `RankingView`, or `QuantileView`. A `ViewWindow` makes those views active over a specific date range, and `MarketData.with_views(...)` attaches the windows to the market. Users can then inspect the posterior distribution with `market.viewed_returns(...)`, review entropy-pooling diagnostics with `market.view_report(...)`, or plot the probability movement with `market.plot_view(...)`.

```python
import numpy as np

from qraft import MeanView, RankingView, ViewWindow, Views
from qraft.utils.example_data import synthetic_market_example

market_without_views = synthetic_market_example()
as_of = market_without_views.trading_bars[-1]
returns = market_without_views.returns_through(as_of)
high_stress_move = float(np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80))

latest_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("STRESS_INDEX", ">=", high_stress_move),
            RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
        ],
        confidence=1.0,
    ),
    name="latest_high_stress_state",
)

market = market_without_views.with_views([latest_view])
report = market.view_report(as_of)

print(report.diagnostics)
report.plot()
```

Users are also able to add view events for specific date windows. For example, if users have different views for general risk on/off regimes, they can specify those views for historical date windows, attach them to `MarketData`, and then feed this viewed market into forecasting, backtesting, and validation. During a backtest, the active view for each decision date is picked up from the market snapshot, so historical views can directly affect the forecast scenarios and optimizer inputs used by the policy.


### Forecasting

QRAFT uses a variant of filtered historical simulation (FHS) for forecasting assets' future paths. Instead of the standard FHS, we broadly follow Meucci's "The Prayer: Ten-Step Checklist for Advanced Risk and Portfolio Management" (Meucci, A., 2011; SSRN 1753788) to derive this. This starts by finding a suitable **univariate** model that makes the asset's innovation strongly IID. The forecast stack in QRAFT follows a general Box-Jenkins methodology (ie remove determinism -> model mean -> model vol), with checks in between to determine the necessity of each step. Once the forecast stack determines a suitable model for each asset, a `ForecastRecipe` is created for a specific date. Users can determine how often they wish to re-run the forecast stack to create new recipes through history.

In practice, QRAFT first tries to make each asset's history closer to IID, then simulates forward from those cleaned innovations.

The main public object is `Forecaster`, which combines a `PipelineConfig` for model selection and a `SimulationForecastConfig` for path simulation. Users can call `forecaster.recipes(...)` to build a `ForecastRecipeHistory`, or `forecaster.run(...)` to directly produce a `ForecastRun`. Each run contains dated forecast steps, and each step contains `ForecastPaths`, which stores simulated future price paths, path probabilities, diagnostics, and plotting helpers.

Following the univariate structural models from `ForecastRecipe`, we are able to run a **forecast simulation** of assets' returns taking into account their cross-asset dependencies from their joint distribution of IID residuals. The simulation is usually done by **bootstrapping** innovations using the scenario probabilities, which can reflect our current views of the market. Users can also opt to use the **Copula-Marginal Algorithm** (Meucci, A., "A New Breed of Copulas for Risk and Portfolio Management," 2011, Risk, 24(9); SSRN 1752702) through `CMAConfig`, allowing them to specify the marginal distributions, the copula, or both.

```python
from qraft import Forecaster, PipelineConfig, SimulationForecastConfig
from qraft.utils.example_data import synthetic_market_example

market = synthetic_market_example()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=250, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

run = forecaster.run(
    market,
    min_history=180,
    forecast_cadence="quarter_end",
)

print(run.model_health_df())

latest_step = run.steps[-1]
latest_forecast = latest_step.forecast

print(latest_forecast.at_step(1, subset="tradable").values.head())
latest_forecast.plot_asset_paths(subset="tradable", max_assets=5, ncols=3)
```

Forecasting also naturally takes views into account. If `MarketData` has active `ViewWindow` events, the forecast snapshot for that date uses the posterior scenario probabilities from the active view. This means users can express something like a risk-on or risk-off state, attach it to the market, and then simulate forecast paths from the viewed scenario distribution rather than the original prior distribution.

```python
import numpy as np

from qraft import Forecaster, MeanView, PipelineConfig, RankingView
from qraft import RebalanceSchedule, SimulationForecastConfig, ViewWindow, Views
from qraft.utils.example_data import synthetic_market_example

market_without_views = synthetic_market_example()
bars = market_without_views.trading_bars
forecast_dates = [
    d
    for d, _ in RebalanceSchedule("quarter_end").decision_steps(bars)
    if bars.index(d) + 1 >= 180
]
as_of = forecast_dates[-1]
returns = market_without_views.returns_through(as_of)
high_stress_move = float(np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80))

risk_off = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("STRESS_INDEX", ">=", high_stress_move),
            RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
        ],
        confidence=0.75,
    ),
    name="risk_off_latest_forecast",
)

market = market_without_views.with_views([risk_off])

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=21, n_sims=500, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

run = forecaster.run(market, min_history=180, forecast_cadence="quarter_end")
forecast = run.forecast_at(as_of)

print(market.view_report(as_of).diagnostics)
forecast.plot_asset_paths(subset="tradable", max_assets=5, ncols=3)
```

### Policies

This module is heavily inspired by the existing `cvxportfolio` python package, and more in general by the accompanying paper related to it: Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., & Speth, J. (2017). "Multi-Period Trading via Convex Optimization." Foundations and Trends in Optimization, 3(1), 1-76 (arXiv:1705.00109). In short, the core problem we solve here is multi-period and **MUST** be convex. In this instance, only the first horizon is actionable; future ones are diagnostic/planning information.

The main public objects are `Allocation`, `EqualWeightPolicy`, and `MPOPolicy`. `Allocation` is the facade that takes a `MarketData` object, a policy, forecasts, and an optional `PortfolioState`, then produces a point-in-time `PolicyRun`. The `PolicyRun` contains the actual `PolicyDecision`, target weights, forecast projection, optimizer inputs, and diagnostics. Users can start simple with `EqualWeightPolicy`, use `MPOPolicy.preset(...)` for common multi-period optimization objectives, or build a custom convex problem with `MPOProblem` / `MPOProblemBuilder`.

The optimization layer is controlled by objects such as `InputPlan`, which tells QRAFT how forecasts become optimizer inputs, and constraints such as `LongOnly`, `MinCashWeight`, `MaxWeight`, and `TurnoverLimit`. Objective terms can be preset, or explicitly built from pieces like `ExpectedReturn`, `CovarianceRisk`, `CVaRRisk`, `CashReturn`, `TransactionCost`, and `HoldingCost`.

```python
from qraft import Allocation, Forecaster, MPOPolicy
from qraft import PipelineConfig, SimulationForecastConfig
from qraft.construction import LongOnly, MaxWeight, MinCashWeight, TurnoverLimit
from qraft.construction.optimization import InputPlan
from qraft.utils.example_data import synthetic_market_example

market = synthetic_market_example()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=500, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=5.0,
    constraints=(
        LongOnly(),
        MinCashWeight(0.05),
        MaxWeight(0.60),
        TurnoverLimit(0.25),
    ),
    input_plan=InputPlan(expected_returns="forecast", max_horizons=10),
    min_history=180,
    name="mean_covariance_example",
)

run = Allocation(
    market=market,
    policy=policy,
    forecasts=forecaster,
    initial_cash=100_000,
).at()

print(run.target_weights)
print({"cash": run.decision.target_cash_weight})
print(run.plan_metrics())

result = run.mpo_result()
print({"solver_status": result.status, "turnover": result.turnover})

run.plot_weights()
run.projection.plot()
```

This module is then responsible for taking our previous simulated forecasts and the current portfolio, applying a policy and producing the next portfolio allocation. The policy itself only decides the target allocation; the projection lets users inspect how that decision behaves across the simulated forecast paths before plugging the same policy into a backtest.

### Policy Evaluation/Backtest

Following our policy creation and forecast recipes, we are able to simulate how our strategy would have behaved historically. `Backtest` takes a `MarketData` object, a policy, optional forecasts, and a `BacktestConfig`, then returns a `BacktestResult` containing NAV, trades, cash, costs, warnings, diagnostics, and summary metrics.

```python
from qraft import Backtest, BacktestConfig, EqualWeightPolicy, RebalanceSchedule
from qraft.utils.example_data import synthetic_market_example

market = synthetic_market_example()
policy = EqualWeightPolicy(target_cash_weight=0.05)

result = Backtest(
    market=market,
    policy=policy,
    config=BacktestConfig(
        schedule=RebalanceSchedule("month_end"),
        initial_cash=100_000,
    ),
).run()

print(result.summary_df())
print(result.period_turnovers[:5])
result.plot_nav()
result.plot_drawdown()
```

Rebalances follow a **time cadence** through `RebalanceSchedule`. At each decision date, the backtest builds a current market snapshot and `PortfolioState`, asks the `Policy` for the next allocation, then executes the decision on the next available trading bar. The engine simulates trading through time, including shares, cash, transaction costs, holding costs, NAV, solver status, and rebalance-level diagnostics.

Views are taken into account through the `MarketData` passed into the backtest. If the market has dated `ViewWindow` events, each decision snapshot uses the active view for that date. This means forecast scenarios and optimizer inputs can reflect the user's historical market views, and the result can be inspected with `view_activity_df()`.

```python
import numpy as np

from qraft import Backtest, BacktestConfig, Forecaster, MeanView, MPOPolicy
from qraft import PipelineConfig, RebalanceSchedule, SimulationForecastConfig
from qraft import ViewWindow, Views
from qraft.construction import LongOnly, MinCashWeight, TurnoverLimit
from qraft.construction.optimization import InputPlan
from qraft.utils.example_data import synthetic_market_example

market_without_views = synthetic_market_example()
bars = market_without_views.trading_bars
returns = market_without_views.returns_through(bars[120])
high_stress_move = float(np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80))

risk_off = ViewWindow(
    start=bars[120],
    end=bars[180],
    views=Views([MeanView("STRESS_INDEX", ">=", high_stress_move)], confidence=0.75),
    name="risk_off",
)

market = market_without_views.with_views([risk_off])

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=5, n_sims=1_000, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=4.0,
    constraints=(LongOnly(), MinCashWeight(0.05), TurnoverLimit(0.25)),
    input_plan=InputPlan(expected_returns="forecast", max_horizons=5),
    min_history=30,
)

result = Backtest(
    market=market,
    policy=policy,
    forecasts=forecaster,
    config=BacktestConfig(schedule=RebalanceSchedule("month_end")),
).run()

print(result.view_activity_df())
```

Finally, users can tune their policies as well. Much of this is based on Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Wiley (Ch. 7 -- cross-validation; Ch. 11-12 -- CPCV). Overfitting diagnostics follow Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio," Journal of Portfolio Management, 40(5), and Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting," Journal of Computational Finance, 20(4). Users can either use the standard **Walk-Forward Validation** or **Combinatorial Purged Cross-Validation** to tune the policy, both using out-of-sample selection.

```python
from qraft import BacktestConfig, RebalanceSchedule, Validation, WalkForwardConfig

validation = Validation(
    market=market,
    base_policy=policy,
    grid={"risk_aversion": [2.0, 4.0, 8.0]},
    forecasts=forecaster,
    backtest_config=BacktestConfig(schedule=RebalanceSchedule("month_end")),
)

walk = validation.walk_forward(
    WalkForwardConfig(train_size=8, test_size=3, metric="sharpe")
)
tuned = validation.tune(walk)

print(walk.summary_df())
print(tuned.selected_params)
print(tuned.backtest.summary_df())
```

### Risk Management

Taking into account our portfolio projection based on the policy decision and existing forecasts, we are able to analyze where risk comes from. Inputs are inherently probabilistic because of this. We are not just looking at one expected return, but at the distribution of outcomes coming from the simulated forecast paths.

The main public object here is `PortfolioRisk`, which users will usually get from a `PolicyRun` by calling `run.risk(...)`, or directly from `Allocation.risk(...)`. This gives users the standard risk measures such as `VaR` and `CVaR`, but also breaks risk down into factor contributions where factors are available in the market universe. Users can inspect the output using `summary_df(...)`, `risk_contribution(...)`, `risk_at_horizon(...)`, and `effective_bets(...)`. The effective-bets and minimum-torsion math follows Meucci, A., Santangelo, A., & Deguest, R. (2015). "Risk Budgeting and Diversification Based on Optimized Uncorrelated Factors," Risk, 29(11) (SSRN 2276632).

```python
risk = run.risk()

print(risk.summary_df(alpha=0.05))
print(risk.risk_contribution("cvar", alpha=0.05))
print(risk.risk_at_horizon("var", alpha=0.05))
```

The factor attribution side is important here. If the forecast contains factors, QRAFT tries to explain the projected portfolio risk using those factor paths plus an idiosyncratic component. This lets users see whether their risk is mainly coming from market/factor exposure, specific state variables, or residual unexplained risk.

```python
cvar_contrib = risk.risk_contribution("cvar", alpha=0.05)

print(cvar_contrib.contributions)
cvar_contrib.plot()

effective_bets = risk.effective_bets()
print(effective_bets.effective_bets)
effective_bets.plot()
```

At a lower level, `qraft.risk` also exposes objects and functions such as `RiskContributions`, `EffectiveBets`, `PortfolioRiskAttribution`, `portfolio_factor_attribution`, `factor_ols_regression`, `var`, `cvar`, and `minimum_torsion_matrix`. Most users should not need to start there, but they are available for more custom attribution workflows.

## Alpha Limitations and Modeling Assumptions

- Backtest decisions use market information through decision bar `t` and execute at the single observed price on the next available trading bar, `t+1`.
- Execution assumes fractional shares. Lot sizes, bid-ask paths, intrabar fills, and partial fills are not modeled.
- Dividends and corporate actions are not applied to market prices or positions. `HoldingCost.dividends` is only an explicit optimizer/backtest income-rate assumption.
- Transaction and holding costs are debited from cash. A fully invested portfolio, or any allocation without enough cash, can fail when costs are charged; use a positive cash target such as `MinCashWeight`.
- `HoldingCost` inputs are annualized decimal rates: `0.00119` means 0.119% annually. QRAFT divides these values by `periods_per_year`; it does not divide them by 100.
- MPO weights are linked across planning horizons by trades without applying forecast-return drift. Only horizon 0 is actionable; later horizons are planning and diagnostic outputs.
- When `periods_per_year` is not configured, daily observations map to 252 and other cadences use `365.25 / median calendar-day spacing`. This annualization heuristic is not an exchange-calendar or trading-day model. Holding-cost accrual uses actual elapsed calendar days, so it can differ from simple per-trading-bar assumptions.
- CMA copula fitting supports `ml` and `itau`; `irho` is not supported. With the pinned `copulae` implementation, two-dimensional panels require `ml` because rank-based `itau` fitting is unsupported.


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance, including the docstring style decision record.

---
