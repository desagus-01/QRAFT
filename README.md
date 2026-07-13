# QRAFT

> [!WARNING] This package is still in alpha so please expect breaking changes.

**Quantitative Risk, Allocation, and Forecasting Toolkit**

QRAFT is an advanced end-to-end portfolio construction toolkit aimed at p-style quants who wish to emphasize scenarios based on their market views throughout the allocation process.

The package is heavily inspired by the teachings of Atillio Meucci for the 'core' of views using entropy pooling and Stephen Boyd and the cvxportfolio library for policy creation and evaluation.

> **Disclaimer:** Research and educational use only. Not financial or investment advice.

---

## What QRAFT Does

There are 4 core pillars for this project:
1. Views
2. Forecaster
3. Policy
4. Risk Management

I'll briefly explain each below (I will make docs at one point with more detail on each!).

---

### Views

Probably the biggest differentation between this project and other portfolio construction packages out there. 'Views' refers to the ability of the investor to add their **market** views to the whole investment process explicity and in a formal manner. This is done through entropy pooling, which is formalized by Atillio Meucci in his paper regarding 'Fully Flexible Probabilities' [**I NEED TO CITE THIS**]. In short, the package allows investors to input their future market expectations (for example `VIX <= 15`), infer an updated discrete `scenario probability` from it, and simulate futures deriving from these views.

```python
latest_view = ViewWindow(
    views=Views(
        [
            MeanView("VIX", "<=", 15),
            RankingView(["TLT", "GLD", "SPY"]),
        ],
        confidence=1.0,
    ),
    name="latest_high_vix_state",
)

market = market_without_views.with_views([latest_view])
```

Users are also able to add view events for specific date windows. For example, if users have different views for general risk on/off regimes, they can specify those views for a date window which will then feed into a backtest/validation module. Please see [**CREATE EXAMPLE AND WRITE HERE**] for how this could be done.


### Forecasting

QRAFT uses a derivative of filtered historical simulation (FHS) for forecasting assets' future paths. Instead of the standard FHS, we instead, broadly follow Meucci's 'The Prayer' [**INSERT CITATION HERE**] tto derive this. This starts by finding a suitable **univariate** model that makes the asset's innovation strongly IID. The forecast stack in QRAFT follows a general Box-Jenkins methodology (ie remove determinism -> model mean -> model vol) with checks in between to determine the necessity of each. Once the forecast stack determines a suitable model for each asset, a `ForecastRecipe` is created for a specific date, users can determine how often they wish to re-run the forecast stack to create new recipes through history.


Following the univariate structural models from `ForecastRecipes`, we are able to run a **forecast simulation** of assets' returns taking into account their cross-asset depedencies from their joint distribution of IID residuals. The simulation is done by **bootstraping** our innovations using the scenario probability (which can reflect our current views of the market).  Users can also opt to use the **Copula-Marginal Algorithm** [**PUT CITATION FOR MEUCCIS WORK**] for bootstrapping instead of just using the non-parametric method, this allows users to specify either/or the marginal of assets and/or the copula.


[**ADD IMAGE EXAMPLE HERE**]

### Policies

This module is heavily inspired by the existing `cvxportfolio` python package, and more in general in the accompanying paper related to it [**cite paper here**]. In short, the core problem we solve here is multi period and **MUST** be convex. In this instance, only the first horizon is actionable, future ones are diagnostic/planning information.

The core of this module is to allow the user to craft their own multi-period optimization problem by using the `MPOProblem` and the `MPOProblemBuilder` objects, or alternatives using one of the preset ones. 

[**ADD EXAMPLE HERE**]

This module is then responsible for taking our previous simulated forecasts and the current portfolio, applying a policy and producing the next portfolio allocation.

**MAYBE ADD LIST OF OBJS AND CONSTRAINTS?**

### Policy Evaluation/Backtest

Following our policy creation and forecast recipes, we are able to simulate how our strategy would have behaved historically. The backtest methodology is currently able to handle the different view events historically, as per below.

[**ADD EXAMPLE HERE**]

Rebalanes currently follow a **time cadence**, based on the rebalance date, it seems the `ForecastRecipe`, `View`, `PortfolioState`, and `Policy` at the time to derive the optimal allocation. Once policies make decisions, the backtest will be responsible for simulating trading, taking into account costs, cash, and shares.

[**should I add some simple examples here?**]

Finally, users can tune their policies as well. Much of this is based from De Prado's paper [**cite here**]. Users can either use the standard **Walk-Forward Validation** or **Combinatorial Purged Cross-Validation** to tune the policy, both using out-of-sample selection.

[**Add simple example here**]

### Risk Management

Taking into account our portfolio projection based on policy decision and existing forecasts, we are able to analyze where risk comes from. Inputs are inheriently probabilistic because of this.

This module provides traditional risk measures such as VaR, CVaR, and variance. Also diversification measures such as effective number of bets. Users are also incentived to use factors to break their both performance and risk attribution and contribution.

[**Example below**]

**In addition to the examples displayed above please look into the examples directory for further examples**

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance, including the docstring style decision record.

---

