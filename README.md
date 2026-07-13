# QRAFT

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

[**Add example here**]

Users are also able to add view events for specific date windows. For example, if users have different views for general risk on/off regimes, they can specify those views for a date window which will then feed into a backtest/validation module. Please see [**CREATE EXAMPLE AND WRITE HERE**] for how this could be done.


### Forecasting

QRAFT uses a derivative of filtered historical simulation (FHS) for forecasting assets' future paths. Instead of the standard FHS, we instead, broadly follow Meucci's 'The Prayer' [**INSERT CITATION HERE**] tto derive this. This starts by finding a suitable **univariate** model that makes the asset's innovation strongly IID. The forecast stack in QRAFT follows a general Box-Jenkins methodology (ie remove determinism -> model mean -> model vol) with checks in between to determine the necessity of each. Once the forecast stack determines a suitable model for each asset, a `ForecastRecipe` is created for a specific date, users can determine how often they wish to re-run the forecast stack to create new recipes through history.

[**CREATE EXAMPLE HERE**]

Following the univariate structural models from `ForecastRecipes`, we are able to run a **forecast simulation** of assets' returns taking into account their cross-asset depedencies from their joint distribution of IID residuals. The simulation is done by **bootstraping** our innovations using the scenario probability (which can reflect our current views of the market).  Users can also opt to use the **Copula-Marginal Algorithm** [**PUT CITATION FOR MEUCCIS WORK**] for bootstrapping instead of just using the non-parametric method, this allows users to specify either/or the marginal of assets and/or the copula.


[**ADD IMAGE EXAMPLE HERE**]

### Policies

Because every scenario object carries an explicit probability vector, the full simulation output is always available for downstream risk and allocation steps nothing is collapsed to a point estimate prematurely. This also makes it straightforward to encode  **views** directly onto the scenario distribution using **Entropy Pooling**. Rather than discarding scenarios, their probabilities are updated to be consistent with the view (e.g. "AAPL mean return will be below historical average") by solving a minimum KL-divergence problem. Views can be placed on means, volatilities, correlations, or arbitrary moments.

Portfolio-level risk is computed from the simulated loss distribution, supporting both **VaR** and **CVaR** (more coming soon!). Risk attribution is available at the factor level, using top-down exposure estimation, minimum-torsion orthogonalisation, and Euler decomposition to attribute marginal and total risk contributions to each factor.

### Policy Evaluation



### Risk Management

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance, including the docstring style decision record.

---
