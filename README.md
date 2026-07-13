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

### Probabilistic Forecasting

QRAFT provides a full probabilistic forecasting pipeline for financial time series. Raw price data is preprocessed per-asset (stationarity checks, detrending, deseasoning), a mean and volatility model is selected automatically, and innovations are extracted and used to drive a Monte Carlo simulation. The output is never a single forecast, it is always a set of probability-weighted simulated price paths across all assets and horizons, preserving the full uncertainty in the distribution.

Beyond pure extrapolation, QRAFT supports three simulation methods for forecasting: **bootstrap**, **historical pass-through**, and **Copula-Marginal Adjustment (CMA)**, allowing the user to control the shape of the joint return distribution, stress tail dependence, or impose fat-tailed marginals on specific assets.

### Policy Creation

Because every scenario object carries an explicit probability vector, the full simulation output is always available for downstream risk and allocation steps nothing is collapsed to a point estimate prematurely. This also makes it straightforward to encode  **views** directly onto the scenario distribution using **Entropy Pooling**. Rather than discarding scenarios, their probabilities are updated to be consistent with the view (e.g. "AAPL mean return will be below historical average") by solving a minimum KL-divergence problem. Views can be placed on means, volatilities, correlations, or arbitrary moments.

Portfolio-level risk is computed from the simulated loss distribution, supporting both **VaR** and **CVaR** (more coming soon!). Risk attribution is available at the factor level, using top-down exposure estimation, minimum-torsion orthogonalisation, and Euler decomposition to attribute marginal and total risk contributions to each factor.

### Policy Evaluation



### Risk Management

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance, including the docstring style decision record.

---
