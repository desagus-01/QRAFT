# QRAFT

**Quantitative Risk, Allocation, and Forecasting Toolkit**

QRAFT is an end-to-end quantitative portfolio construction toolkit built around semi-parametric/non-parametrics methods based on scenarios.

The goal is a single coherent research environment where forecasting, risk, and portfolio decisions are not siloed tools bolted together, but stages of one integrated, simulation-driven workflow.

> **Disclaimer:** Research and educational use only. Not financial or investment advice.

---

## What QRAFT Does

### Research & Forecasting

QRAFT provides a full probabilistic forecasting pipeline for financial time series. Raw price data is preprocessed per-asset (stationarity checks, detrending, deseasoning), a mean and volatility model is selected automatically, and innovations are extracted and used to drive a Monte Carlo simulation. The output is never a single forecast, it is always a set of probability-weighted simulated price paths across all assets and horizons, preserving the full uncertainty in the distribution.

Beyond pure extrapolation, QRAFT supports three simulation methods for forecasting: **bootstrap**, **historical pass-through**, and **Copula-Marginal Adjustment (CMA)**, allowing the user to control the shape of the joint return distribution, stress tail dependence, or impose fat-tailed marginals on specific assets.

### Risk Management

Because every scenario object carries an explicit probability vector, the full simulation output is always available for downstream risk and allocation steps nothing is collapsed to a point estimate prematurely. This also makes it straightforward to encode  **views** directly onto the scenario distribution using **Entropy Pooling**. Rather than discarding scenarios, their probabilities are updated to be consistent with the view (e.g. "AAPL mean return will be below historical average") by solving a minimum KL-divergence problem. Views can be placed on means, volatilities, correlations, or arbitrary moments.

Portfolio-level risk is computed from the simulated loss distribution, supporting both **VaR** and **CVaR** (more coming soon!). Risk attribution is available at the factor level, using top-down exposure estimation, minimum-torsion orthogonalisation, and Euler decomposition to attribute marginal and total risk contributions to each factor.

### Portfolio Construction

**STILL BEING WORKED ON**

---

## Roadmap

- [ ] Signals interface for incorporating alpha into allocation
- [ ] Portfolio optimisation (risk-budgeting, mean-variance under constraints)
- [ ] Execution module (order generation, cost modelling)
- [ ] Entropy Pooling integration into simulation probability weights
- [ ] CRPS scores for forecast evaluation
- [ ] Block resampling to better preserve return dynamics
- [ ] Expand test coverage across pipeline and simulation entry points

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidance, including the docstring style decision record.

---
