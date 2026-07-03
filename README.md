# QRAFT

**Quantitative Risk, Allocation, and Forecasting Toolkit**

QRAFT is an end-to-end quantitative portfolio construction toolkit built around probabilistic, scenario-based thinking. Rather than producing point estimates, every stage of the pipeline, from forecasting through to risk and allocation, works with full distributions of simulated outcomes, each carrying an explicit probability weight.

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

## Logging

QRAFT uses standard Python logging with project-level event helpers. The default `INFO` output is intended to show lifecycle milestones without flooding notebooks or validation runs.

```python
import logging

from qraft import LogConfig, setup_logging

setup_logging(LogConfig(level=logging.INFO))
```

Recommended modes:

- Research default: `LogConfig(level=logging.INFO)`
- Quiet notebooks: `LogConfig(level=logging.WARNING)`
- Debug one run: `LogConfig(level=logging.DEBUG)`
- Quiet console with detailed file logs: `LogConfig(level=logging.DEBUG, console_level=logging.WARNING, file_level=logging.DEBUG, log_file="qraft.log")`

Level policy:

- `DEBUG`: repeated internals such as per-asset model choices, simulation shapes, optimizer success details, and candidate success details.
- `INFO`: major lifecycle events such as backtest start/end, validation start/end, fold summaries, forecast recipe selection, and policy input construction.
- `WARNING`: recoverable degradations that may affect output quality, including solver failures, held decisions, dropped forecast assets, and non-convergence.
- `ERROR`: invalid states or failures that prevent a valid result.

Core event names:

- Backtest: `backtest.started`, `backtest.completed`, `backtest.decision_failed`, `backtest.no_decisions`
- Policy inputs and allocation: `policy_inputs.started`, `policy_inputs.completed`, `optimization.completed`, `optimization.failed`, `optimization.nonconverged`
- Validation: `validation.started`, `validation.candidates_precomputed`, `validation.candidate_completed`, `validation.candidate_failed`, `validation.fold_completed`, `validation.completed`, `validation.walk_forward_started`, `validation.walk_forward_completed`, `validation.combinatorial_started`, `validation.combinatorial_completed`
- Forecasting: `forecast.recipe_build_started`, `forecast.recipe_selected`, `forecast.recipe_build_completed`, `forecast.run_started`, `forecast.recipe_applied`, `forecast.completed`, `forecast.asset_dropped`

Each event stores structured context on the log record as `qraft_context`. By default that context is appended to the message. To keep messages compact while preserving structured context for handlers/tests, use:

```python
setup_logging(LogConfig(level=logging.INFO, include_context=False))
```
