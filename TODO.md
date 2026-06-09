## Backtest
- [ ] risk evaluation/backtest
Kupiec test
Christoffersen test
Mixed Kupiec test

- [ ] full predictive distribution
PIT tests
CRPS scoring
Tail calibration

## Signals
- [x] Add basic raw signals ✅ 2026-06-09
    - [ ] Distinguish between market-wide and cross-sectional
- [x] Add signals processing: ✅ 2026-06-09
    - [x] Smoothing ✅ 2026-06-09
    - [x] Scoring ✅ 2026-06-09
    - [x] Ranking ✅ 2026-06-09
- [x] Processed signals to expected returns (Characteristics) ✅ 2026-06-09

## Optimization
- [ ] Review and implement better covariances techniques
    - [ ] Look at 72.2.1


## Forecasting pipeline
- [ ] Add metrics to evaluate forecast:
    - [ ] Add CRPS scores 
- [ ] Change model selection pipeline to drop bad forecast models
- [ ] Add EP option for simulation probabilities to reflect forecast views
- [x] create weighted OLS ✅ 2026-03-31
- [ ] Check out with white_noise test produce different answers 
- [ ] GARCH/ARCH should reflect prob weight...
- [ ] apply multiple tests to iid pipeline?
- [ ] Add check whether final innovs are iid?
- [ ] Add 'block' resampling instead to better preserve dynamics
- [ ] Add back probabilities for forecasts that were used from MC
- [ ] Simplify seasonality as currently just noise fitting
- [ ] Check whether random walk model is being activated correctly


## Lower Priority
- [ ] Go back and find ways of fixing missing values for copulas, curretly just dropping, but possible to replace...somehow...
- [ ] Create a 'prior' object to be used on ScenarioDist 
- [ ] Finish setting up effective_ranks (helps us know whether our views are fine to use by determining rank of matrix)
- [ ] Change assigned prob plot to always start at 0 
- [ ] Write own 'estimation' for copulas?





## Completed
