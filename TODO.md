## Alpha releast checklist
- [ ] Update README
- [ ] Get pypi ready
- [x] Update docstrings ✅ 2026-07-13
- [ ] Create examples
- [x] Full green-suite + tests organization ✅ 2026-07-13
- [ ] Review MarketData as a bit too large?
- [ ] Also whether AssetUniverse is needed...


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
- [ ] Distinguish between market-wide and cross-sectional
- [ ] Add layer to create expected returns


## Optimization
- [ ] Review and implement better covariances techniques
    - [ ] Look at 72.2.1
- [ ] Add option to do against benchmark
    - [ ] current portfolio
    - [ ] equal weight portfolio
    - [ ] all cash
    - [ ] custom
- [ ] Add sparse optimization

## Forecasting pipeline
- [ ] Change model selection pipeline to drop bad forecast models?
- [ ] GARCH/ARCH should reflect prob weight...


## Lower Priority
- [ ] Go back and find ways of fixing missing values for copulas, curretly just dropping, but possible to replace...somehow...


## Completed
