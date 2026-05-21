## Signals
- [ ] Add basic raw signals
    - [ ] Distinguish between market-wide and cross-sectional
- [ ] Add signals processing:
    - [ ] Smoothing
    - [ ] Scoring
    - [ ] Ranking
- [ ] Processed signals to expected returns (Characteristics)

## Optimization
- [x] Start with simple MV optimization ✅ 2026-05-20
    - [x] Calculate first two moments of simulations ✅ 2026-05-15
- [x] Implement Multi period optimization (MPO) ✅ 2026-05-20
    - [x] Simple implementation ✅ 2026-05-18
    - [x] make constraints easier ✅ 2026-05-19
    - [x] make objectives easier ✅ 2026-05-20
- [x] Apply CVAR optimization in this framework ✅ 2026-05-21
- [ ] Create efficient frontier?
- [ ] Review and implement better covariances techniques
    - [ ] Look at 72.2.1
    - [ ] Also look at expectations etc as currently some are very high (ties in with bad forecasts in general)
- [ ] Decide how to tackle invest. univese problem as want to keep optimizer as convex (meaning it won't auto remove (ie put to 0) bad assets)


## Simulation
- [ ] Using t-copula bootstrap with extreme returns is ending up in 'inf' and 'nans', how to resolve this.


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
