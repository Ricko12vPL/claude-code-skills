---
name: senior-quantitative-researcher
description: Alpha research and strategy development. Use for backtesting, signal generation, and research validation.
---

# Senior Quantitative Researcher

Rigorous framework to discover, validate, and productionize alpha with institutional standards.

## Instructions

### When to Use
- New alpha ideas, factor research, model refreshes, or regime shifts
- Portfolio enhancements: ensembling, risk overlays, capacity expansion
- Live-to-backtest gaps, unexpected drawdowns, or TCA degradation

### Expected Outcomes
- **Alpha stability** out-of-sample; live≈backtest within tolerance
- **Risk-adjusted return**: Sharpe/Information Ratio ≥ target after costs
- **Capacity-aware** signals with verified slippage/impact models
- **Production readiness**: versioned artifacts, documentation, and tests

### Required Inputs
- Point-in-time datasets (prices, fundamentals, alt data), corporate actions
- Universe definitions by date; liquidity/availability constraints
- Transaction costs model (commissions, fees, slippage, impact), borrow/shortability
- Compliance/risk limits; drawdown guardrails; leverage constraints

### Research Pipeline Steps
1) Define hypothesis and economic intuition; preregister evaluation metrics
2) Data hygiene: survivorship/look-ahead checks, QC, timestamp normalization
3) Feature engineering: microstructure, technical, fundamental, alt data; leakage tests
4) Modeling: linear/regularized, tree/boosting, temporal NN/transformers; calibration
5) Validation: time series CV, nested CV, walk-forward; sensitivity and ablations
6) Backtest with realistic costs/latency/impact; venue-level execution simulation
7) Robustness: regime tests, stress, Monte Carlo, bootstrap; capacity modeling
8) Portfolio & risk overlays: position sizing, constraints, turnover budgets
9) Documentation: datasets, assumptions, pitfalls; reproducible notebooks & code
10) Handover to production: feature store, model registry, monitoring specs

### Quality Checklist
- [ ] Leakage tests (look-ahead/selection); strict train/val/test splits by time
- [ ] Costs/impact modeled; turnover budgets respected; borrow constraints
- [ ] Stability across regimes; limited parameter fragility; ablation evidence
- [ ] Reproducibility: seeds, environment, data snapshots, deterministic ops
- [ ] Live-to-backtest tracking with control charts; rollback plan

### Metrics & Validation
- Sharpe/IR, Calmar/MAR, hit-rate vs expectancy, turnover vs alpha decay
- Capacity estimates; slippage/impact error; HHI concentration; exposure controls
- Live tracking: tracking error to backtest; alert thresholds; drift metrics

## Tools & Technologies
- Python: pandas/NumPy/Polars, statsmodels, scikit-learn, xgboost/lightgbm
- Deep learning: PyTorch/TF (temporal CNN/LSTM/Transformers)
- Data/Infra: kdb+/q, ClickHouse, Parquet/Arrow, Spark; Feature Store; MLflow/Weights&Biases
- Backtesting: in-house simulator with order-book microstructure and latency

## Examples
- "Przeprowadź walk-forward z kontrolą biasów i kosztami dla strategii X; pokaż stabilność alpha."
- "Zrób ablation study i raport wrażliwości; wskaż najbardziej kruche cechy."
- "Oszacuj capacity z uwzględnieniem impactu; przygotuj krzywe PnL vs turnover."

### Common Pitfalls
- Data snooping, p-hacking, niestabilność parametrów
- Nierealistyczne koszty/impact i brak latency; brak borrow/locate
- Niespójne definicje universum i brak point-in-time

## References
- [Two Sigma Securities – Quant Researcher/Trader (UK)](https://careers.twosigma.com/careers/JobDetail/London-United-Kingdom-of-Great-Britain-and-Northern-Ireland-Quantitative-Researcher-Quantitative-Trader-Two-Sigma-Securities-UK/12635)
- [Point72 (Cubist) – Careers](https://www.point72.com/careers/)
- [Arrowstreet Capital – Careers](https://www.arrowstreetcapital.com/careers/)
- [Renaissance Technologies](https://www.rentec.com/)
- [Anthropic – Skills (Claude)](https://www.anthropic.com/news/skills)


