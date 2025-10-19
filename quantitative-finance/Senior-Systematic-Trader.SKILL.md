---
name: senior-systematic-trader
description: Live systematic trading and execution management. Use for TCA, execution optimization, and PnL ownership.
---

# Senior Systematic Trader

Operational playbook for running and improving systematic strategies with disciplined execution and risk.

## Instructions

### When to Use
- Live PnL ownership, intraday risk, and execution quality improvements
- Regime shifts requiring parameter/venue/routing adjustments
- Scale-up/rollout of new models with governance

### Expected Outcomes
- **Alpha after costs** improved via TCA and execution tuning
- **Risk discipline**: limit breaches minimized; controlled drawdowns
- **Operational excellence**: clean reconciliations, swift incident handling, compliant logs

### Required Inputs
- OMS/EMS access, venues, order types, routing constraints, dark/ATS policies
- Risk limits (instrument/sector/portfolio), leverage/margin rules, circuit breakers
- TCA baselines (slippage, fill rate, venue quality), liquidity profiles

### Operating Workflow
1) Establish dashboards/alerts: PnL, exposures, TCA (slippage/impact/fill), venue stats
2) Calibrate execution: participation/urgency curves, schedule (TWAP/VWAP/IS), venue selection
3) Monitor intraday anomalies; apply guardrails (pauses, de-risking, throttle)
4) Post-trade analytics: attribute costs, refine routing/params; change proposals with diffs
5) Governance: CRs for rollouts; canary with kill-switch; rollback on predefined thresholds
6) Daily reconciliation with broker/clearing; resolve breaks; audit trails

### Quality Checklist
- [ ] Pre-trade risk checks enforced; kill-switch tested
- [ ] Canary rollout with clear success criteria; automated rollback
- [ ] TCA segmented by venue/order type/time-of-day; seasonal effects
- [ ] Incident runbooks covered (data/feed, OMS, venue outages)
- [ ] Reconciliation and audit completeness

### Metrics & Validation
- TCA: implementation shortfall, slippage bps vs arrival, fill rates, cancel/replace efficacy
- Risk: exposure vs limits, VaR/ES, stress scenarios; breach count and duration
- Ops: reconciliation breaks, incident MTTR, change failure rate

## Tools & Technologies
- Languages: Python (operations tooling), SQL/kdb+
- Trading: OMS/EMS, FIX/OUCH; venue analytics; tick databases
- Observability: Grafana/Prometheus, ELK; alerting with paging

## Examples
- "Zbuduj dashboard TCA z segmentacją venue×order type×godzina; wskaż top 3 obszary poprawy."
- "Przygotuj parametry participation/urgency dla reżimu wysokiej zmienności i plan przełączenia."
- "Oceń efekty rolloutu canary i zdecyduj o rollbacku zgodnie z progami."

### Common Pitfalls
- Overfitting parametrów do krótkich okien; routing bez kontroli ryzyka
- Brak planu na awarie venue/oms/feed; niejawne ryzyko koncentracji

## References
- [Citadel Securities – Quantitative Trader](https://www.citadelsecurities.com/careers/details/quantitative-trader/)
- [Jane Street – Trading](https://www.janestreet.com/join-jane-street/)
- [Jump Trading – Careers](https://www.jumptrading.com/careers/)
- [Anthropic – Skills (Claude)](https://www.anthropic.com/news/skills)


