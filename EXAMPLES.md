# Przykłady praktycznych promptów dla Skills

## 🐍 Python Programming Skill

### Przykład 1: Type Hints i dokumentacja
```
Prompt: "Napisz funkcję do przetwarzania danych JSON z pełnymi 
type hints i docstring zgodnym z PEP 257"

Claude użyje: python-programming skill
- Zastosuje type hints (PEP 484)
- Doda docstring (PEP 257)
- Użyje error handlingu
```

### Przykład 2: Async programming
```
Prompt: "Potrzebuję async funkcji do pobierania danych z 
wielu API jednocześnie używając asyncio"

Claude użyje: python-programming skill
- Użyje asyncio
- Implementuje async context managers
- Zastosuje error handling
```

### Przykład 3: Testing
```
Prompt: "Napisz testy pytest z fixtures i mocking dla 
mojej funkcji API call"

Claude użyje: python-programming skill
- Stworzy fixtures pytest
- Użyje pytest.mark.parametrize
- Implementuje mocking z unittest.mock
```

---

## 🏗️ Software Engineering Skill

### Przykład 1: SOLID principles
```
Prompt: "Review tego kodu pod kątem SOLID principles 
i zaproponuj refaktoring"

Claude użyje: software-engineering skill
- Sprawdzi każdą z 5 zasad SOLID
- Zidentyfikuje naruszenia
- Zaproponuje konkretny refactoring
```

### Przykład 2: Microservices architecture
```
Prompt: "Zaprojektuj event-driven microservices architecture 
dla systemu e-commerce z payment, inventory i order services"

Claude użyje: software-engineering skill
- Zaproponuje Event-Driven Architecture
- Użyje message queue (RabbitMQ/Kafka)
- Zaproponuje API Gateway pattern
```

### Przykład 3: API Design
```
Prompt: "Zaprojektuj RESTful API dla systemu blogowego 
z wersjonowaniem i autentykacją"

Claude użyje: software-engineering skill
- Zastosuje REST best practices
- Zaproponuje versioning strategy
- Implementuje authentication/authorization
```

---

## 🤖 Machine Learning Skill

### Przykład 1: Complete ML Pipeline
```
Prompt: "Stwórz kompletny ML pipeline do predykcji churn:
- preprocessing danych
- feature engineering  
- training XGBoost
- evaluation i tuning
- deployment z FastAPI"

Claude użyje: machine-learning skill
- Data preprocessing (scaling, encoding)
- Feature engineering
- Model training z XGBoost
- Hyperparameter tuning
```

### Przykład 2: Deep Learning
```
Prompt: "Potrzebuję CNN w PyTorch do klasyfikacji obrazów 
z data augmentation i transfer learning"

Claude użyje: machine-learning skill
- Stworzy CNN architecture
- Implementuje data augmentation
- Użyje transfer learning (ResNet/VGG)
```

### Przykład 3: MLOps
```
Prompt: "Jak wdrożyć model ML do produkcji z:
- versioning (MLflow)
- containerization (Docker)
- monitoring
- A/B testing"

Claude użyje: machine-learning skill
- MLflow dla versioning
- Docker deployment
- Monitoring (Prometheus)
```

---

## 📈 Quantitative Finance & Trading Skill

### Przykład 1: Backtesting Framework
```
Prompt: "Zbuduj professional backtesting framework z:
- Realistic transaction costs i slippage
- Position sizing z Kelly criterion
- Walk-forward optimization
- Performance metrics (Sharpe, Calmar, max DD)"

Claude użyje: quantitative-finance skill
- Backtesting engine z realistic assumptions
- Kelly criterion position sizing
- Walk-forward validation
- Comprehensive metrics calculation
```

### Przykład 2: Statistical Arbitrage Strategy
```
Prompt: "Implement pairs trading strategy:
- Cointegration test dla pair selection
- Z-score mean reversion signals
- Half-life calculation
- Entry/exit thresholds optimization"

Claude użyje: quantitative-finance skill
- Statistical tests (ADF, cointegration)
- Mean reversion strategy logic
- Parameter optimization
- Backtesting with proper validation
```

### Przykład 3: Order Management System
```
Prompt: "Build order management system z:
- Pre-trade risk checks
- Order routing logic
- Fill simulation
- Position reconciliation"

Claude użyje: quantitative-finance skill
- OMS architecture
- Risk management integration
- Order lifecycle management
- Real-time position tracking
```

### Przykład 4: Portfolio Optimization
```
Prompt: "Optimize portfolio używając:
- Mean-variance optimization
- Risk parity approach
- Black-Litterman with views
- Rebalancing strategy"

Claude użyje: quantitative-finance skill
- Portfolio optimization algorithms
- Risk models
- Constraint handling
- Performance attribution
```

### Przykład 5: Market Microstructure Analysis
```
Prompt: "Analyze order book dynamics:
- Bid-ask spread analysis
- Order book imbalance
- Market impact modeling
- Optimal execution (TWAP/VWAP)"

Claude użyje: quantitative-finance skill
- Order book processing
- Microstructure metrics
- Execution algorithms
- Transaction cost analysis
```

### Przykład 6: ML Trading Strategy
```
Prompt: "Stwórz ML-based trading strategy:
- Feature engineering z technical indicators
- XGBoost dla signal generation
- Walk-forward validation
- Risk-adjusted position sizing"

Claude użyje: quantitative-finance + machine-learning
- Financial feature engineering
- ML model training z cross-validation
- Backtesting framework
- Risk management
```

### Przykład 7: Senior Quantitative Developer (role-specific Skill)
```
Prompt: "Reduce P99 latency of feed-handler to < 50 µs, show flamegraph
before/after and describe changes in memory allocation"

Claude użyje: senior-quantitative-developer
- Hot path profiling and flamegraph analysis
- Elimination of allocations on critical path
- NIC/IRQ/RSS tuning and end-to-end timestamping
- Structured approach: Outcomes → Inputs → Implementation → Validation
```

### Przykład 8: Senior Quantitative Researcher (role-specific Skill)
```
Prompt: "Conduct walk-forward for momentum strategy with realistic costs,
check for leakage and prepare alpha stability report (OOS)"

Claude użyje: senior-quantitative-researcher
- Bias controls (look-ahead, survivorship)
- Time-series CV / walk-forward and ablation studies
- Capacity/impact modeling and live↔backtest tracking
- Research pipeline: Hypothesis → Data → Model → Validation → Production
```

### Przykład 9: Senior Systematic Trader (role-specific Skill)
```
Prompt: "Build TCA dashboard (venue×order type×time),
propose participation/urgency adjustments for high volatility regime"

Claude użyje: senior-systematic-trader
- TCA and cost segmentation
- Execution calibration (TWAP/VWAP/IS, routing)
- Canary rollout and rollback conditions
- Operating workflow: Monitor → Calibrate → Adjust → Reconcile
```

### Przykład 10: Senior Quantitative Trader (role-specific Skill)
```
Prompt: "Prepare quarterly attribution review (alpha/beta/costs),
recommend sizing and hedging adjustments for lower maxDD while maintaining CAGR"

Claude użyje: senior-quantitative-trader
- Portfolio KPIs (PnL, MAR/Calmar, maxDD)
- Factor control and capacity management
- Rollout/rollback decisions based on KPIs and TCA
- Lifecycle management: Objectives → Calibration → Oversight → Review
```

---

## 🏢 Role-specific Skills - Struktura zgodna z Anthropic

Wszystkie 4 role-specific Skills są zgodne z oficjalną dokumentacją Anthropic i zawierają:

### ✅ Frontmatter (metadata - zawsze ładowane)
```yaml
---
name: skill-name              # ≤64 znaki
description: brief purpose    # ≤1024 znaki
---
```

### ✅ Instructions (ładowane przy użyciu)
- **When to Use**: kiedy aktywować ten Skill
- **Expected Outcomes**: mierzalne rezultaty
- **Required Inputs**: wymagane dane/kontekst
- **Implementation Steps**: przepływ pracy krok po kroku
- **Quality Checklist**: kontrola jakości
- **Metrics & Validation**: jak mierzyć sukces
- **Common Pitfalls**: czego unikać

### ✅ Tools & Technologies (ładowane na żądanie)
Stack technologiczny specyficzny dla roli

### ✅ Examples (ładowane na żądanie)
Gotowe prompty i zadania

### ✅ References (ładowane na żądanie)
Linki do oficjalnych źródeł (firmy top-tier + Anthropic)

**Zalety tego podejścia:**
- ⚡ **Progressive loading** - Claude ładuje tylko to, co jest potrzebne
- 🔗 **Composability** - Skills mogą współpracować (np. senior-quant-researcher + machine-learning)
- 📱 **Portability** - działa w Claude.ai, Claude Code, API
- 🎯 **Specificity** - precyzyjne guidance dla konkretnej roli i zadania

---

## 🎯 Kombinacje Skills

### Przykład 1: Complete Quant Trading System
```
Prompt: "Stwórz kompletny systematic trading system:
1. Statistical arbitrage strategy z mean reversion
2. Backtesting framework z realistic costs
3. ML enhancement z feature engineering
4. Risk management z VaR limits
5. Production deployment z monitoring
6. REST API dla trade signals"

Claude użyje wszystkie 4 skills:
✓ quantitative-finance: strategy, backtesting, risk management
✓ machine-learning: ML model, feature engineering
✓ python-programming: clean code, type hints, async
✓ software-engineering: architecture, API design, deployment
```

### Przykład 2: High-Frequency Trading Infrastructure
```
Prompt: "Design HFT system z:
- Low-latency market data processing
- Order book analysis w real-time
- Optimal execution algorithms
- Pre-trade risk checks
- Performance monitoring
- Microservices architecture"

Claude użyje:
✓ quantitative-finance: microstructure, execution, risk
✓ software-engineering: low-latency design, microservices
✓ python-programming: high-performance implementation
```

### Przykład 3: Quantitative Research Platform
```
Prompt: "Build research platform dla alpha discovery:
- Data pipeline z multiple sources
- Feature store dla ML
- Backtesting engine z walk-forward
- Factor analysis tools
- Performance attribution
- Research notebook integration"

Claude użyje:
✓ quantitative-finance: research framework, factors
✓ machine-learning: feature store, ML pipeline
✓ software-engineering: data architecture
✓ python-programming: implementation
```

### Przykład 4: Algorithmic Trading Production System
```
Prompt: "Deploy algorithmic trading system z:
- Multiple trading strategies
- Real-time position management
- Risk limits monitoring
- PnL tracking i attribution
- Automated alerts
- Trade reconciliation"

Claude użyje:
✓ quantitative-finance: trading logic, risk, PnL
✓ software-engineering: production architecture, monitoring
✓ python-programming: async operations, logging
```

---

## 💡 Pro Tips

### Tip 1: Bądź konkretny w financial context
```
❌ "Stwórz trading strategy"
✅ "Implement momentum strategy z 20-day lookback, 
    2-sigma entry threshold i Kelly position sizing"
```

### Tip 2: Wymień konkretne metody
```
❌ "Optimize portfolio"
✅ "Optimize portfolio używając mean-variance optimization 
    z constraints: max 10% per position, sector limits"
```

### Tip 3: Określ validation approach
```
❌ "Backtest strategy"
✅ "Backtest strategy z walk-forward optimization (2-year 
    training, 6-month testing) i realistic transaction costs"
```

### Tip 4: Łącz domain expertise
```
"Używając quantitative-finance i machine-learning skills,
build ML trading strategy z:
- Feature engineering z technical + fundamental data
- LightGBM model z proper cross-validation
- Risk-adjusted position sizing
- Production monitoring"
```

---

## 🧪 Test Your Skills

Wypróbuj te prompty w Claude Code:

### Test 1: Python
```
"Napisz decorator w Python do mierzenia czasu wykonania funkcji, 
z type hints i proper error handling"
```
**Oczekiwane:** python-programming skill

### Test 2: Software Engineering  
```
"Zaprojektuj CQRS pattern dla systemu zamówień z event sourcing"
```
**Oczekiwane:** software-engineering skill

### Test 3: Machine Learning
```
"Stwórz LSTM w PyTorch do predykcji szeregów czasowych 
z early stopping i learning rate scheduling"
```
**Oczekiwane:** machine-learning skill

### Test 4: Quantitative Finance
```
"Implement pairs trading strategy z cointegration test,
mean reversion signals, i walk-forward optimization"
```
**Oczekiwane:** quantitative-finance skill

### Test 5: Multiple Skills - Quant System
```
"Build complete algorithmic trading system:
- Mean reversion strategy z statistical tests
- Professional backtesting framework
- ML signal enhancement z XGBoost
- Risk management z VaR limits
- REST API deployment
- Real-time monitoring"
```
**Oczekiwane:** wszystkie 4 skills

---

## 📊 Matryca promptów

### Podstawowe Skills

| Chcę... | Używaj słów kluczowych | Skills |
|---------|------------------------|--------|
| Napisać kod Python | "function", "type hints", "pytest" | Python |
| Zaprojektować system | "architecture", "microservices" | SWE |
| Zbudować model ML | "model", "training", "PyTorch" | ML |
| Trading strategy | "backtest", "signals", "portfolio" | Quant |
| Quant research | "statistical test", "factor analysis" | Quant |
| Order execution | "OMS", "execution", "slippage" | Quant |
| Portfolio opt | "mean-variance", "risk parity" | Quant |
| Market microstructure | "order book", "market impact" | Quant |
| ML trading | "feature engineering", "ML model" | Quant + ML |
| Production trading | "deployment", "monitoring", "risk" | Quant + SWE |

### Role-specific Skills (Senior Positions)

| Role | Użyj gdy... | Kluczowe słowa | Skill |
|------|------------|----------------|-------|
| **Quant Developer** | Optymalizacja latencji/throughput | "latency", "P99", "profiling", "flamegraph" | senior-quantitative-developer |
| **Quant Researcher** | Alpha research i validation | "walk-forward", "bias", "OOS", "capacity" | senior-quantitative-researcher |
| **Systematic Trader** | Live execution i TCA | "TCA", "execution", "venue", "rollout" | senior-systematic-trader |
| **Quant Trader** | Portfolio i PnL management | "attribution", "KPI", "sizing", "hedging" | senior-quantitative-trader |

---

## 🎓 Learning Path - Quantitative Trading

### Beginner
1. Zacznij od prostych strategii
   ```
   "Implement simple moving average crossover strategy"
   "Calculate Sharpe ratio dla returns"
   ```

2. Naucz się backtesting basics
   ```
   "Build basic backtesting framework z transaction costs"
   ```

3. Zrozum risk management
   ```
   "Implement position sizing z fixed percentage"
   ```

### Intermediate  
1. Statistical arbitrage
   ```
   "Test for cointegration i implement pairs trading"
   ```

2. Factor models
   ```
   "Build multi-factor model z Fama-French factors"
   ```

3. ML enhancement
   ```
   "Add ML layer do existing strategy"
   ```

### Advanced
1. Complex strategies
   ```
   "Build multi-strategy portfolio z risk allocation"
   ```

2. Production systems
   ```
   "Deploy trading system z real-time monitoring"
   ```

3. Research automation
   ```
   "Build automated alpha discovery platform"
   ```

---

## 🚀 Real-World Scenarios

### Scenario 1: Junior Quant Developer
```
"Jestem junior quant developer. Pomóż mi:
1. Zrozumieć backtesting best practices
2. Implement simple mean reversion strategy
3. Add proper transaction costs
4. Calculate performance metrics"

Claude użyje: quantitative-finance + python-programming
```

### Scenario 2: Quantitative Researcher
```
"Conducting research na momentum anomaly:
1. Statistical tests dla momentum effect
2. Factor regression analysis
3. Controlling for risk factors
4. Out-of-sample validation"

Claude użyje: quantitative-finance + machine-learning
```

### Scenario 3: Trading System Developer
```
"Building production trading system:
1. Low-latency architecture
2. Order management system
3. Real-time risk monitoring
4. Trade reconciliation
5. Automated alerts"

Claude użyje: quantitative-finance + software-engineering
```

### Scenario 4: Portfolio Manager
```
"Optimizing multi-asset portfolio:
1. Mean-variance optimization
2. Risk budgeting across strategies
3. Dynamic rebalancing
4. Performance attribution"

Claude użyje: quantitative-finance
```

---

## 🚀 Advanced Scenarios - Role-specific Skills

### Scenario 5: Senior Quantitative Developer @ HFT Firm
```
"Jestem Senior Quant Developer w HFT. Potrzebuję:
1. Zredukować P99 latencji market data handlera z 200µs do <50µs
2. Zaimplementować bounded backpressure dla peak volume
3. Dodać end-to-end timestamping z PTP
4. Przygotować canary deployment z automated rollback
5. Dashboardy observability (latency histograms, flamegraphs)"

Claude użyje: senior-quantitative-developer
- Structured approach: Outcomes → Inputs → Implementation → Validation
- Hot path optimization (C++, zero-copy, lock-free)
- Network tuning (NIC/IRQ/RSS, kernel bypass)
- CI/CD hardening (canary, rollback, SLOs)
```

### Scenario 6: Senior Quantitative Researcher @ Hedge Fund
```
"Prowadzę research nad equity momentum. Potrzebuję:
1. Pipeline danych z kontrolą biasów (survivorship, look-ahead)
2. Feature engineering (microstructure, technical, fundamental)
3. Walk-forward validation z realistic costs/impact
4. Ablation study i sensitivity analysis
5. Capacity modeling i live-to-backtest tracking
6. Dokumentacja i handover do production"

Claude użyje: senior-quantitative-researcher
- Research pipeline: Hypothesis → Data → Features → Model → Validation
- Bias controls i robustness checks
- Production-ready artifacts (versioned, documented, tested)
```

### Scenario 7: Senior Systematic Trader @ Prop Trading
```
"Zarządzam live systematic strategies. Potrzebuję:
1. TCA dashboard (segmentacja: venue×order type×time)
2. Execution parameter tuning dla high-volatility regime
3. Canary rollout nowej wersji strategii z kill-switch
4. Daily reconciliation automation
5. Incident playbooks dla data/OMS/venue outages"

Claude użyje: senior-systematic-trader
- Operating workflow: Monitor → Calibrate → Adjust → Reconcile
- TCA optimization i execution quality
- Governance (canary, rollback, compliance)
```

### Scenario 8: Senior Quantitative Trader @ Multi-Manager
```
"Prowadzę multi-strategy portfolio. Potrzebuję:
1. Quarterly attribution review (alpha/beta/costs breakdown)
2. Factor exposure analysis i rebalancing recommendations
3. Sizing/hedging adjustments dla maxDD reduction
4. Rollout decision framework (KPIs, thresholds, escalation)
5. Automated KPI tracking i alerts"

Claude użyje: senior-quantitative-trader
- Lifecycle: Objectives → Calibration → Oversight → Review
- Data-driven decisions (attribution, TCA, risk)
- Portfolio governance i discipline
```

---

## 🎯 Wybór odpowiedniego Skill - Decision Tree

```
Pytanie: Jak wybrać odpowiedni Skill?

┌─ Pracujesz z trading/finance? ─────────────────┐
│                                                 │
│  NIE → python-programming / software-eng / ML  │
│                                                 │
│  TAK ↓                                          │
│  ┌─ Jaka rola/zadanie? ────────────────────┐   │
│  │                                          │   │
│  │ Infrastructure/latency optimization?     │   │
│  │ → senior-quantitative-developer          │   │
│  │                                          │   │
│  │ Alpha research/strategy development?     │   │
│  │ → senior-quantitative-researcher         │   │
│  │                                          │   │
│  │ Live execution/TCA/operations?           │   │
│  │ → senior-systematic-trader               │   │
│  │                                          │   │
│  │ Portfolio management/attribution/PnL?    │   │
│  │ → senior-quantitative-trader             │   │
│  │                                          │   │
│  │ General trading/backtesting?             │   │
│  │ → quantitative-finance (base)            │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

Powodzenia w budowaniu trading systems! 🚀📈
