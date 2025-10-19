# Claude Code Skills - Python, Software Engineering, ML, Quantitative Finance

Profesjonalne Skills dla Claude Code zgodne z oficjalną specyfikacją Anthropic.

## 📦 Co zawiera ten pakiet?

### 1. 🐍 Python Programming Skill

**Kiedy użyć:** Pisanie, refaktoryzacja, debugowanie kodu Python

**Co zawiera:**
- ✅ PEP 8, PEP 257, PEP 484 (style, docstrings, type hints)
- ✅ Struktura projektów Python (pyproject.toml)
- ✅ Nowoczesne features (dataclasses, context managers, async/await)
- ✅ Design patterns (Singleton, Factory, Decorator, Observer)
- ✅ Error handling i custom exceptions
- ✅ Testing z pytest (fixtures, parametrize, mocking)
- ✅ Performance optimization (comprehensions, generators, profiling)
- ✅ Async programming (asyncio, async context managers)
- ✅ Dependencies management
- ✅ Best practices checklist

---

### 2. 🏗️ Software Engineering Skill

**Kiedy użyć:** Projektowanie systemów, code review, architektura

**Co zawiera:**
- ✅ SOLID principles (wszystkie 5 z przykładami)
- ✅ Architectural patterns (Layered, Microservices, Event-Driven, CQRS)
- ✅ Design patterns (Creational, Structural, Behavioral)
- ✅ Clean Code principles
- ✅ Testing strategies (Test Pyramid, Unit, Integration, E2E)
- ✅ CI/CD best practices (GitHub Actions, pre-commit hooks)
- ✅ API design (RESTful, versioning)
- ✅ Database design (normalization, indexing, migrations)
- ✅ Monitoring & Observability
- ✅ Security best practices

---

### 3. 🤖 Machine Learning Skill

**Kiedy użyć:** Budowanie modeli ML/DL, training, deployment

**Co zawiera:**
- ✅ Complete ML workflow (data → deployment)
- ✅ Data preprocessing (missing data, outliers, scaling, encoding)
- ✅ Feature engineering (creation, selection, importance)
- ✅ Classical ML algorithms (Linear, Trees, SVM, KNN, Naive Bayes)
- ✅ Ensemble methods (Random Forest, XGBoost, LightGBM, CatBoost)
- ✅ Model evaluation (classification & regression metrics)
- ✅ Hyperparameter tuning (Grid, Random, Bayesian)
- ✅ Deep Learning z PyTorch (Neural Networks, CNN, RNN/LSTM)
- ✅ Transfer Learning
- ✅ MLOps (serialization, versioning, monitoring)
- ✅ Model deployment (FastAPI, Docker)

---

### 4. 📈 Quantitative Finance & Trading Skill

**Kiedy użyć:** Trading algorithms, quantitative research, systematic trading

**Co zawiera:**
- ✅ Quantitative research framework (alpha research, backtesting)
- ✅ Trading system architecture (OMS, execution systems)
- ✅ Statistical methods (time series, GARCH, cointegration)
- ✅ Factor models (multi-factor attribution)
- ✅ Machine learning for trading (feature engineering, ML models)
- ✅ Professional backtesting engine
- ✅ Market microstructure (order book analysis, execution optimization)
- ✅ Portfolio optimization (mean-variance, risk parity, Black-Litterman)
- ✅ Risk management (VaR, position sizing, Kelly criterion)
- ✅ Production deployment (monitoring, alerting, reconciliation)
- ✅ Best practices for quant development

#### 4.a Role-specific Skills (Quant) 🆕

Dodaliśmy 4 wyspecjalizowane Skills przeznaczone dla ról seniorskich w tradingu ilościowym, oparte na wymaganiach z top firm (Citadel, Jane Street, HRT, Jump, Two Sigma, Point72, PDT, Arrowstreet). Każdy jest osobnym plikiem w `quantitative-finance/`:

**`Senior-Quantitative-Developer.SKILL.md`**
- Low-latency infrastructure (market data, execution, risk)
- Performance optimization (P50/P95/P99 latency, throughput)
- Production hardening (observability, CI/CD, incident response)
- Stack: C++20/23, Python, Bazel/CMake, DPDK/XDP, kdb+/q

**`Senior-Quantitative-Researcher.SKILL.md`**
- Alpha research pipeline (hypothesis → validation → production)
- Bias-safe backtesting (look-ahead, survivorship, costs)
- Walk-forward, capacity modeling, live↔backtest tracking
- Stack: Python, pandas/NumPy, scikit-learn, PyTorch, kdb+/q

**`Senior-Systematic-Trader.SKILL.md`**
- Live PnL ownership and execution management
- TCA optimization (venue, order type, timing)
- Canary rollout/rollback with governance
- Stack: Python, SQL/kdb+, OMS/EMS, Grafana

**`Senior-Quantitative-Trader.SKILL.md`**
- Portfolio-level strategy ownership
- KPI tracking (PnL, Sharpe, MAR, maxDD)
- Attribution (alpha/beta/costs) and sizing/hedging
- Stack: Python, SQL/kdb+, portfolio analytics, dashboards

**Zgodność z Anthropic Skills:**
- ✅ Frontmatter limits (name ≤64, description ≤1024)
- ✅ Structured sections (Instructions, Tools, Examples, References)
- ✅ Progressive disclosure design
- ✅ Composable with other Skills

---

## 🚀 Szybki start

### 1. Instalacja (wybierz jedną metodę)

**Personal Skills** (dostępne wszędzie):
```bash
cp -r python-programming software-engineering machine-learning quantitative-finance ~/.claude/skills/
```

**Project Skills** (tylko w projekcie):
```bash
mkdir -p .claude/skills
cp -r python-programming software-engineering machine-learning quantitative-finance .claude/skills/
```

### 2. Weryfikacja
```bash
ls ~/.claude/skills/
# Powinieneś zobaczyć:
# python-programming/
# software-engineering/
# machine-learning/
# quantitative-finance/
```

### 3. Użycie

Skills są **automatycznie** wykrywane przez Claude:

```bash
# Otwórz Claude Code
claude-code

# Po prostu zadaj pytanie - Claude sam wybierze odpowiedni Skill
"Stwórz REST API w Python używając FastAPI i type hints"
→ Claude automatycznie użyje: python-programming + software-engineering

"Zbuduj backtesting framework dla mean reversion strategy"
→ Claude automatycznie użyje: quantitative-finance + python-programming
```

---

## 📖 Kiedy Claude używa którego Skill?

| Pytanie/Zadanie | Użyte Skills |
|----------------|--------------|
| "Napisz funkcję Python z type hints" | `python-programming` |
| "Zaprojektuj microservices architecture" | `software-engineering` |
| "Stwórz model klasyfikacji z XGBoost" | `machine-learning` |
| "Zbuduj trading system z order management" | `quantitative-finance` |
| "Backtest momentum strategy" | `quantitative-finance` + `python-programming` |
| "Optymalizuj portfolio używając mean-variance" | `quantitative-finance` |
| "Zredukuj P99 latencji feed-handlera <50 µs" | `senior-quantitative-developer` |
| "Walk-forward dla momentum z kosztami i leakage checks" | `senior-quantitative-researcher` |
| "Dashboard TCA i korekty execution (vol regime)" | `senior-systematic-trader` |
| "Przegląd attribution i korekty sizingu/hedgingu" | `senior-quantitative-trader` |
| "Deploy ML model do produkcji" | `machine-learning` + `software-engineering` |
| "Factor analysis dla trading strategy" | `quantitative-finance` + `machine-learning` |

---

## 💡 Przykłady realnych użyć

### Przykład 1: Quantitative Trading System
```
Prompt: "Stwórz kompletny trading system:
1. Mean reversion strategy z statistical tests
2. Backtesting framework z realistic costs
3. Risk management z Kelly criterion
4. Order management system
5. Production deployment z monitoring"

Claude użyje:
✓ quantitative-finance (strategy, backtesting, risk, OMS)
✓ python-programming (clean code, type hints, async)
✓ software-engineering (architecture, deployment, monitoring)
```

### Przykład 2: ML Trading Strategy
```
Prompt: "Zbuduj ML-based trading strategy:
- Feature engineering z market data
- XGBoost model dla signal generation
- Walk-forward optimization
- Portfolio construction
- Performance attribution"

Claude użyje:
✓ quantitative-finance (trading framework, portfolio optimization)
✓ machine-learning (ML model, feature engineering)
✓ python-programming (implementation)
```

### Przykład 3: High-Frequency Trading Infrastructure
```
Prompt: "Design HFT system z:
- Low-latency market data feed
- Order book analysis
- Optimal execution (TWAP/VWAP)
- Market impact modeling
- Real-time risk monitoring"

Claude użyje:
✓ quantitative-finance (microstructure, execution, risk)
✓ software-engineering (low-latency architecture)
✓ python-programming (high-performance code)
```

---

## 📊 Statystyki Skills

| Skill | Rozmiar | Sekcje | Przykłady kodu | Główne tematy |
|-------|---------|--------|----------------|---------------|
| Python Programming | 12 KB | 15 | 30+ | PEP 8, testing, async, patterns |
| Software Engineering | 28 KB | 20 | 40+ | SOLID, architecture, CI/CD |
| Machine Learning | 27 KB | 25 | 50+ | ML workflow, PyTorch, MLOps |
| Quantitative Finance | 58 KB | 18 | 60+ | Trading, backtesting, portfolio |
| **Senior Quant Developer** | **3 KB** | **9** | **3** | **Low-latency, observability** |
| **Senior Quant Researcher** | **2.5 KB** | **9** | **3** | **Alpha research, validation** |
| **Senior Systematic Trader** | **2 KB** | **9** | **3** | **TCA, execution, governance** |
| **Senior Quant Trader** | **2 KB** | **9** | **3** | **Portfolio, attribution, KPIs** |
| **TOTAL** | **~135 KB** | **~105** | **~190+** | **8 Skills** |

### ✅ Zgodność z Anthropic Skills Framework:
- **Progressive disclosure**: metadata → instructions → resources (load only what's needed)
- **Composable**: Skills automatycznie współpracują (np. quant-researcher + machine-learning)
- **Portable**: ten sam format działa w Claude.ai, Claude Code i API
- **Efficient**: frontmatter limits (name ≤64, description ≤1024) zapewniają szybkie ładowanie
- **Structured**: sekcje Instructions, Tools, Examples, References zgodne z best practices

---

## 🎯 Use Cases dla Quantitative Finance Skill

### Quantitative Developer
```
"Zaimplementuj order management system z pre-trade risk checks"
"Stwórz high-performance market data processor"
"Build execution optimizer z TWAP/VWAP strategies"
```

### Quantitative Researcher
```
"Conduct statistical analysis na mean reversion signal"
"Implement walk-forward optimization framework"
"Test for cointegration między assets pairs"
```

### Systematic Trader
```
"Zaprojektuj momentum strategy z position sizing"
"Implement risk-parity portfolio construction"
"Build market regime detection system"
```

### Portfolio Manager
```
"Optimize portfolio używając Black-Litterman model"
"Calculate factor attribution dla returns"
"Implement dynamic rebalancing strategy"
```

---

## 📄 Pełna dokumentacja

Zobacz szczegółowe instrukcje:
- **INSTALLATION.md** - Instalacja i troubleshooting
- **EXAMPLES.md** - Praktyczne przykłady promptów i scenariusze
- **COMPLIANCE.md** - Pełna weryfikacja zgodności z oficjalną dokumentacją Anthropic
- **VERIFICATION.md** - Finalna weryfikacja jakości i production readiness

---

## 📚 Oficjalne zasoby

### Anthropic Skills Documentation:
- [Introducing Agent Skills](https://www.anthropic.com/news/skills) - główne ogłoszenie
- [Skills Overview](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) - pełna dokumentacja
- [Skills Quickstart](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/quickstart) - szybki start
- [Skills Best Practices](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices) - najlepsze praktyki
- [Skills Cookbook](https://github.com/anthropics/claude-cookbooks/tree/main/skills) - przykłady

### Top Quant Firms (źródła dla role-specific Skills):
- [Citadel Securities](https://www.citadelsecurities.com/careers/)
- [Jane Street](https://www.janestreet.com/join-jane-street/)
- [Hudson River Trading](https://www.hudsonrivertrading.com/careers/)
- [Jump Trading](https://www.jumptrading.com/careers/)
- [Two Sigma](https://www.twosigma.com/careers/)
- [Point72 (Cubist)](https://www.point72.com/careers/)
- [PDT Partners](https://www.pdtpartners.com/careers/)
- [Arrowstreet Capital](https://www.arrowstreetcapital.com/careers/)
- [Renaissance Technologies](https://www.rentec.com/)
- [Radix Trading](https://www.radix-trading.com/careers/)
- [TGS Management](https://www.tgsmanagement.com/careers/)

---

## 🎉 Ready to use!

Twoje Skills są gotowe. Claude będzie automatycznie używał ich gdy zauważy odpowiedni kontekst w Twoich pytaniach.

**Szybki test - podstawowy:**
```bash
# W Claude Code
"Napisz backtesting framework z transaction costs i slippage modeling"
# Claude automatycznie użyje quantitative-finance skill! 🚀
```

**Szybki test - role-specific:**
```bash
# Test dla Senior Quantitative Developer
"Reduce P99 latency of market data handler to < 100 µs and show profiling results"
# Claude użyje: senior-quantitative-developer

# Test dla Senior Quantitative Researcher
"Conduct walk-forward validation for mean-reversion with bias checks"
# Claude użyje: senior-quantitative-researcher

# Test dla Senior Systematic Trader
"Build TCA dashboard and optimize execution parameters"
# Claude użyje: senior-systematic-trader

# Test dla Senior Quantitative Trader
"Prepare attribution analysis and recommend portfolio adjustments"
# Claude użyje: senior-quantitative-trader
```

## ✅ Checklist zgodności z Anthropic

Wszystkie Skills w tym pakiecie spełniają oficjalne wymagania:

- [x] **Frontmatter limits**: name ≤64, description ≤1024 znaków
- [x] **Structured format**: Instructions → Tools → Examples → References
- [x] **Progressive disclosure**: metadata zawsze, content on-demand
- [x] **Composability**: Skills współpracują automatycznie
- [x] **Portability**: format kompatybilny z Claude.ai, Code, API
- [x] **Quality**: oparte na najlepszych firmach (Citadel, Jane Street, HRT, etc.)
- [x] **Documentation**: pełne linki do źródeł i oficjalnej dokumentacji Anthropic

---

**Potrzebujesz pomocy?** Sprawdź `INSTALLATION.md` lub zapytaj Claude! 💬
