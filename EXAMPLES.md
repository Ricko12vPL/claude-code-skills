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

Powodzenia w budowaniu trading systems! 🚀📈
