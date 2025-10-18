# Claude Code Skills - Python, Software Engineering, Machine Learning

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
- ✅ Dependencies management (pyproject.toml, requirements.txt)
- ✅ Structured logging (JSON formatting)
- ✅ Best practices checklist
- ✅ Common pitfalls to avoid
- ✅ Essential development tools (Black, Ruff, mypy, pytest)

**Przykład użycia:**
```
"Napisz funkcję w Python z type hints do przetwarzania CSV"
"Zrefaktoruj ten kod zgodnie z PEP 8"
"Jak stworzyć custom exception w Python?"
```

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
- ✅ Code review checklist
- ✅ API design (RESTful, versioning)
- ✅ Database design (normalization, indexing, migrations)
- ✅ Monitoring & Observability (logging, metrics, health checks)
- ✅ Security best practices (validation, authentication, authorization)
- ✅ Performance optimization (caching, query optimization)
- ✅ Documentation standards

**Przykład użycia:**
```
"Zaprojektuj microservices architecture dla e-commerce"
"Review tego kodu pod kątem SOLID principles"
"Jak zaimplementować CQRS pattern?"
```

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
- ✅ Cross-validation strategies
- ✅ Hyperparameter tuning (Grid, Random, Bayesian)
- ✅ Deep Learning z PyTorch (Neural Networks, CNN, RNN/LSTM)
- ✅ Transfer Learning
- ✅ MLOps (serialization, versioning, monitoring)
- ✅ Model deployment (FastAPI, Docker, containerization)
- ✅ Model interpretability (SHAP, LIME)
- ✅ Production best practices
- ✅ Data drift detection

**Przykład użycia:**
```
"Stwórz model klasyfikacji obrazów używając CNN w PyTorch"
"Jak wdrożyć model ML do produkcji z FastAPI i Docker?"
"Potrzebuję pipeline z preprocessing, training i monitoring"
```

---

## 🚀 Szybki start

### 1. Instalacja (wybierz jedną metodę)

**Personal Skills** (dostępne wszędzie):
```bash
cp -r python-programming software-engineering machine-learning ~/.claude/skills/
```

**Project Skills** (tylko w projekcie):
```bash
mkdir -p .claude/skills
cp -r python-programming software-engineering machine-learning .claude/skills/
```

### 2. Weryfikacja
```bash
ls ~/.claude/skills/
# Powinieneś zobaczyć:
# python-programming/
# software-engineering/
# machine-learning/
```

### 3. Użycie

Skills są **automatycznie** wykrywane przez Claude:

```bash
# Otwórz Claude Code
claude-code

# Po prostu zadaj pytanie - Claude sam wybierze odpowiedni Skill
"Stwórz REST API w Python używając FastAPI i type hints"
→ Claude automatycznie użyje: python-programming + software-engineering

"Zbuduj model ML do predykcji churn rate i wdróż go"
→ Claude automatycznie użyje: machine-learning + python-programming
```

---

## 📖 Kiedy Claude używa którego Skill?

| Pytanie/Zadanie | Użyte Skills |
|----------------|--------------|
| "Napisz funkcję Python z type hints" | `python-programming` |
| "Zaprojektuj microservices architecture" | `software-engineering` |
| "Stwórz model klasyfikacji z XGBoost" | `machine-learning` |
| "Build REST API with authentication" | `python-programming` + `software-engineering` |
| "ML pipeline z deployment na produkcję" | `machine-learning` + `python-programming` + `software-engineering` |
| "Review kodu pod kątem SOLID" | `software-engineering` |
| "Optymalizuj ten kod Python" | `python-programming` |
| "Preprocessing danych do ML" | `machine-learning` |

---

## 💡 Przykłady realnych użyć

### Przykład 1: Full-Stack ML Project
```
Prompt: "Stwórz kompletny projekt ML:
1. Preprocessing danych z CSV
2. Training modelu XGBoost
3. REST API w FastAPI
4. Docker deployment
5. Monitoring w produkcji"

Claude użyje:
✓ machine-learning (preprocessing, training, monitoring)
✓ python-programming (clean code, type hints, async)
✓ software-engineering (API design, deployment, best practices)
```

### Przykład 2: Microservices Architecture
```
Prompt: "Zaprojektuj event-driven microservices 
architecture dla systemu zamówień z:
- Event Bus
- CQRS pattern
- API Gateway
- Service discovery"

Claude użyje:
✓ software-engineering (architecture patterns, CQRS, API design)
✓ python-programming (implementation details)
```

### Przykład 3: Production ML Pipeline
```
Prompt: "Build production-ready ML pipeline:
- Data validation
- Feature store
- Model versioning with MLflow
- A/B testing
- Drift monitoring"

Claude użyje:
✓ machine-learning (pipeline, MLOps, monitoring)
✓ software-engineering (architecture, CI/CD)
✓ python-programming (implementation)
```

---

## 📄 Pełna dokumentacja

Zobacz szczegółowe instrukcje instalacji w pliku [INSTALLATION.md](./INSTALLATION.md)

---

## 📚 Oficjalne zasoby

- [Claude Skills Documentation](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- [Anthropic Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Anthropic Skills GitHub](https://github.com/anthropics/skills)

---

## 🎉 Ready to use!

Twoje Skills są gotowe. Claude będzie automatycznie używał ich gdy zauważy odpowiedni kontekst w Twoich pytaniach.

**Szybki test:**
```bash
# W Claude Code
"Napisz funkcję Python z type hints do przetwarzania JSON"

# Claude automatycznie użyje python-programming skill! 🚀
```

---

**Potrzebujesz pomocy?** Sprawdź `INSTALLATION.md` lub zapytaj Claude! 💬
