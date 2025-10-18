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

### Przykład 4: Refaktoryzacja
```
Prompt: "Zrefaktoruj ten kod zgodnie z PEP 8 i dodaj 
type hints. Popraw też nazewnictwo zmiennych."

Claude użyje: python-programming skill
- Zastosuje PEP 8 style guide
- Doda type hints
- Użyje meaningful names
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
- Uwzględni service discovery
```

### Przykład 3: API Design
```
Prompt: "Zaprojektuj RESTful API dla systemu blogowego 
z wersjonowaniem i autentykacją"

Claude użyje: software-engineering skill
- Zastosuje REST best practices
- Zaproponuje versioning strategy
- Implementuje authentication/authorization
- Doda proper HTTP status codes
```

### Przykład 4: Testing Strategy
```
Prompt: "Jaka powinna być strategia testowania dla 
naszego projektu? Mamy microservices z bazą danych."

Claude użyje: software-engineering skill
- Zaproponuje Test Pyramid
- Rozróżni Unit/Integration/E2E
- Użyje contract testing dla microservices
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
- Model evaluation metrics
```

### Przykład 2: Deep Learning
```
Prompt: "Potrzebuję CNN w PyTorch do klasyfikacji obrazów 
z data augmentation i transfer learning"

Claude użyje: machine-learning skill
- Stworzy CNN architecture
- Implementuje data augmentation
- Użyje transfer learning (ResNet/VGG)
- Doda training loop z early stopping
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
- A/B testing strategy
```

### Przykład 4: Feature Engineering
```
Prompt: "Mam dataset z datami i kategoriami. 
Jak zrobić feature engineering?"

Claude użyje: machine-learning skill
- Date features (year, month, day_of_week)
- Categorical encoding (one-hot, target)
- Feature selection methods
- Interaction features
```

---

## 🎯 Kombinacje Skills

### Przykład 1: Full Stack ML Project
```
Prompt: "Stwórz kompletny projekt:
1. Model ML w Python
2. REST API z FastAPI
3. Deployment z Docker
4. CI/CD pipeline
5. Monitoring"

Claude użyje wszystkie 3 skills:
✓ machine-learning: model, training, evaluation
✓ python-programming: clean code, type hints, async
✓ software-engineering: API design, CI/CD, monitoring
```

### Przykład 2: Production-Ready System
```
Prompt: "Build production system dla real-time predictions z:
- High availability
- Load balancing
- Caching
- Monitoring
- Logging"

Claude użyje:
✓ software-engineering: architecture, HA, monitoring
✓ python-programming: implementation, logging
✓ machine-learning: model serving, optimization
```

### Przykład 3: Code Review
```
Prompt: "Review tego ML projektu pod kątem:
- Code quality
- Architecture
- ML best practices"

Claude użyje wszystkie 3 skills:
✓ python-programming: PEP 8, type hints, testing
✓ software-engineering: SOLID, patterns, structure
✓ machine-learning: ML practices, evaluation
```

---

## 💡 Pro Tips

### Tip 1: Bądź konkretny
```
❌ "Pomóż mi z kodem"
✅ "Napisz funkcję Python z type hints do parsowania CSV"
```

### Tip 2: Wymień technologie
```
❌ "Stwórz model ML"
✅ "Stwórz model klasyfikacji używając XGBoost z hyperparameter tuning"
```

### Tip 3: Określ kontekst
```
❌ "Zaprojektuj system"
✅ "Zaprojektuj event-driven microservices dla e-commerce 
     z payment, inventory i order services"
```

### Tip 4: Wymuś konkretny Skill (jeśli potrzebne)
```
"Używając python-programming skill, zrefaktoruj ten kod 
zgodnie z PEP 8 i dodaj type hints"
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

### Test 4: Multiple Skills
```
"Build complete ML system:
- XGBoost model
- FastAPI endpoint  
- Docker deployment
- Prometheus monitoring
- Structured logging"
```
**Oczekiwane:** wszystkie 3 skills

---

## 📊 Matryca promptów

| Chcę... | Używaj słów kluczowych | Skill |
|---------|------------------------|-------|
| Napisać kod Python | "function", "type hints", "pytest" | Python |
| Zaprojektować system | "architecture", "microservices", "design" | SWE |
| Zbudować model ML | "model", "training", "evaluation" | ML |
| Review kodu | "SOLID", "clean code", "refactor" | SWE |
| Deploy model | "deployment", "docker", "API" | ML + SWE |
| Optymalizować kod | "performance", "async", "profiling" | Python |
| Testować | "pytest", "mocking", "integration" | Python/SWE |

---

## 🎓 Learning Path

### Beginner
1. Zacznij od pojedynczych Skills
2. Używaj prostych, konkretnych promptów
3. Analizuj wygenerowany kod

### Intermediate  
1. Łącz 2-3 Skills w jednym zadaniu
2. Prosij o best practices
3. Implementuj w realnych projektach

### Advanced
1. Prosij o review kompleksowych systemów
2. Łącz wszystkie Skills
3. Buduj production-ready solutions

---

Powodzenia! 🚀
