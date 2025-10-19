# Claude Code Skills - Instrukcje Instalacji

## 📦 Zawartość

Otrzymałeś 4 profesjonalne Skills dla Claude Code:

1. **python-programming** - Kompleksowy przewodnik po programowaniu w Python
2. **software-engineering** - Architektura, wzorce projektowe, SOLID, best practices
3. **machine-learning** - ML/DL, algorytmy, training, deployment, MLOps
4. **quantitative-finance** - Trading algorithms, quant research, systematic trading

## 🚀 Instalacja

### Metoda 1: Personal Skills (Zalecana)

Skills będą dostępne we wszystkich projektach:

```bash
# Skopiuj foldery do katalogu personal skills
cp -r python-programming ~/.claude/skills/
cp -r software-engineering ~/.claude/skills/
cp -r machine-learning ~/.claude/skills/
cp -r quantitative-finance ~/.claude/skills/
```

### Metoda 2: Project Skills

Skills będą dostępne tylko w konkretnym projekcie:

```bash
# W katalogu swojego projektu
mkdir -p .claude/skills
cp -r python-programming .claude/skills/
cp -r software-engineering .claude/skills/
cp -r machine-learning .claude/skills/
cp -r quantitative-finance .claude/skills/
```

### Metoda 3: Wybiórcza instalacja

Jeśli potrzebujesz tylko niektórych Skills:

```bash
# Tylko Python i Quant Finance
cp -r python-programming quantitative-finance ~/.claude/skills/

# Tylko ML i Software Engineering
cp -r machine-learning software-engineering ~/.claude/skills/
```

## 📖 Jak używać Skills

### Automatyczne wykrywanie

Skills są **automatycznie** wywoływane przez Claude, gdy:

1. **Python Programming Skill** - Gdy piszesz kod w Python
   ```
   "Napisz funkcję w Python do przetwarzania danych CSV"
   "Zrefaktoruj ten kod zgodnie z PEP 8"
   ```

2. **Software Engineering Skill** - Gdy projektujesz systemy
   ```
   "Zaprojektuj architekturę mikrousług dla e-commerce"
   "Review tego kodu pod kątem SOLID principles"
   ```

3. **Machine Learning Skill** - Gdy budujesz modele ML
   ```
   "Stwórz model klasyfikacji obrazów używając PyTorch"
   "Jak wdrożyć model ML do produkcji z FastAPI?"
   ```

4. **Quantitative Finance Skill** - Gdy pracujesz z trading i finance
   ```
   "Zbuduj backtesting framework dla mean reversion strategy"
   "Implement order management system z risk checks"
   "Optimize portfolio używając Black-Litterman"
   ```

### Przykłady użycia

```bash
# Claude automatycznie użyje odpowiedniego Skill
"Pomóż mi zbudować REST API w FastAPI używając best practices"
→ Claude użyje: software-engineering + python-programming

"Stwórz ML trading strategy z feature engineering i backtesting"
→ Claude użyje: quantitative-finance + machine-learning + python-programming

"Design high-frequency trading system z low-latency architecture"
→ Claude użyje: quantitative-finance + software-engineering
```

## ✅ Weryfikacja instalacji

Sprawdź czy Skills zostały zainstalowane:

```bash
# Personal skills
ls ~/.claude/skills/

# Project skills
ls .claude/skills/

# Powinieneś zobaczyć:
# python-programming/
# software-engineering/
# machine-learning/
# quantitative-finance/
```

Każdy folder powinien zawierać plik `SKILL.md`:

```bash
cat ~/.claude/skills/quantitative-finance/SKILL.md | head -20
```

## 🔄 Aktualizacja Skills

Aby zaktualizować Skill:

1. Edytuj plik `SKILL.md` w odpowiednim folderze
2. Zapisz zmiany
3. Zrestartuj Claude Code (jeśli jest uruchomiony)

## 🎯 Struktura Skill

Każdy Skill zawiera:

```yaml
---
name: skill-name
description: Co robi skill i kiedy go używać
---

# Tytuł Skill

## Sekcje z instrukcjami
- Szczegółowe wskazówki
- Przykłady kodu
- Best practices
- Najczęstsze błędy
```

## 💡 Tips & Tricks

### 1. Wymuś użycie konkretnego Skill
```
"Użyj quantitative-finance skill żeby zbudować backtesting framework"
```

### 2. Łącz wiele Skills
```
"Używając quantitative-finance, machine-learning i python-programming skills, 
stwórz ML-based trading strategy z deployment do produkcji"
```

### 3. Sprawdź co Claude załadował
Claude pokazuje w "chain of thought" które Skills użył:
```
Reading /home/user/.claude/skills/quantitative-finance/SKILL.md
```

### 4. Dostosuj Skills do swojego use case
```bash
# Edytuj Skill aby dodać własne strategie
vim ~/.claude/skills/quantitative-finance/SKILL.md

# Dodaj sekcję z twoimi specyficznymi wymaganiami
## My Custom Trading Strategies

### Strategy 1: Custom Mean Reversion
[your content]
```

## 📋 Checklist pierwszego uruchomienia

- [ ] Skopiowałem foldery do `~/.claude/skills/` lub `.claude/skills/`
- [ ] Sprawdziłem że pliki `SKILL.md` istnieją
- [ ] Zrestartowałem Claude Code
- [ ] Przetestowałem Python Skill: "Napisz funkcję Python z type hints"
- [ ] Przetestowałem Quant Skill: "Implement backtesting framework"
- [ ] Claude automatycznie użył odpowiednich Skills

## 🆘 Troubleshooting

### Problem: Claude nie używa Skills

**Rozwiązanie:**
1. Sprawdź lokalizację: `ls ~/.claude/skills/`
2. Upewnij się że nazwa pliku to dokładnie `SKILL.md` (wielkie litery)
3. Sprawdź YAML frontmatter (czy jest poprawny)
4. Zrestartuj Claude Code

### Problem: Błąd parsowania YAML

**Rozwiązanie:**
```bash
# Sprawdź poprawność YAML
cat ~/.claude/skills/quantitative-finance/SKILL.md | head -5

# Powinno wyglądać tak:
# ---
# name: quantitative-finance
# description: Expert guidance for quantitative...
# ---
```

### Problem: Skill się nie ładuje

**Rozwiązanie:**
1. Sprawdź uprawnienia do odczytu: `ls -la ~/.claude/skills/`
2. Sprawdź czy folder zawiera `SKILL.md`
3. Zobacz logi Claude Code

### Problem: Za dużo Skills, wolne działanie

**Rozwiązanie:**
Skills używają progressive disclosure - każdy Skill to tylko ~30-50 tokenów.
Nie powinno być problemów z wydajnością, ale możesz:
```bash
# Usuń nieużywane Skills
rm -rf ~/.claude/skills/skill-name

# Lub przenieś do backup
mv ~/.claude/skills/unused-skill ~/claude-skills-backup/
```

## 🎓 Use Case - Quantitative Trading

Przykładowy workflow dla quant developera:

```bash
# 1. Zainstaluj Skills
cp -r python-programming quantitative-finance ~/.claude/skills/

# 2. Otwórz Claude Code
claude-code

# 3. Research phase
"Conduct statistical analysis na mean reversion signal 
używając ADF test i half-life calculation"
→ Claude użyje: quantitative-finance

# 4. Development phase
"Implement backtesting framework z:
- Realistic transaction costs
- Slippage modeling
- Position sizing z Kelly criterion
- Walk-forward optimization"
→ Claude użyje: quantitative-finance + python-programming

# 5. Production phase
"Build production trading system z:
- Order management
- Risk monitoring
- Real-time alerts
- Performance tracking"
→ Claude użyje: quantitative-finance + software-engineering

# 6. ML Enhancement
"Add ML layer z:
- Feature engineering z technical indicators
- XGBoost model dla signal generation
- Model monitoring i retraining"
→ Claude użyje: quantitative-finance + machine-learning
```

## 📚 Dodatkowe zasoby

- [Oficjalna dokumentacja Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- [Anthropic Skills GitHub](https://github.com/anthropics/skills)
- [Skills Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

## 🎉 Gotowe!

Twoje Skills są zainstalowane i gotowe do użycia. Claude będzie automatycznie używał ich gdy wykryje odpowiedni kontekst w Twoich pytaniach.

**Przykład pełnego workflow dla quantitative trading:**

```bash
# 1. Zainstaluj Skills
cp -r python-programming software-engineering machine-learning quantitative-finance ~/.claude/skills/

# 2. Otwórz Claude Code
claude-code

# 3. Zapytaj Claude
"Stwórz kompletny systematic trading system:
- Mean reversion strategy z statistical tests
- Professional backtesting z realistic assumptions  
- ML enhancement z feature engineering
- Risk management z Kelly criterion
- Production deployment z monitoring"

# 4. Claude automatycznie użyje wszystkie 4 Skills:
#    - quantitative-finance (trading, backtesting, risk)
#    - machine-learning (ML model, features)
#    - python-programming (clean code, type hints)
#    - software-engineering (architecture, deployment)
```

---

**Pytania?** Sprawdź dokumentację Skills lub zapytaj Claude o pomoc! 🚀
