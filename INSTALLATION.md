# Claude Code Skills - Instrukcje Instalacji

## 📦 Zawartość

Otrzymałeś 4 profesjonalne Skills dla Claude Code:

1. **python-programming** - Kompleksowy przewodnik po programowaniu w Python
2. **software-engineering** - Architektura, wzorce projektowe, SOLID, best practices
3. **machine-learning** - ML/DL, algorytmy, training, deployment, MLOps
4. **quantitative-finance** - Trading algorithms, quant research, systematic trading

oraz 4 role-specific Skills dla quantitative trading:

- **senior-quantitative-developer** (w `quantitative-finance/`)
- **senior-quantitative-researcher** (w `quantitative-finance/`)
- **senior-systematic-trader** (w `quantitative-finance/`)
- **senior-quantitative-trader** (w `quantitative-finance/`)

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

Role-specific Skills (opcjonalnie możesz skopiować tylko pliki *.SKILL.md):

```bash
# Przykład: zainstaluj same role-specific Skills do katalogu Quant
mkdir -p ~/.claude/skills/quantitative-finance
cp quantitative-finance/Senior-Quantitative-Developer.SKILL.md ~/.claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Quantitative-Researcher.SKILL.md ~/.claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Systematic-Trader.SKILL.md ~/.claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Quantitative-Trader.SKILL.md ~/.claude/skills/quantitative-finance/
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

Role-specific Skills (tylko pliki, w obrębie projektu):

```bash
mkdir -p .claude/skills/quantitative-finance
cp quantitative-finance/Senior-Quantitative-Developer.SKILL.md .claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Quantitative-Researcher.SKILL.md .claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Systematic-Trader.SKILL.md .claude/skills/quantitative-finance/
cp quantitative-finance/Senior-Quantitative-Trader.SKILL.md .claude/skills/quantitative-finance/
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
#   Senior-Quantitative-Developer.SKILL.md
#   Senior-Quantitative-Researcher.SKILL.md
#   Senior-Systematic-Trader.SKILL.md
#   Senior-Quantitative-Trader.SKILL.md
```

Każdy folder powinien zawierać plik `SKILL.md`:

```bash
cat ~/.claude/skills/quantitative-finance/SKILL.md | head -20
cat ~/.claude/skills/quantitative-finance/Senior-Quantitative-Developer.SKILL.md | head -20
```

## 🔄 Aktualizacja Skills

Aby zaktualizować Skill:

1. Edytuj plik `SKILL.md` w odpowiednim folderze
2. Zapisz zmiany
3. Zrestartuj Claude Code (jeśli jest uruchomiony)

## 🎯 Struktura Skill

Każdy Skill zawiera zgodnie z oficjalną specyfikacją Anthropic:

```yaml
---
name: skill-name                    # max 64 znaki
description: Co robi skill          # max 1024 znaki
---

# Tytuł Skill

## Instructions                     # Główna sekcja instrukcji
### When to Use                     # Kiedy użyć
### Expected Outcomes               # Mierzalne rezultaty
### Required Inputs                 # Wymagane dane wejściowe
### Implementation Steps            # Krok po kroku
### Quality Checklist               # Kontrola jakości
### Metrics & Validation            # Metryki sukcesu

## Tools & Technologies             # Stack technologiczny
## Examples                         # Przykłady użycia
## References                       # Linki do dokumentacji
```

**Zgodność z Anthropic:**
- ✅ Progressive disclosure (ładuje tylko to, co potrzebne)
- ✅ Composable (Skills współpracują ze sobą)
- ✅ Portable (działa w Claude.ai, Claude Code, API)
- ✅ Efficient (minimalne obciążenie)

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
3. Sprawdź YAML frontmatter (limity: name ≤64, description ≤1024 znaków)
4. Zrestartuj Claude Code

### Problem: Błąd parsowania YAML

**Rozwiązanie:**
```bash
# Sprawdź poprawność YAML
cat ~/.claude/skills/quantitative-finance/SKILL.md | head -5

# Powinno wyglądać tak:
# ---
# name: quantitative-finance
# description: Expert guidance for algorithmic trading...
# ---
```

### Problem: Role-specific Skills nie działają

**Rozwiązanie:**
```bash
# Sprawdź czy pliki są we właściwej lokalizacji
ls ~/.claude/skills/quantitative-finance/Senior-*.SKILL.md

# Upewnij się że frontmatter jest poprawny
cat ~/.claude/skills/quantitative-finance/Senior-Quantitative-Developer.SKILL.md | head -5
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

### Oficjalna dokumentacja Anthropic:
- [Introducing Agent Skills](https://www.anthropic.com/news/skills) - główne ogłoszenie
- [Skills Overview](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) - dokumentacja techniczna
- [Skills Quickstart](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/quickstart) - szybki start
- [Skills Best Practices](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices) - najlepsze praktyki
- [Skills Cookbook](https://github.com/anthropics/claude-cookbooks/tree/main/skills) - przykłady

### Źródła dla role-specific Skills (top firms):
- [Citadel Securities - Careers](https://www.citadelsecurities.com/careers/)
- [Jane Street - Join](https://www.janestreet.com/join-jane-street/)
- [Hudson River Trading - Careers](https://www.hudsonrivertrading.com/careers/)
- [Jump Trading - Careers](https://www.jumptrading.com/careers/)
- [Two Sigma - Careers](https://www.twosigma.com/careers/)
- [Point72 (Cubist) - Careers](https://www.point72.com/careers/)
- [PDT Partners - Careers](https://www.pdtpartners.com/careers/)
- [Arrowstreet Capital - Careers](https://www.arrowstreetcapital.com/careers/)

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
