# Claude Code Skills - Instrukcje Instalacji

## 📦 Zawartość

Otrzymałeś 3 profesjonalne Skills dla Claude Code:

1. **python-programming** - Kompleksowy przewodnik po programowaniu w Python
2. **software-engineering** - Architektura, wzorce projektowe, SOLID, best practices
3. **machine-learning** - ML/DL, algorytmy, training, deployment, MLOps

## 🚀 Instalacja

### Metoda 1: Personal Skills (Zalecana)

Skills będą dostępne we wszystkich projektach:

```bash
# Skopiuj foldery do katalogu personal skills
cp -r python-programming ~/.claude/skills/
cp -r software-engineering ~/.claude/skills/
cp -r machine-learning ~/.claude/skills/
```

### Metoda 2: Project Skills

Skills będą dostępne tylko w konkretnym projekcie:

```bash
# W katalogu swojego projektu
mkdir -p .claude/skills
cp -r python-programming .claude/skills/
cp -r software-engineering .claude/skills/
cp -r machine-learning .claude/skills/
```

### Metoda 3: Plugin Marketplace (Opcjonalna)

Jeśli chcesz udostępnić Skills jako plugin:

```bash
# Utwórz repozytorium Git
git init my-skills-repo
cd my-skills-repo
cp -r python-programming software-engineering machine-learning .
git add .
git commit -m "Initial commit: Python, Software Engineering, ML skills"

# W Claude Code zarejestruj jako plugin
# (wymaga uruchomienia Claude Code)
```

## 📖 Jak używać Skills

### Automatyczne wykrywanie

Skills są **automatycznie** wywoływane przez Claude, gdy:

1. **Python Programming Skill** - Gdy piszesz kod w Python, refaktoryzujesz, debugujesz
   ```
   "Napisz funkcję w Python do przetwarzania danych CSV"
   "Zrefaktoruj ten kod zgodnie z PEP 8"
   ```

2. **Software Engineering Skill** - Gdy projektujesz systemy, robisz code review
   ```
   "Zaprojektuj architekturę mikrousług dla e-commerce"
   "Review tego kodu pod kątem SOLID principles"
   ```

3. **Machine Learning Skill** - Gdy budujesz modele ML, trenujesz sieci
   ```
   "Stwórz model klasyfikacji obrazów używając PyTorch"
   "Jak wdrożyć model ML do produkcji z FastAPI?"
   ```

### Przykłady użycia

```bash
# Claude automatycznie użyje odpowiedniego Skill
"Pomóż mi zbudować REST API w FastAPI używając best practices"
→ Claude użyje: software-engineering + python-programming

"Stwórz pipeline ML z preprocessing, training i deployment"
→ Claude użyje: machine-learning + python-programming

"Zaprojektuj system event-driven z microservices"
→ Claude użyje: software-engineering
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
```

Każdy folder powinien zawierać plik `SKILL.md`:

```bash
cat ~/.claude/skills/python-programming/SKILL.md | head -20
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
"Użyj python-programming skill żeby sprawdzić czy ten kod jest zgodny z PEP 8"
```

### 2. Łącz wiele Skills
```
"Używając software-engineering i machine-learning skills, 
zaprojektuj architekturę systemu rekomendacji"
```

### 3. Sprawdź co Claude załadował
Claude pokazuje w "chain of thought" które Skills użył:
```
Reading /home/user/.claude/skills/python-programming/SKILL.md
```

## 📋 Checklist pierwszego uruchomienia

- [ ] Skopiowałem foldery do `~/.claude/skills/` lub `.claude/skills/`
- [ ] Sprawdziłem że pliki `SKILL.md` istnieją
- [ ] Zrestartowałem Claude Code
- [ ] Przetestowałem działanie: "Napisz funkcję w Python do..."
- [ ] Claude automatycznie użył odpowiedniego Skill

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
cat ~/.claude/skills/python-programming/SKILL.md | head -5

# Powinno wyglądać tak:
# ---
# name: python-programming
# description: Expert Python programming...
# ---
```

### Problem: Skill się nie ładuje

**Rozwiązanie:**
1. Sprawdź uprawnienia do odczytu: `ls -la ~/.claude/skills/`
2. Sprawdź czy folder zawiera `SKILL.md`
3. Zobacz logi Claude Code

## 📚 Dodatkowe zasoby

- [Oficjalna dokumentacja Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- [Anthropic Skills GitHub](https://github.com/anthropics/skills)
- [Skills Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

## 🎉 Gotowe!

Twoje Skills są zainstalowane i gotowe do użycia. Claude będzie automatycznie używał ich gdy wykryje odpowiedni kontekst w Twoich pytaniach.

**Przykład pełnego workflow:**

```bash
# 1. Zainstaluj Skills
cp -r python-programming ~/.claude/skills/

# 2. Otwórz Claude Code
claude-code

# 3. Zapytaj Claude
"Stwórz REST API w FastAPI z autentykacją JWT, 
używając best practices i type hints"

# 4. Claude automatycznie użyje:
#    - python-programming skill (type hints, FastAPI)
#    - software-engineering skill (API design, security)
```

---

**Pytania?** Sprawdź dokumentację Skills lub zapytaj Claude o pomoc! 🚀
