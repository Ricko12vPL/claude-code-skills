# Szczegółowa Lista Promptów dla Claude Code Skills

## Spis Treści

1. [Programowanie w Pythonie (50 promptów)](#1-programowanie-w-pythonie-prompty-50)
2. [Inżynieria Oprogramowania (50 promptów)](#2-inżynieria-oprogramowania-prompty-50)
3. [Uczenie Maszynowe (50 promptów)](#3-uczenie-maszynowe-prompty-50)
4. [Finanse Kwantatywne (50 promptów)](#4-finanse-kwantatywne-prompty-50)
5. [Senior Quantitative Developer (50 promptów)](#5-senior-quantitative-developer-prompty-50)
6. [Senior Quantitative Researcher (50 promptów)](#6-senior-quantitative-researcher-prompty-50)
7. [Senior Systematic Trader (50 promptów)](#7-senior-systematic-trader-prompty-50)
8. [Senior Quantitative Trader (50 promptów)](#8-senior-quantitative-trader-prompty-50)

**Łącznie: 400 promptów zgodnych z najlepszymi praktykami prompt engineering**

---

## 1. Programowanie w Pythonie Prompty (50)

### Kategoria: Jakość Kodu i Styl

### 1.1 Refaktoryzacja Przestrzegania PEP 8
```
Zrefaktoryzuj poniższy kod Python, aby był w 100% zgodny z PEP 8:
- Popraw nazewnictwo (snake_case dla funkcji/zmiennych, PascalCase dla klas)
- Usuń białe znaki i popraw wcięcia
- Dodaj odpowiednie odstępy wokół operatorów
- Ogranicz długość linii do 79 znaków dla kodu, 72 dla docstringów
- Dodaj odpowiednie puste linie między funkcjami/klasami
- Posortuj importy (stdlib, third-party, local)
- Zapewnij spójne cytowanie (preferuj pojedyncze cudzysłowy)
Użyj black, isort i flake8 do weryfikacji.
```

### 1.2 Wyczerpująca Dokumentacja z Docstringami
```
Dodaj kompleksowe docstringi w stylu Google/NumPy do wszystkich modułów, klas i funkcji:
- Krótkie podsumowanie jednolinijkowe
- Szczegółowy opis (jeśli potrzebny)
- Argumenty z typami i opisami
- Zwracane wartości z typami
- Wyjątki, które mogą być zgłoszone
- Przykłady użycia (doctest gdzie stosowne)
- Ostrzeżenia lub uwagi dla użytkowników
Upewnij się, że może być parsowane przez Sphinx/pdoc.
```

### 1.3 Adnotacje Typów i Sprawdzanie mypy
```
Dodaj pełne adnotacje typów do wszystkich sygnatur funkcji:
- Parametry funkcji z typami
- Typy zwracane
- Zmienne tam gdzie niejasne
- Używaj typing.* dla złożonych typów (List, Dict, Optional, Union, etc.)
- Używaj Protocol dla duck typing
- Dodaj Generic[T] dla klas generycznych
Uruchom mypy w trybie strict i rozwiąż wszystkie błędy.
```

### 1.4 Kompleksowe Sprawdzanie Błędów
```
Dodaj solidne sprawdzanie błędów i walidację:
- Waliduj wszystkie dane wejściowe (typy, zakresy, ograniczenia)
- Używaj odpowiednich niestandardowych wyjątków
- Dodaj bloki try-except z określonym obsługiwaniem błędów
- Uwzględnij przypadki brzegowe (None, puste, zero, ujemne)
- Zapewnij komunikaty o błędach z kontekstem
- Użyj assert dla niezmienników w deweloperskim trybie
- Dodaj walidację dla operacji IO (plik nie znaleziony, uprawnienia)
Unikaj gołych klauzul except.
```

### 1.5 Refaktoryzacja Używając List Comprehensions
```
Zrefaktoryzuj pętle do zwięzłych list/dict/set comprehensions:
- Zastąp pętle for z append() list comprehensions
- Użyj dict comprehensions dla transformacji słowników
- Użyj set comprehensions dla unikalnych kolekcji
- Dodaj warunki inline dla filtrowania
- Użyj zagnieżdżonych comprehensions dla zagnieżdżonych pętli (jeśli czytelne)
- Rozważ wyrażenia generatorowe dla dużych sekwencji
Upewnij się, że jest czytelniejsze niż oryginalne pętle.
```

### 1.6 Eliminacja Zduplikowanego Kodu
```
Zidentyfikuj i wyeliminuj duplikację kodu:
- Wyodrębnij wspólną logikę do funkcji pomocniczych
- Utwórz klasy bazowe dla wspólnego zachowania
- Użyj dekoratorów dla cross-cutting concerns
- Zastosuj wzorzec strategii dla wariantowej logiki
- Użyj functools.partial dla częściowego zastosowania
- Rozważ metaprogramowanie dla powtarzalnego kodu boilerplate
Przestrzegaj zasady DRY (Don't Repeat Yourself).
```

### 1.7 Poprawa Nazewnictwa
```
Popraw wszystkie nazwy dla czytelności:
- Funkcje: czasownik + rzeczownik (get_user, calculate_total)
- Klasy: rzeczowniki (UserAccount, DataProcessor)
- Zmienne: opisowe rzeczowniki (total_amount, user_count)
- Stałe: UPPER_SNAKE_CASE
- Prywatne: przedrostek _underscore
- Unikaj jednoliterowych nazw (oprócz i, j w pętlach)
- Unikaj skrótów chyba że standardowe
Priorytet czytelności nad zwięzłością.
```

### 1.8 Optymalizacja Importów
```
Organizuj i optymalizuj instrukcje importu:
- Grupuj: stdlib, third-party, local (oddzielone pustymi liniami)
- Sortuj alfabetycznie w każdej grupie
- Używaj importów absolutnych
- Unikaj import * (jawnie importuj nazwy)
- Używaj importów relatywnych dla pakietów lokalnych
- Przenieś importy na górę (chyba że opóźnione ładowanie)
- Usuń nieużywane importy
Użyj isort do automatycznego formatowania.
```

### 1.9 Dodanie Komentarzy do Złożonej Logiki
```
Dodaj jasne komentarze do nietrywialnego kodu:
- Wyjaśnij "dlaczego", nie "co"
- Udokumentuj założenia i niezmienniki
- Ostrzeż o subtelnych zachowaniach
- Dodaj TODO/FIXME/HACK gdzie stosowne
- Użyj docstringów dla publicznych API
- Użyj inline komentarzy dla złożonych wyrażeń
- Unikaj redundantnych komentarzy
Kod powinien być samodokumentujący gdzie możliwe.
```

### 1.10 Linting i Auto-formatowanie
```
Skonfiguruj i uruchom kompletny pipeline lintingu:
- black dla formatowania kodu
- isort dla sortowania importów
- flake8 dla sprawdzania stylu
- pylint dla głębszej analizy statycznej
- mypy dla sprawdzania typów
- bandit dla skanowania bezpieczeństwa
Skonfiguruj pyproject.toml i napraw wszystkie problemy.
```

### Kategoria: Nowoczesne Funkcje Pythona

### 1.11 Konwersja na f-stringi
```
Zamień wszystkie formatowanie stringów na f-stringi:
- Zastąp % formatowanie → f-stringi
- Zastąp .format() → f-stringi
- Dodaj precyzję dla floatów (f"{value:.2f}")
- Użyj wyrażeń w klamrach {expression}
- Rozważ debugowanie z f"{var=}"
- Obsługuj wielolinijkowe f-stringi dla czytelności
F-stringi są najszybsze i najbardziej czytelne.
```

### 1.12 Data Classes dla Kontenerów Danych
```
Zamień zwykłe klasy na @dataclass:
- Dodaj @dataclass dekorator
- Zdefiniuj atrybuty z adnotacjami typów
- Używaj default/default_factory dla wartości domyślnych
- Dodaj frozen=True dla niemutowalności
- Implementuj __post_init__ dla walidacji
- Używaj field() dla zaawansowanej konfiguracji
- Auto-generuje __init__, __repr__, __eq__
Redukuje kod boilerplate o 70%.
```

### 1.13 Async/Await dla IO-Bound
```
Konwertuj kod IO-bound na async/await:
- Zidentyfikuj operacje IO (sieć, plik, DB)
- Zmień def → async def
- Dodaj await przed wywołaniami async
- Użyj aiohttp dla żądań HTTP
- Użyj aiofiles dla operacji na plikach
- Użyj asyncio.gather() dla współbieżności
- Dodaj proper event loop management
Mierz poprawę wydajności (cel: 3-10x dla wielu IO).
```

### 1.14 Context Managers dla Zarządzania Zasobami
```
Implementuj context managers dla zasobów:
- Użyj @contextmanager dla prostych przypadków
- Implementuj __enter__ i __exit__ dla klas
- Upewnij się o proper cleanup w __exit__
- Obsługuj wyjątki w cleanup
- Użyj contextlib.ExitStack dla wielu zasobów
- Rozważ async context managers (async with)
Zapewnia automatyczne zarządzanie zasobami.
```

### 1.15 Generatory dla Dużych Sekwencji
```
Zamień listy na generatory dla efektywności pamięci:
- Zamień list return → yield
- Użyj wyrażeń generatorowych zamiast list comprehensions
- Implementuj __iter__ i __next__ dla niestandardowych iteratorów
- Użyj yield from dla delegacji
- Rozważ generatory dla pipeline'ów danych
- Mierz użycie pamięci (przed/po)
Szczególnie cenne dla przetwarzania dużych plików.
```

### 1.16 Enum dla Stałych
```
Zamień stałe stringowe/int na Enum:
- Definiuj klasy Enum dla powiązanych stałych
- Użyj auto() dla automatycznych wartości
- Dodaj metody dla logiki biznesowej
- Użyj @unique dla zapewnienia unikalności
- Implementuj __str__ dla prezentacji
- Użyj IntEnum/StrEnum dla kompatybilności
Type-safe i samodokumentujące.
```

### 1.17 Pathlib dla Operacji na Plikach
```
Zamień os.path na pathlib.Path:
- Zastąp os.path.join() → Path() / operator
- Zastąp os.path.exists() → .exists()
- Użyj .read_text()/.write_text() dla prostych IO
- Użyj .glob() dla dopasowywania wzorców
- Łańcuchuj operacje w sposób czytelny
- Obsługuj WindowsPath vs PosixPath automatycznie
Bardziej obiektowe i czytelne.
```

### 1.18 Match-Case (Python 3.10+)
```
Użyj structural pattern matching gdzie stosowne:
- Zamień długie if-elif łańcuchy → match-case
- Użyj dopasowania wzorców dla struktur danych
- Dodaj guards dla dodatkowych warunków
- Obsługuj wielokrotne wzorce z |
- Użyj _ dla przypadku domyślnego
- Rozważ dopasowanie klas z wzorcami
Bardziej wyraziste niż if-elif dla złożonych warunków.
```

### 1.19 Dekoratory dla Cross-Cutting Concerns
```
Implementuj dekoratory dla powtarzalnej logiki:
- Cache z @functools.lru_cache
- Timing z niestandardowym dekoratorem
- Logging z wrappingiem
- Retry logic z exponential backoff
- Sprawdzanie uprawnień/walidacja
- Użyj functools.wraps zachowania metadanych
- Składaj dekoratory dla wielu concerns
Redukuje powielanie kodu cross-cutting.
```

### 1.20 Walrus Operator dla Zwięzłości
```
Użyj := gdzie poprawia czytelność:
- W warunkach if (if (n := len(items)) > 10:)
- W pętlach while (while (line := f.readline()):)
- W list comprehensions dla wielokrotnego użycia
- Unikaj nadużywania (czytelność najpierw)
Redukuje duplikację wyrażeń.
```

### Kategoria: Wzorce Projektowe

### 1.21 Singleton Pattern
```
Implementuj Singleton pattern:
- Użyj metaclass dla wymuszenia pojedynczej instancji
- Alternatywnie: moduł-poziom singleton
- Thread-safe implementation
- Rozważ @dataclass z __new__
- Dodaj testy dla pojedynczej instancji
- Dokumentuj uzasadnienie (dlaczego singleton potrzebny)
Używaj oszczędnie - często można uniknąć przez DI.
```

### 1.22 Factory Pattern
```
Implementuj factory pattern dla tworzenia obiektów:
- Definiuj interfejs factory
- Różne implementacje dla różnych typów
- Użyj registracji dla dynamicznego odkrywania
- Dodaj walidację parametrów
- Rozważ Abstract Factory dla rodzin powiązanych obiektów
- Dokumentuj wsparcie typów
Oddziela tworzenie od użycia.
```

### 1.23 Builder Pattern
```
Implementuj builder pattern dla złożonych obiektów:
- Utwórz klasę builder z fluent interface
- Metody do ustawiania każdego atrybutu
- Metoda build() do konstruowania obiektu finalnego
- Waliduj w build() przed utworzeniem
- Rozważ director dla wspólnych konfiguracji
- Użyj adnotacji typów dla bezpieczeństwa
Doskonałe dla obiektów z wieloma opcjonalnymi parametrami.
```

### 1.24 Observer Pattern
```
Implementuj observer pattern dla event handling:
- Definiuj subject z attach/detach/notify
- Definiuj interfejs observer
- Implementuj konkretnych observers
- Obsługuj wielokrotnych observers
- Rozważ async observers dla IO
- Dodaj filtrowanie zdarzeń
- Testuj propagację zdarzeń
Używany dla reaktywnych systemów.
```

### 1.25 Strategy Pattern
```
Implementuj strategy pattern dla wymiennej logiki:
- Definiuj interfejs strategii (Protocol/ABC)
- Implementuj konkretne strategie
- Context klasa przechowuje i używa strategii
- Pozwól na zmianę strategii w runtime
- Użyj type hints dla jasności
- Dodaj domyślną strategię
Preferuj kompozycję nad dziedziczenie.
```

### 1.26 Decorator Pattern
```
Implementuj decorator pattern (nie Python dekorator):
- Definiuj interfejs komponentu
- Implementuj konkretny komponent
- Utwórz dekorator bazowy wrapujący komponent
- Implementuj konkretne dekoratory dodające zachowanie
- Składaj dekoratory dla wielokrotnych funkcji
- Zachowaj interfejs w całym łańcuchu
Dynamiczne dodawanie odpowiedzialności.
```

### 1.27 Command Pattern
```
Implementuj command pattern dla akcji:
- Definiuj interfejs command z execute()
- Implementuj konkretne commands
- Receiver wykonuje faktyczną pracę
- Invoker wyzwala commands
- Dodaj undo() dla odwracalnych commands
- Queue commands dla batch processing
Używany dla undo/redo, transakcji.
```

### 1.28 Repository Pattern
```
Implementuj repository pattern dla dostępu do danych:
- Abstrakcja operacji CRUD
- Ukryj szczegóły dostępu do danych
- Zwracaj encje domenowe, nie surowe dane
- Dodaj metody zapytań (find_by_*)
- Użyj Protocol dla interfejsu
- Implementuj in-memory repo dla testów
Oddziela logikę biznesową od dostępu do danych.
```

### 1.29 Unit of Work Pattern
```
Implementuj Unit of Work dla transakcji:
- Śledź zmiany obiektów
- Commit wszystkiego lub rollback wszystkiego
- Integruj z repozytoriami
- Obsługuj transakcje DB
- Implementuj jako context manager
- Dodaj testy dla atomowości
Zapewnia spójność transakcji.
```

### 1.30 Dependency Injection
```
Implementuj Dependency Injection:
- Przekazuj zależności jako parametry
- Użyj Protocol dla abstrakcji
- Rozważ DI container (dependency-injector)
- Konfiguruj przez constructor injection
- Użyj factory dla złożonego tworzenia
- Dodaj testy z mock dependencies
Poprawia testowalność i elastyczność.
```

### Kategoria: Obsługa Błędów i Logowanie

### 1.31 Niestandardowa Hierarchia Wyjątków
```
Definiuj niestandardowe wyjątki dla domeny:
- Utwórz bazowy wyjątek aplikacji
- Podklasy dla specyficznych błędów (NotFound, ValidationError, etc.)
- Dodaj atrybuty kontekstu (user_id, resource_id)
- Implementuj __str__ dla czytelnych wiadomości
- Dokumentuj kiedy każdy jest rzucany
- Organizuj w hierarchię dla catch patterns
Unikaj używania samego Exception.
```

### 1.32 Strukturalne Logowanie
```
Implementuj strukturalne logowanie:
- Użyj biblioteki jak structlog lub loguru
- Loguj JSON dla łatwego parsowania
- Uwzględnij kontekst (request_id, user_id)
- Używaj poziomów logowania poprawnie (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Dodaj rotację logów i retention
- Redaguj wrażliwe dane
- Konfiguruj różne poziomy dla środowisk
Ułatwia wyszukiwanie i monitoring.
```

### 1.33 Try-Except-Else-Finally
```
Użyj pełnych bloków try poprawnie:
- try: kod który może się nie udać
- except: obsługuj specyficzne wyjątki
- else: kod jeśli brak wyjątków
- finally: cleanup który zawsze działa
- Unikaj gołych except (catch konkretnych typów)
- Loguj wyjątki z kontekstem
- Re-raise jeśli nie możesz obsłużyć
Proper error handling flow.
```

### 1.34 Walidacja z Pydantic
```
Użyj Pydantic do walidacji danych:
- Definiuj modele jako podklasy BaseModel
- Adnotacje typów dla automatycznej walidacji
- Użyj Field() dla ograniczeń (gt, lt, regex)
- Implementuj custom validators z @validator
- Obsługuj zagnieżdżone modele
- Serializuj do/z JSON, dict
- Generuj JSON schema automatycznie
Production-ready walidacja danych.
```

### 1.35 Obsługa Retry z Backoff
```
Implementuj retry logic z exponential backoff:
- Użyj tenacity lub implementuj własną
- Konfiguruj max attempts i delay
- Exponential backoff z jitter
- Retry tylko na retryable errors
- Loguj retry attempts
- Dodaj timeout dla całkowitego czasu
- Testuj z mock failures
Essential dla zewnętrznych API i sieciowych operacji.
```

### 1.36 Circuit Breaker Pattern
```
Implementuj circuit breaker dla usług zewnętrznych:
- Stany: closed (normalny), open (failing), half-open (testowanie)
- Otwórz po N failures
- Timeout przed próbą again
- Monitoruj health metrics
- Fallback logic kiedy otwarty
- Loguj zmiany stanu
Prevents cascading failures.
```

### 1.37 Dekorator Logowania Wykonania
```
Utwórz dekorator dla logowania wykonania funkcji:
- Loguj wejście z argumentami
- Loguj wyjście z wartością zwracaną
- Loguj czas wykonania
- Loguj wyjątki z traceback
- Redaguj wrażliwe argumenty
- Konfigurowalny poziom logowania
- Użyj functools.wraps
Valuable dla debugging.
```

### 1.38 Dead Letter Queue
```
Implementuj dead letter queue dla nieudanych zadań:
- Kolejkuj nieudane zadania oddzielnie
- Przechowuj oryginalną wiadomość + error
- Dodaj metadata (timestamp, attempt count)
- Implementuj retry logic z DLQ
- Monitoruj rozmiar DLQ
- Alerty dla nadmiernych failures
Prevents data loss w systemach asynchronicznych.
```

### 1.39 Graceful Degradation
```
Implementuj graceful degradation:
- Wykrywaj failures w zależnościach
- Przełącz na fallback behavior
- Zwracaj częściowe wyniki jeśli możliwe
- Loguj degraded operations
- Monitoruj degradation events
- Testuj degraded mode
Lepsze częściowe działanie niż całkowity failure.
```

### 1.40 Health Checks
```
Implementuj comprehensive health checks:
- Endpoint /health dla liveness
- Sprawdź połączenia DB
- Sprawdź dostępność zewnętrznych usług
- Sprawdź disk space i memory
- Zwracaj status code (200, 503)
- Uwzględnij szczegóły dla każdego komponentu
- Używane przez load balancers i orchestrators
Essential dla production deployments.
```

### Kategoria: Testowanie

### 1.41 Testy Jednostkowe z pytest
```
Napisz wyczerpujące testy jednostkowe z pytest:
- Jeden test na zachowanie/przypadek brzegowy
- Używaj fixtures dla setup/teardown
- Parametryzuj testy dla wielu wejść
- Mock zewnętrzne zależności
- Sprawdź wartości zwracane i side effects
- Testuj wyjątki z pytest.raises
- Cel: >80% pokrycie kodu (pytest-cov)
- Organizuj w test_*.py pliki
Szybkie, izolowane, powtarzalne.
```

### 1.42 Mock Dependencies
```
Mock zewnętrzne zależności w testach:
- Użyj unittest.mock lub pytest-mock
- Mock API calls, DB queries, file IO
- Użyj patch() jako dekoratora lub context manager
- Skonfiguruj return values z return_value
- Sprawdź wywołania z assert_called_with()
- Użyj MagicMock dla złożonych obiektów
- Preferuj dependency injection dla łatwego mockowania
Izoluj testy od zewnętrznych systemów.
```

### 1.43 Testy Integracyjne
```
Napisz testy integracyjne dla komponentów:
- Testuj interakcje między modułami
- Użyj prawdziwych DB (testcontainers)
- Testuj pełne API request/response flows
- Sprawdź side effects (DB inserts, file creation)
- Użyj fixtures do setup test data
- Cleanup po każdym teście
- Wolniejsze niż unit testy, ale wartościowe
Oddzielnie od unit tests (np. tests/integration/).
```

### 1.44 Property-Based Testing
```
Implementuj property-based testing z hypothesis:
- Definiuj properties które zawsze powinny być prawdziwe
- Użyj @given do generowania test inputs
- Hypothesis znajduje edge cases automatycznie
- Testuj invariants (reverse(reverse(x)) == x)
- Użyj strategies dla złożonych typów danych
- Łącz z pytest
Znajduje bugs których by się nie spodziewał.
```

### 1.45 Testy Coverage
```
Osiągnij wysokie pokrycie testami:
- Użyj pytest-cov do pomiaru
- Cel: >80% line coverage, >90% dla critical paths
- Generuj raporty HTML dla przeglądu
- Identyfikuj nieprzetestowane gałęzie
- Dodaj testy dla brakującego pokrycia
- Skonfiguruj CI do wymuszania minimum coverage
- Użyj coverage w review procesu
Nie cel sam w sobie, ale wskaźnik thoroughness.
```

### 1.46 Test Fixtures i Setup
```
Organizuj test fixtures efektywnie:
- Użyj @pytest.fixture do reużywalnego setup
- Scopes: function, class, module, session
- Użyj autouse dla automatycznego setup
- Yield fixtures dla teardown
- Parametryzowane fixtures dla wariantów
- Fixtures z fixtures (composition)
- Organizuj w conftest.py dla sharowania
Redukuje duplikację w test setup.
```

### 1.47 Testowanie Asynchronicznego Kodu
```
Testuj async funkcje poprawnie:
- Użyj pytest-asyncio
- Oznacz testy z @pytest.mark.asyncio
- Await w async test functions
- Mock async functions z AsyncMock
- Testuj concurrent behavior
- Sprawdź proper cleanup
Wymaga specjalnego handling.
```

### 1.48 Mutation Testing
```
Użyj mutation testing dla jakości testów:
- Użyj mutpy lub cosmic-ray
- Automatycznie mutuje kod (zmienia operatory, etc.)
- Sprawdza czy testy wykrywają mutacje
- Wysoki mutation score = silne testy
- Identyfikuj słabe testy
- Dodaj testy dla przeżywających mutantów
Testuje testy.
```

### 1.49 Testy Snapshot
```
Użyj snapshot testing dla złożonych output:
- Przechowuj oczekiwany output jako snapshot
- Porównaj aktualny output ze snapshot
- Użyj pytest-snapshot lub podobnej
- Przydatne dla API responses, UI renders
- Review zmian snapshotów w code review
- Update snapshots gdy zamierzone zmiany
Łapie niezamierzone zmiany w output.
```

### 1.50 Test-Driven Development
```
Praktykuj TDD workflow:
- Red: napisz nieprzechodzący test najpierw
- Green: napisz minimum kodu by test przeszedł
- Refactor: popraw kod zachowując przechodzenie testów
- Powtórz dla każdej funkcji
- Testy dokumentują expected behavior
- Wymusza testable design
- Wyższy confidence w refactoringu
Wymaga dyscypliny ale opłacalne.
```

### Kategoria: Wydajność i Optymalizacja

### 1.51 Profilowanie z cProfile
```
Profiluj kod do identyfikacji bottlenecków:
- Użyj cProfile.run() lub @profile decorator
- Analizuj output dla funkcji wykorzystujących najwięcej czasu
- Sortuj według cumtime lub tottime
- Wizualizuj z snakeviz
- Profiluj representative workloads
- Porównaj przed/po optymalizacji
- Dokument findings i improvements
Mierz najpierw, optymalizuj potem.
```

### 1.52 Optymalizacja Użycia Pamięci
```
Redukuj zużycie pamięci:
- Użyj memory_profiler dla analiz line-by-line
- Zamień listy na generatory gdzie możliwe
- Użyj __slots__ w klasach
- Uwolnij duże obiekty po użyciu (del)
- Użyj itertools dla lazy evaluation
- Rozważ memory-mapped files dla dużych danych
- Monitoruj z tracemalloc
Krytyczne dla dużych dataset'ów.
```

### 1.53 Współbieżność z multiprocessing
```
Użyj multiprocessing dla CPU-bound tasks:
- Pool dla równoległych zadań
- Process dla długo działających workers
- Użyj Queue dla komunikacji inter-process
- Implementuj proper shutdown
- Obsługuj wyjątki w procesach
- Testuj na wielordzeniowej maszynie
- Mierz speedup (ideally linear z cores)
Obchodzi Python GIL dla CPU work.
```

### 1.54 Threading dla IO-Bound
```
Użyj threading dla IO-bound operations:
- ThreadPoolExecutor dla pool workers
- Thread dla długo działających background tasks
- Użyj Queue dla thread-safe komunikacji
- Użyj Lock/RLock dla shared state
- Unikaj deadlocks z proper ordering
- Implementuj graceful shutdown
- Testuj thread safety
Alternatywnie rozważ asyncio.
```

### 1.55 Caching Strategii
```
Implementuj caching dla drogich operacji:
- @lru_cache dla czystych funkcji
- Redis/Memcached dla distributed caching
- Cache aside pattern (check cache, load, update cache)
- Konfiguruj TTL i eviction policies
- Cache invalidation strategia
- Monitoruj hit/miss ratios
- Testuj z cache misses
Dramatycznie poprawia latency.
```

### 1.56 Database Query Optimization
```
Optymalizuj database queries:
- Używaj indexów na często zapytywanych kolumnach
- Unikaj N+1 queries (use joins lub prefetch)
- Ogranicz zwracane kolumny (SELECT specific)
- Użyj paginacji dla dużych resultsets
- Batch inserts/updates
- Użyj connection pooling
- Profiluj queries z EXPLAIN
- Monitoruj slow query log
Często biggest bottleneck w web apps.
```

### 1.57 Lazy Loading
```
Implementuj lazy loading dla drogich zasobów:
- Ładuj dane tylko kiedy potrzebne
- Użyj @property z cachingiem
- Proxy pattern dla on-demand loading
- Lazy import dla rzadko używanych modules
- Database lazy relationships
- Balansuj z eager loading dla N+1
Redukuje startup time i pamięć.
```

### 1.58 Vectorization z NumPy
```
Vectorizuj obliczenia numeryczne:
- Zamień Python loops → NumPy array operations
- Użyj broadcasting dla element-wise operations
- Utilize NumPy built-ins (sum, mean, std)
- Unikaj Python loops nad arrays
- Użyj einsum dla złożonych operacji
- Mierz speedup (często 10-100x)
Essential dla numeric/scientific computing.
```

### 1.59 Kompilacja z Numba
```
Kompiluj performance-critical funkcje z Numba:
- Dodaj @jit decorator do numeric functions
- Używaj nopython=True dla full speedup
- Wsparcie dla NumPy operations
- Parallel=True dla automatic parallelization
- Profiluj compiled vs interpreted
- Warm-up JIT przed benchmarking
Może osiągnąć C-level performance.
```

### 1.60 Batch Processing
```
Implementuj batch processing dla throughput:
- Przetwarzaj records w batchach nie pojedynczo
- Amortize overhead (DB connections, API calls)
- Balansuj batch size (memory vs throughput)
- Użyj itertools.islice dla chunking
- Implementuj error handling per-batch
- Monitoruj batch processing metrics
- Rozważ parallel batch processing
Zwiększa throughput dla dużego volume.
```

---

## 2. Inżynieria Oprogramowania Prompty (50)

### Kategoria: Zasady SOLID

### 2.1 Single Responsibility Principle
```
Refaktoryzuj klasę naruszającą SRP:
- Identyfikuj wiele odpowiedzialności w jednej klasie
- Wyodrębnij każdą odpowiedzialność do oddzielnej klasy
- Każda klasa powinna mieć tylko jeden powód do zmiany
- Użyj composition do łączenia odpowiedzialności
- Nazwy klas powinny jasno wskazywać pojedynczą odpowiedzialność
- Testuj każdą klasę niezależnie
Przykład: UserManager → oddziel na UserAuthenticator, UserRepository, UserValidator.
```

### 2.2 Open/Closed Principle
```
Zrefaktoryzuj kod by był otwarty na rozszerzenie, zamknięty na modyfikację:
- Identyfikuj kod wymagający modyfikacji dla nowych funkcji
- Wprowadź abstrakcje (interfejsy/klasy abstrakcyjne)
- Użyj polimorfizmu zamiast if/switch statements
- Implementuj Strategy pattern dla wymiennej logiki
- Nowe funkcje = nowe klasy, nie zmiany w istniejących
- Testuj że istniejący kod nie zmienia się
Przykład: PaymentProcessor wsparcie dla nowych typów płatności przez nowe klasy, nie if statements.
```

### 2.3 Liskov Substitution Principle
```
Zapewnij zgodność z LSP:
- Podklasy muszą być podstawialne za klasy bazowe
- Nie wzmacniaj preconditions w podklasach
- Nie osłabiaj postconditions w podklasach
- Zachowaj invariants klasy bazowej
- Podklasy nie powinny rzucać nowych wyjątków
- Testuj z polymorphic wywołaniami
Przykład: Bird klasa bazowa, Penguin nie powinna dziedziczyć fly() jeśli nie może latać.
```

### 2.4 Interface Segregation Principle
```
Podziel fat interfaces na mniejsze, specyficzne:
- Identyfikuj interfejsy z wieloma niezwiązanymi metodami
- Klienty nie powinny zależeć od metod których nie używają
- Podziel na role-specific interfaces
- Klasy mogą implementować wiele małych interfaces
- Użyj Protocol w Pythonie dla duck typing
Przykład: Worker interface → podziel na Workable, Eatable, Sleepable.
```

### 2.5 Dependency Inversion Principle
```
Inwertuj zależności do abstrakcji:
- High-level modules nie powinny zależeć od low-level modules
- Oba powinny zależeć od abstrakcji
- Definiuj interfejsy/protocols dla zależności
- Wstrzykuj zależności przez konstruktor
- Używaj DI container dla złożonych grafów
- Mock abstrakcje w testach
Przykład: OrderService zależy od IPaymentProcessor interface, nie ConcretePaymentProcessor.
```

### 2.6 Kompozycja zamiast Dziedziczenia
```
Preferuj kompozycję nad dziedziczeniem:
- Identyfikuj głębokie hierarchie dziedziczenia
- Wyodrębnij zachowania do oddzielnych komponentów
- Klasa komponuje zachowania przez dependencies
- Bardziej elastyczne i testowalne
- Unikaj fragile base class problem
- Użyj interfejsów dla zachowań
Przykład: Car z Engine component, nie Car extends Engine.
```

### 2.7 Law of Demeter
```
Przestrzegaj Law of Demeter (principle of least knowledge):
- Obiekt powinien tylko rozmawiać z bezpośrednimi znajomymi
- Nie chain method calls (a.getB().getC().doSomething())
- Dodaj metody delegujące w swoim obiekcie
- Redukuj coupling między klasami
- "Nie rozmawiaj z nieznajomymi"
Poprawia maintainability poprzez luźne coupling.
```

### 2.8 Tell, Don't Ask
```
Praktykuj "Tell, Don't Ask":
- Nie pytaj obiektu o stan i decyduj
- Powiedz obiektowi co zrobić, on decyduje jak
- Enkapsuluje logikę biznesową w odpowiednich obiektach
- Redukuje proceduralne zapytania danych
- Obiekty są odpowiedzialne za swoje zachowanie
Przykład: zamiast if user.isActive(): user.lastLogin = now, użyj user.recordLogin().
```

### 2.9 YAGNI (You Aren't Gonna Need It)
```
Zastosuj YAGNI principle:
- Nie dodawaj funkcjonalności dopóki faktycznie nie potrzebna
- Usuń nieużywany kod i przedwczesne abstrakcje
- Upraszczaj over-engineered solutions
- Iteruj bazując na rzeczywistych wymaganiach
- Refaktoryzuj gdy potrzeby się wyłaniają
Zapobiega complexity i over-engineering.
```

### 2.10 DRY (Don't Repeat Yourself)
```
Eliminuj duplikację (DRY):
- Identyfikuj zduplikowany kod w wielu lokalizacjach
- Wyodrębnij wspólną logikę do reużywalnych funkcji/klas
- Użyj inheritance lub composition dla wspólnych zachowań
- Parametryzuj warianty zamiast duplikować
- Unikaj copy-paste programming
- Balansuj DRY z premature abstraction
Duplicacja to wróg maintainability.
```

### Kategoria: Wzorce Projektowe

### 2.11 MVC Architecture
```
Implementuj Model-View-Controller pattern:
- Model: logika biznesowa i stan danych
- View: prezentacja i UI rendering
- Controller: obsługuje user input, koordynuje Model i View
- Wyraźne separacja concerns
- Model niezależny od View
- Controller mediates między nimi
Klasyczny pattern dla web applications.
```

### 2.12 Repository Pattern
```
Implementuj Repository dla dostępu do danych:
- Abstrakcja warstwy dostępu do danych
- Interfejs przypominający kolekcję (add, remove, find)
- Enkapsuluje zapytania i persistence logic
- Różne implementacje (SQL, NoSQL, in-memory)
- Testuj z in-memory implementacją
- Zwracaj domain entities, nie DTOs
Oddziela biznesową logikę od warstwy danych.
```

### 2.13 Service Layer Pattern
```
Wprowadź Service Layer:
- Definiuj granice aplikacji i API
- Enkapsuluje logikę biznesową
- Koordynuje operacje na wielu obiektach
- Transactional boundaries
- Używany przez Controllers/handlers
- Testuj service logic w izolacji
Czyste API dla use cases aplikacji.
```

### 2.14 Adapter Pattern
```
Implementuj Adapter pattern:
- Konwertuj interfejs klasy na inny oczekiwany przez klientów
- Wrapper zewnętrznych bibliotek/APIs
- Pozwala niekompatybilnym interfejsom pracować razem
- Własne API zamiast vendor lock-in
- Łatwe do zmockowaia w testach
Szczególnie cenne dla third-party integrations.
```

### 2.15 Facade Pattern
```
Utwórz Facade dla złożonych subsystemów:
- Uproszczony interfejs do complex subsystem
- Ukryj złożoność implementacji
- Pojedynczy entry point dla related operations
- Nie usuwa dostępu do underlying subsystem
- Redukuje learning curve
Ułatwia używanie skomplikowanych systemów.
```

### 2.16 Template Method Pattern
```
Implementuj Template Method:
- Definiuj skeleton algorytmu w bazowej metodzie
- Podklasy override specific steps
- Bazowa klasa kontroluje overall flow
- Hook methods dla optional customization
- Inversion of control
Reużywa overall structure, customize details.
```

### 2.17 Chain of Responsibility
```
Implementuj Chain of Responsibility:
- Łańcuch handlerów dla procesowania żądań
- Każdy handler decyduje czy procesować lub przekazać dalej
- Decouple sender od receivers
- Dynamiczne dodawanie/usuwanie handlerów
- Użyj dla middleware, validation pipelines
Przykład: logging → authentication → authorization → business logic.
```

### 2.18 State Pattern
```
Implementuj State pattern dla state-dependent behavior:
- Enkapsuluje states jako obiekty
- Behavior zmienia się z state transitions
- Context deleguje do current state object
- States implementują wspólny interfejs
- Eliminuje złożone if/switch na state
Czyste state machines.
```

### 2.19 Mediator Pattern
```
Implementuj Mediator dla komunikacji między obiektami:
- Centralizuje złożoną komunikację
- Obiekty komunikują się przez mediatora, nie bezpośrednio
- Redukuje coupling między komponentami
- Mediator zawiera coordination logic
Używany w event buses, message brokers.
```

### 2.20 Null Object Pattern
```
Użyj Null Object zamiast null checks:
- Dostarcz object implementujący interfejs z no-op behavior
- Eliminuje potrzebę null checking
- Polimorficzne traktowanie z real objects
- Definiuj reasonable default behavior
Redukuje defensive null checking.
```

### Kategoria: Wzorce Architektoniczne

### 2.21 Layered Architecture
```
Organizuj w warstwy:
- Presentation Layer (UI, API endpoints)
- Application Layer (use cases, orchestration)
- Domain Layer (biznesowa logika, entities)
- Infrastructure Layer (DB, external services)
- Zależności płyną w dół
- Każda warstwa ma jasną odpowiedzialność
Klasyczna n-tier architecture.
```

### 2.22 Hexagonal Architecture (Ports & Adapters)
```
Implementuj Hexagonal Architecture:
- Core domain w centrum
- Ports definiują interfejsy (inputs/outputs)
- Adapters implementują ports (konkretne tech)
- Business logic niezależna od infrastructure
- Łatwe swapowanie adapters
- Testuj core bez external dependencies
Also known as Ports & Adapters.
```

### 2.23 Clean Architecture
```
Zastosuj Clean Architecture principles:
- Entities (business objects)
- Use Cases (application business rules)
- Interface Adapters (controllers, presenters)
- Frameworks & Drivers (external)
- Zależności wskazują do środka
- Inner layers nie znają outer layers
Framework-agnostic design.
```

### 2.24 CQRS (Command Query Responsibility Segregation)
```
Podziel na Commands i Queries:
- Commands: zmieniają stan (void return)
- Queries: zwracają dane (no side effects)
- Osobne modele dla read/write
- Optymalizuj każdy oddzielnie
- Może używać różnych datastores
Skalowalność dla read-heavy workloads.
```

### 2.25 Event-Driven Architecture
```
Projektuj event-driven system:
- Komponenty komunikują się przez eventy
- Publishers emitują eventy
- Subscribers konsumują eventy
- Loose coupling między componentami
- Asynchronous processing
- Event sourcing dla auditability
Scalable i resilient.
```

### 2.26 Microservices Architecture
```
Podziel na microservices:
- Małe, niezależnie deployowalne serwisy
- Każdy serwis odpowiada za specyficzną domenę
- Komunikacja przez APIs (REST, gRPC, messaging)
- Niezależne bazy danych per-service
- Decentralized governance
- Fault isolation
Trade-offs: complexity vs scalability.
```

### 2.27 API Gateway Pattern
```
Implementuj API Gateway:
- Pojedynczy entry point dla klientów
- Routing żądań do odpowiednich microservices
- Cross-cutting concerns (auth, rate limiting, logging)
- Response aggregation
- Protocol translation
Simplifies client code.
```

### 2.28 Backend for Frontend (BFF)
```
Utwórz BFFs dla różnych klientów:
- Osobny backend per client type (web, mobile, etc.)
- Tailored API dla specyficznych client needs
- Każdy BFF aggreegates data z microservices
- Optimized dla client use cases
Lepsze niż one-size-fits-all API.
```

### 2.29 Circuit Breaker Pattern
```
Implementuj Circuit Breaker:
- Monitoruj failures w wywołaniach zewnętrznych
- Otwarty: fast fail bez wywołania
- Half-open: testuj czy service recovered
- Closed: normalne wywołania
- Prevents cascading failures
- Timeouts i retry logic
Essential dla distributed systems.
```

### 2.30 Saga Pattern
```
Implementuj Saga dla distributed transactions:
- Long-running transaction split na local transactions
- Choreography: events trigger steps
- Orchestration: coordinator diriguje
- Compensating transactions dla rollback
- Eventual consistency
Distributed transactions bez 2PC.
```

### Kategoria: Design API

### 2.31 RESTful API Design
```
Projektuj RESTful APIs:
- Używaj HTTP methods poprawnie (GET, POST, PUT, DELETE, PATCH)
- Resource-based URLs (/users/123)
- Plural nouns dla kolekcji
- Stateless requests
- Proper status codes (200, 201, 400, 404, 500)
- HATEOAS links dla navigation
- Versioning (/v1/users)
Przestrzegaj REST constraints.
```

### 2.32 GraphQL API Design
```
Projektuj GraphQL API:
- Zdefiniuj schema z types, queries, mutations
- Clients query dla exactly needed data
- Eliminuje over/under-fetching
- Single endpoint
- Introspection dla documentation
- Implementuj DataLoader dla batching
- Handle N+1 query problem
Elastyczne dla klientów.
```

### 2.33 API Versioning
```
Implementuj API versioning strategy:
- URL versioning (/v1/, /v2/)
- Header versioning (Accept: application/vnd.api.v2+json)
- Query parameter (?version=2)
- Maintain backward compatibility w minor versions
- Deprecation policy i migration guide
- Version specific documentation
Plan dla breaking changes.
```

### 2.34 API Documentation
```
Dokładna dokumentacja API:
- OpenAPI/Swagger specification
- Każdy endpoint: description, parameters, responses
- Przykładowe requests/responses
- Authentication requirements
- Rate limiting policies
- Error codes i znaczenia
- Interactive API explorer (Swagger UI)
Good docs = łatwiejsza adopcja.
```

### 2.35 API Rate Limiting
```
Implementuj rate limiting:
- Limituj requesty per user/IP
- Multiple tiers (free, paid)
- Return 429 Too Many Requests
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining
- Sliding window lub token bucket
- Graceful degradation dla limitów
Chroni przed abuse i overload.
```

### 2.36 API Authentication & Authorization
```
Zapewnij proper auth:
- Authentication: JWT, OAuth 2.0, API keys
- Authorization: role-based lub permission-based
- Secure token storage i transmission (HTTPS)
- Token expiration i refresh
- Proper error messages (bez information leakage)
- Audit logging access
Security first.
```

### 2.37 Error Handling w APIs
```
Spójne error handling:
- Standard error response format
- Error code, message, details
- HTTP status codes matching error type
- User-friendly messages
- Developer-friendly debug info (w dev mode)
- Nie expose stack traces w produkcji
- Documentation wszystkich error codes
Predictable error handling.
```

### 2.38 API Pagination
```
Implementuj pagination dla dużych collections:
- Offset/limit: ?offset=20&limit=10
- Cursor-based: ?cursor=xyz&limit=10
- Response metadata: total count, next/prev links
- Consistent ordering
- Reasonable default page size
- Maximum page size limit
Essential dla performance.
```

### 2.39 API Filtering & Sorting
```
Wspieraj filtering i sorting:
- Query parameters: ?status=active&role=admin
- Multiple filters z AND logic
- Sorting: ?sort=created_at&order=desc
- Multiple sort fields
- Document supported filters
- Waliduj filter parameters
Flexible data retrieval.
```

### 2.40 Idempotency w APIs
```
Zapewnij idempotency:
- GET, PUT, DELETE naturally idempotent
- POST użyj idempotency keys
- Client sends unique key w header
- Server checks dla duplicate requests
- Returns same result dla duplicates
- TTL dla stored keys
Bezpieczniejsze retries.
```

### Kategoria: Database Design

### 2.41 Normalizacja Bazy Danych
```
Normalizuj do 3NF (minimum):
- 1NF: atomic values, no repeating groups
- 2NF: no partial dependencies
- 3NF: no transitive dependencies
- Eliminuj redundancję danych
- Redukuje anomalie update
- Denormalizuj wybiórczo dla performance
Trade-offs między normalizacją a performance.
```

### 2.42 Index Optimization
```
Optymalizuj indeksy:
- Index kolumny w WHERE clauses
- Index foreign keys
- Composite indexes dla multi-column queries
- Consider index order dla composite
- Unikaj over-indexing (koszt przy writes)
- Monitoruj query plans
- Drop unused indexes
Biggest performance win dla reads.
```

### 2.43 Query Optimization
```
Optymalizuj SQL queries:
- Użyj EXPLAIN ANALYZE
- Unikaj SELECT * (specify columns)
- Użyj JOINs zamiast subqueries gdzie możliwe
- Index covering queries
- Limit result sets
- Batch operations
- Unikaj N+1 queries
- Użyj prepared statements
Measure before/after performance.
```

### 2.44 Database Transactions
```
Używaj transakcji poprawnie:
- ACID properties (Atomicity, Consistency, Isolation, Durability)
- Rozpocznij transaction, commit lub rollback
- Najmniejszy możliwy scope
- Proper isolation levels
- Handle deadlocks z retry
- Unikaj długo działających transactions
Data consistency.
```

### 2.45 Database Migrations
```
Zarządzaj schema migrations:
- Version-controlled migration scripts
- Użyj narzędzi (Alembic, Flyway)
- Up i down migrations
- Testuj migrations w staging
- Rollback plan
- Data migrations oddzielnie od schema
- Zero-downtime migrations dla produkcji
Safe schema evolution.
```

### 2.46 Connection Pooling
```
Implementuj connection pooling:
- Reuse database connections
- Konfiguruj min/max pool size
- Connection timeout settings
- Test connections przed użyciem
- Proper connection cleanup
- Monitoruj pool utilization
Znacząco poprawia performance.
```

### 2.47 Read Replicas
```
Użyj read replicas dla scalability:
- Master dla writes
- Replicas dla reads
- Route read queries do replicas
- Handle replication lag
- Failover do master jeśli needed
- Monitoruj replication lag
Scales read-heavy workloads.
```

### 2.48 Database Sharding
```
Implementuj sharding dla masywnej skali:
- Partycjonuj dane across multiple databases
- Sharding key selection (np. user_id)
- Hash-based lub range-based sharding
- Routing layer dla żądań
- Handle cross-shard queries
- Rebalancing strategy
Horizontal scalability.
```

### 2.49 Soft Deletes
```
Implementuj soft deletes:
- Dodaj deleted_at kolumnę
- UPDATE zamiast DELETE
- Filter deleted records w queries
- Archival strategy dla starych deleted records
- Compliance z data retention policies
- Provide undelete functionality
Auditability i recoverability.
```

### 2.50 Database Auditing
```
Implementuj audit trail:
- Track all changes (create, update, delete)
- Audit table: who, what, when
- Trigger-based lub application-level
- Przechowuj old i new values
- Immutable audit log
- Retention policy
- Query capabilities dla audit data
Compliance i debugging.
```
