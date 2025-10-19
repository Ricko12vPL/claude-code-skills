# 🎯 Claude Code Skills - Complete Prompt Library

**400 Professional Prompts for 8 Skills**

This comprehensive guide contains 50 carefully crafted prompts for each Skill, following prompt engineering best practices:
- ✅ Clear, specific, and actionable
- ✅ Context-aware
- ✅ Expected outcomes defined
- ✅ Measurable results
- ✅ Production-ready scenarios

---

## 📚 Table of Contents

1. [Python Programming](#1-python-programming-50-prompts) (50 prompts)
2. [Software Engineering](#2-software-engineering-50-prompts) (50 prompts)
3. [Machine Learning](#3-machine-learning-50-prompts) (50 prompts)
4. [Quantitative Finance](#4-quantitative-finance-50-prompts) (50 prompts)
5. [Senior Quantitative Developer](#5-senior-quantitative-developer-50-prompts) (50 prompts)
6. [Senior Quantitative Researcher](#6-senior-quantitative-researcher-50-prompts) (50 prompts)
7. [Senior Systematic Trader](#7-senior-systematic-trader-50-prompts) (50 prompts)
8. [Senior Quantitative Trader](#8-senior-quantitative-trader-50-prompts) (50 prompts)

---

# 1. Python Programming (50 Prompts)

## Category: Code Quality & Style (10 prompts)

### 1.1 Basic Style Compliance
```
Refactor this Python function to be fully PEP 8 compliant. Add type hints for all
parameters and return values. Include a docstring following PEP 257 conventions
with Args, Returns, and Raises sections.

[paste your function here]
```

### 1.2 Type Hints Enhancement
```
Add comprehensive type hints to this Python module using typing.List, Dict, Optional,
Union, and Callable where appropriate. Ensure compatibility with Python 3.9+.
Include type hints for class attributes and method parameters.

[paste your code here]
```

### 1.3 Docstring Documentation
```
Write complete Google-style docstrings for this Python class. Include class-level
docstring, docstrings for __init__ and all public methods. Document parameters,
return values, raises, examples, and notes sections where relevant.

[paste your class here]
```

### 1.4 Code Review for Standards
```
Review this Python code for PEP 8 violations, missing type hints, and documentation
issues. Provide a detailed report with:
1. List of specific violations
2. Line numbers
3. Suggested fixes
4. Priority (critical/important/minor)

[paste your code here]
```

### 1.5 Import Organization
```
Reorganize the imports in this Python file following PEP 8 guidelines:
1. Standard library imports
2. Related third-party imports
3. Local application imports
Remove unused imports and use absolute imports where possible.

[paste your imports here]
```

### 1.6 Naming Conventions
```
Refactor this code to follow Python naming conventions:
- snake_case for functions and variables
- PascalCase for classes
- UPPER_CASE for constants
- _private for internal methods
Provide before/after comparison.

[paste your code here]
```

### 1.7 Line Length Optimization
```
Refactor this Python function to comply with the 88-character line length limit
(Black formatter standard) while maintaining readability. Use appropriate line
breaks, parentheses, and backslashes.

[paste your long lines here]
```

### 1.8 Code Formatting Automation
```
Configure a Python project with Black, isort, and flake8. Create pyproject.toml
with settings for:
- Black: line-length=88, target-version py39
- isort: profile="black"
- flake8: max-line-length=88, ignore=E203,W503
Include pre-commit hooks configuration.
```

### 1.9 Type Checking Setup
```
Set up mypy for strict type checking in a Python project. Create mypy.ini with:
- strict mode enabled
- disallow_untyped_defs
- warn_return_any
- no_implicit_optional
Fix all type errors in the provided module.

[paste your code here]
```

### 1.10 Documentation Generation
```
Generate comprehensive API documentation for this Python package using Sphinx.
Create:
1. RST files for each module
2. API reference with autodoc
3. Usage examples
4. Module hierarchy diagram

[provide package structure]
```

## Category: Modern Python Features (10 prompts)

### 1.11 Dataclasses Implementation
```
Convert this traditional Python class to use dataclasses. Add:
- Type hints for all fields
- Default values where appropriate
- field() for mutable defaults
- __post_init__ for validation
- frozen=True if immutable

[paste your class here]
```

### 1.12 Context Managers
```
Create a context manager for managing database connections with:
- Automatic connection opening
- Transaction management (commit/rollback)
- Connection closing in __exit__
- Error handling and logging
- Type hints and proper resource cleanup
Use both class-based and @contextmanager approaches.
```

### 1.13 Async/Await Implementation
```
Convert this synchronous Python function to async/await pattern. Include:
- async def declaration
- await for I/O operations
- asyncio.gather for concurrent operations
- Error handling with try/except
- Type hints for coroutines
- Example usage with asyncio.run()

[paste your sync function here]
```

### 1.14 Generators and Iterators
```
Implement a generator function for processing large files (>1GB) line by line
without loading entire file into memory. Include:
- yield for memory efficiency
- Context manager for file handling
- Error handling
- Type hints with Generator[str, None, None]
- Example usage
```

### 1.15 Decorators
```
Create a decorator that:
1. Measures function execution time
2. Logs function name, args, kwargs, and result
3. Handles exceptions gracefully
4. Works with both sync and async functions
5. Preserves function signature with functools.wraps
6. Includes type hints
Provide usage examples.
```

### 1.16 Property and Descriptors
```
Refactor this class to use @property decorators for getters/setters with:
- Validation in setter
- Computed properties
- Private attributes (_name)
- Read-only properties
- Type hints
Also demonstrate custom descriptor implementation.

[paste your class here]
```

### 1.17 Enum Classes
```
Convert these string constants to a proper Enum class with:
- Auto-generated values
- Custom string representation
- Iteration support
- Value validation
- Type hints
Include examples of usage in function signatures.

[paste your constants here]
```

### 1.18 Pathlib Usage
```
Refactor this code using os.path to use pathlib.Path with:
- Path concatenation with /
- exists(), is_file(), is_dir() checks
- glob patterns for file matching
- read_text() and write_text()
- Platform-independent paths

[paste your os.path code here]
```

### 1.19 F-strings and String Formatting
```
Convert all string formatting in this code to f-strings. Include:
- Variable interpolation
- Expression evaluation
- Format specifications (:,.2f, :>10, etc.)
- Multi-line f-strings
- Debugging with f"{var=}"

[paste your code here]
```

### 1.20 Pattern Matching (Python 3.10+)
```
Refactor this if-elif chain to use structural pattern matching (match/case)
from Python 3.10+. Include:
- Simple value matching
- Guard clauses with 'if'
- Capturing with 'as'
- OR patterns with |
- Wildcard patterns with _
Type hints for matched objects.

[paste your if-elif code here]
```

## Category: Design Patterns (10 prompts)

### 1.21 Singleton Pattern
```
Implement a thread-safe Singleton pattern in Python using:
1. Metaclass approach
2. Decorator approach
3. __new__ method approach
Compare pros/cons of each. Include type hints, docstrings, and thread safety
using threading.Lock. Demonstrate usage.
```

### 1.22 Factory Pattern
```
Create a Factory pattern for database connections supporting:
- PostgreSQL
- MySQL
- SQLite
- MongoDB
Abstract base class for DB interface, concrete implementations, and factory
method to create instances based on config. Include type hints and error handling.
```

### 1.23 Builder Pattern
```
Implement a Builder pattern for constructing complex HTTP requests with:
- Method chaining
- Optional parameters (headers, params, body, timeout)
- Validation in build() method
- Type hints
- Fluent interface
Example: Request().method('POST').url('...').header('...').body({...}).build()
```

### 1.24 Observer Pattern
```
Create an Observer pattern for event-driven system with:
- Subject class with attach/detach/notify
- Observer abstract base class
- Concrete observers for logging, metrics, alerts
- Type hints with Protocol
- Thread-safe implementation
Demonstrate with stock price monitoring example.
```

### 1.25 Strategy Pattern
```
Implement Strategy pattern for sorting algorithms with:
- Strategy interface (ABC)
- Concrete strategies: BubbleSort, QuickSort, MergeSort
- Context class to use strategies
- Type hints with generics
- Performance comparison
Show how to switch strategies at runtime.
```

### 1.26 Decorator Pattern (not the @ decorator)
```
Implement structural Decorator pattern for text formatting with:
- Component interface
- Concrete component (PlainText)
- Decorators: BoldDecorator, ItalicDecorator, ColorDecorator
- Composable decorators
- Type hints
Example: ColorDecorator(BoldDecorator(PlainText("Hello")))
```

### 1.27 Command Pattern
```
Create Command pattern for implementing undo/redo functionality with:
- Command interface with execute() and undo()
- Concrete commands for text operations
- CommandInvoker with history stack
- Type hints and generics
- Example: TextEditor with undo/redo
```

### 1.28 Adapter Pattern
```
Implement Adapter pattern to make incompatible interfaces work together:
- Target interface (modern API)
- Adaptee (legacy class)
- Adapter class
- Class and object adapter approaches
- Type hints
Example: Adapt old logging system to new interface.
```

### 1.29 Dependency Injection
```
Implement Dependency Injection pattern with:
- Constructor injection
- Property injection
- Method injection
- Simple DI container
- Type hints with Protocol
Example: Service layer with injected repository, logger, and cache.
```

### 1.30 Repository Pattern
```
Create Repository pattern for data access with:
- Generic repository interface
- Concrete repositories for User, Product entities
- Query methods (get, find, add, update, delete)
- Unit of Work pattern integration
- Type hints with generics TypeVar
- Async/await support
```

## Category: Error Handling & Logging (10 prompts)

### 1.31 Custom Exceptions
```
Create a hierarchy of custom exceptions for a REST API client with:
- Base exception: APIException
- Subclasses: NetworkError, AuthenticationError, ValidationError, RateLimitError
- Error codes, messages, and context data
- __str__ and __repr__ methods
- Type hints
Include usage examples with try/except.
```

### 1.32 Exception Context Management
```
Refactor this code to use proper exception handling with:
- Specific exception types (not bare except)
- Exception chaining with 'from'
- finally for cleanup
- Context manager for resources
- Logging exceptions with traceback
- Re-raising with raise

[paste your code here]
```

### 1.33 Logging Setup
```
Configure comprehensive logging for a production application with:
- Multiple handlers: console (INFO), file (DEBUG), rotating file
- Formatters with timestamp, level, module, message
- Logger hierarchy
- Environment-based configuration
- Structured logging with JSON
Use Python's logging module. Include loguru alternative.
```

### 1.34 Retry Logic with Backoff
```
Implement retry decorator with exponential backoff for handling transient failures:
- Configurable max retries
- Exponential backoff (2^n seconds)
- Jitter for distributed systems
- Specific exception types to retry
- Logging retry attempts
- Type hints and async support
```

### 1.35 Circuit Breaker Pattern
```
Implement Circuit Breaker pattern for handling cascading failures:
- States: CLOSED, OPEN, HALF_OPEN
- Failure threshold
- Timeout period
- Success threshold for recovery
- Metrics tracking
- Thread-safe implementation
Example: API call wrapper.
```

### 1.36 Validation with Pydantic
```
Create Pydantic models for API request/response validation with:
- Type validation
- Custom validators with @validator
- Field constraints (min, max, regex)
- Nested models
- JSON serialization/deserialization
- Error handling with ValidationError
Example: User registration API.
```

### 1.37 Assertion-based Validation
```
Write assertion-based validation functions with descriptive error messages:
- Type assertions
- Value range assertions
- State assertions
- Custom assert messages
- AssertionError with context
Include pytest assertions for testing.

[provide data structure to validate]
```

### 1.38 Graceful Degradation
```
Implement graceful degradation pattern for external service failures:
- Try primary service
- Fallback to secondary service
- Cache fallback
- Default values for non-critical features
- Error logging and monitoring
- Circuit breaker integration
Example: Feature flags service with fallback.
```

### 1.39 Error Recovery Strategies
```
Design error recovery strategy for data processing pipeline:
- Dead letter queue for failed items
- Checkpointing for resumable processing
- Partial success handling
- Error categorization (retryable/non-retryable)
- Monitoring and alerting
- Type hints and async support
```

### 1.40 Debug Logging
```
Add comprehensive debug logging to this function:
- Input parameters logging
- Intermediate state logging
- Performance checkpoints
- Exception logging with traceback
- Contextual information (request_id, user_id)
Use Python's logging module with DEBUG level.

[paste your function here]
```

## Category: Testing (10 prompts)

### 1.41 Pytest Test Suite
```
Create comprehensive pytest test suite for this module with:
- Unit tests for all public functions
- Test fixtures
- Parametrized tests with @pytest.mark.parametrize
- Test organization in classes
- Setup/teardown with fixtures
- 90%+ code coverage
Include conftest.py with shared fixtures.

[paste your module here]
```

### 1.42 Mocking External Dependencies
```
Write pytest tests using unittest.mock for:
- API calls (requests library)
- Database queries
- File system operations
- Environment variables
- Time-dependent functions
Include @patch decorator, MagicMock, and side_effect. Verify mock calls.
```

### 1.43 Pytest Fixtures
```
Create reusable pytest fixtures for:
- Database connection (with rollback)
- Test data (users, products)
- Temporary directories
- HTTP client
- Authentication tokens
Use fixture scopes (function/class/module/session) appropriately. Include autouse.
```

### 1.44 Property-Based Testing
```
Write property-based tests using Hypothesis for:
- String manipulation functions
- Math operations
- List processing
- Data structure invariants
Include @given decorator, strategies (st.text(), st.integers()), and assume().
Example: test sort function maintains order and length.

[provide function to test]
```

### 1.45 Integration Tests
```
Create integration test suite for REST API with:
- Test fixtures for database setup/teardown
- HTTP client with TestClient (FastAPI) or TestCase (Flask)
- Test all endpoints (GET, POST, PUT, DELETE)
- Assert status codes, response bodies, headers
- Test error cases and validation
Include database state verification.
```

### 1.46 Async Testing
```
Write pytest tests for async functions using pytest-asyncio:
- @pytest.mark.asyncio
- Async fixtures
- Testing concurrent operations
- Mock async functions
- Timeout handling
Example: test async API client, database operations.

[paste your async code here]
```

### 1.47 Test Coverage Analysis
```
Set up pytest-cov for code coverage analysis:
- pytest.ini configuration
- Coverage thresholds (90% minimum)
- HTML coverage report
- Branch coverage
- Exclude test files and __init__.py
- CI/CD integration
Generate report and identify untested code paths.
```

### 1.48 Exception Testing
```
Write pytest tests for exception handling:
- pytest.raises() context manager
- Asserting exception type and message
- Testing exception chaining
- Testing custom exceptions
- Parametrized exception tests
Include both expected and unexpected exceptions.

[provide code with exceptions]
```

### 1.49 Test Data Factories
```
Create test data factories using factory_boy:
- Factory classes for models
- Faker for realistic data
- Related objects (ForeignKey)
- Sequences for unique values
- Traits for variations
- SubFactory for nested objects
Example: UserFactory, OrderFactory with items.
```

### 1.50 Snapshot Testing
```
Implement snapshot testing for:
- JSON API responses
- Rendered HTML templates
- Generated reports
- Data transformation outputs
Use pytest-snapshot or syrupy. Include:
- Baseline snapshot creation
- Diff on changes
- Snapshot update workflow
```

## Category: Performance & Optimization (10 prompts)

### 1.51 List Comprehensions Optimization
```
Refactor this code to use list/dict/set comprehensions for better performance:
- Replace for loops with comprehensions
- Use conditional comprehensions
- Nested comprehensions where appropriate
- Generator expressions for memory efficiency
Compare performance with timeit.

[paste your loop code here]
```

### 1.52 Profiling with cProfile
```
Profile this Python script to identify performance bottlenecks:
1. Run cProfile with stats
2. Analyze top 20 time-consuming functions
3. Identify optimization opportunities
4. Generate visual profile with snakeviz
5. Provide optimization recommendations with expected improvements

[paste your code here]
```

### 1.53 Memory Profiling
```
Analyze memory usage of this Python program using memory_profiler:
- Line-by-line memory consumption
- Identify memory leaks
- Large object allocations
- Optimization strategies (generators, __slots__, del)
- Before/after comparison
Provide specific memory reduction recommendations.

[paste your code here]
```

### 1.54 Caching with functools
```
Add caching to expensive functions using:
- @functools.lru_cache for functions with hashable args
- @functools.cache for Python 3.9+ (unlimited cache)
- Custom cache with TTL
- Cache statistics (hits, misses)
- Cache invalidation strategy
Measure performance improvement.

[paste your expensive function here]
```

### 1.55 Multiprocessing for CPU-Bound Tasks
```
Parallelize this CPU-bound operation using multiprocessing:
- multiprocessing.Pool
- Process count based on cpu_count()
- Chunk size optimization
- Error handling in workers
- Progress tracking
- Performance comparison (1 vs N processes)

[paste your CPU-bound code here]
```

### 1.56 Asyncio for I/O-Bound Tasks
```
Convert this I/O-bound code to async/await for better concurrency:
- aiohttp for HTTP requests
- asyncio.gather for parallel operations
- asyncio.Semaphore for rate limiting
- Error handling
- Performance metrics (requests/second)
Compare with synchronous version.

[paste your I/O code here]
```

### 1.57 NumPy Vectorization
```
Refactor this Python loop to use NumPy vectorized operations:
- Replace for loops with array operations
- Use broadcasting
- Universal functions (ufuncs)
- Memory-efficient operations
- Performance comparison (100x+ faster)

[paste your numerical loop here]
```

### 1.58 String Operations Optimization
```
Optimize string operations in this code:
- Use str.join() instead of += in loops
- f-strings instead of format() or %
- str.translate() for character replacement
- Regular expressions compilation with re.compile()
- Performance benchmarking
Provide before/after timing.

[paste your string code here]
```

### 1.59 Database Query Optimization
```
Optimize these database queries (SQLAlchemy):
- N+1 query problem fix with joinedload/subqueryload
- Select only needed columns
- Batch operations with bulk_insert_mappings
- Index recommendations
- Query execution plan analysis
Measure query time improvement.

[paste your queries here]
```

### 1.60 Algorithm Complexity Analysis
```
Analyze time and space complexity of this algorithm:
- Big O notation for time complexity
- Space complexity analysis
- Best/average/worst case scenarios
- Suggest more efficient algorithm if O(n²) or worse
- Implement optimized version
- Benchmark with different input sizes

[paste your algorithm here]
```

---

# 2. Software Engineering (50 Prompts)

## Category: SOLID Principles (10 prompts)

### 2.1 Single Responsibility Principle (SRP)
```
Refactor this class that violates SRP by having multiple responsibilities:
1. Identify all responsibilities
2. Extract each into separate classes
3. Maintain clear interfaces
4. Add dependency injection
5. Include type hints and docstrings
Explain how this improves maintainability and testability.

[paste your god class here]
```

### 2.2 Open/Closed Principle (OCP)
```
Refactor this code to follow OCP - open for extension, closed for modification:
- Use abstract base classes or protocols
- Strategy pattern for behaviors
- Plugin architecture where appropriate
- No if/elif chains for types
- New features via new classes, not existing code modification
Include example of adding new functionality.

[paste your code here]
```

### 2.3 Liskov Substitution Principle (LSP)
```
Fix LSP violations in this inheritance hierarchy:
- Ensure derived classes can substitute base class
- No strengthening preconditions
- No weakening postconditions
- Preserve invariants
- Fix improper inheritance relationships
Include examples showing correct substitution.

[paste your class hierarchy here]
```

### 2.4 Interface Segregation Principle (ISP)
```
Refactor this fat interface that violates ISP:
1. Identify distinct client needs
2. Split into smaller, focused interfaces
3. Clients depend only on methods they use
4. Use Protocol or ABC
5. Show multiple clients using different interfaces
Explain benefits for testing and maintenance.

[paste your interface here]
```

### 2.5 Dependency Inversion Principle (DIP)
```
Refactor to follow DIP - depend on abstractions, not concretions:
- Define abstract interfaces
- Inject dependencies via constructors
- High-level modules independent of low-level details
- Use Protocol or ABC
- Include DI container example
Show testability improvements with mocks.

[paste your tightly coupled code here]
```

### 2.6 SOLID Code Review
```
Review this codebase for SOLID violations:
1. List violations for each principle (S, O, L, I, D)
2. Rate severity (critical/major/minor)
3. Provide refactoring plan
4. Prioritize fixes
5. Estimate refactoring effort
Include specific line numbers and code examples.

[paste your code here]
```

### 2.7 SOLID Design from Scratch
```
Design a payment processing system following all SOLID principles:
- Multiple payment methods (credit card, PayPal, crypto)
- Validation, fraud detection, notifications
- Abstract interfaces for each responsibility
- Clear separation of concerns
- Extensible architecture
Include class diagram and code skeleton.
```

### 2.8 Dependency Injection Container
```
Implement a simple DI container in Python:
- Register services (singleton, transient, scoped)
- Resolve dependencies automatically
- Constructor injection
- Circular dependency detection
- Type hints and generic support
Example: resolve UserService with injected repository and logger.
```

### 2.9 Plugin Architecture
```
Design a plugin architecture following OCP:
- Plugin discovery mechanism
- Plugin interface/protocol
- Plugin registration and loading
- Plugin lifecycle management
- Configuration per plugin
Example: notification system with email/SMS/push plugins.
```

### 2.10 Composition Over Inheritance
```
Refactor this deep inheritance hierarchy to use composition:
- Identify shared behaviors
- Extract behaviors into separate components
- Use composition to combine behaviors
- Favor has-a over is-a
- Increase flexibility and reduce coupling
Show before/after comparison.

[paste your inheritance code here]
```

## Category: Design Patterns (10 prompts)

### 2.11 Repository Pattern Implementation
```
Implement Repository pattern for data access layer:
- Generic repository interface with CRUD operations
- Concrete repositories for specific entities
- Unit of Work for transaction management
- Query specifications pattern
- Async/await support
- Type hints with generics
Example: UserRepository, ProductRepository.
```

### 2.12 Service Layer Pattern
```
Create Service layer between API and data layer:
- Business logic encapsulation
- Transaction boundaries
- Validation and error handling
- Dependency injection
- Type hints
- Separation from web framework
Example: UserService, OrderService.
```

### 2.13 CQRS Pattern
```
Implement CQRS (Command Query Responsibility Segregation):
- Separate read and write models
- Command handlers
- Query handlers
- Event sourcing integration (optional)
- Different optimizations for reads/writes
Example: e-commerce order system.
```

### 2.14 Event Sourcing
```
Design event sourcing system:
- Event store
- Event stream
- Aggregate roots
- Event replay for state reconstruction
- Snapshots for performance
- Type-safe events
Example: account balance tracking.
```

### 2.15 Saga Pattern
```
Implement Saga pattern for distributed transactions:
- Orchestration vs choreography
- Compensating transactions
- State machine for saga flow
- Error handling and rollback
- Eventual consistency
Example: order fulfillment across microservices.
```

### 2.16 API Gateway Pattern
```
Design API Gateway for microservices:
- Request routing
- Authentication/authorization
- Rate limiting
- Request/response transformation
- Circuit breaker integration
- Load balancing
Use FastAPI or similar framework.
```

### 2.17 Backend for Frontend (BFF)
```
Implement BFF pattern for different client types:
- Web BFF with full data
- Mobile BFF with optimized payloads
- Shared business logic
- Client-specific aggregations
- GraphQL integration option
Include performance considerations.
```

### 2.18 Strangler Fig Pattern
```
Design migration strategy using Strangler Fig pattern:
- Proxy/facade for legacy system
- Incremental feature migration
- Routing logic (legacy vs new)
- Rollback capability
- Testing strategy during migration
Example: migrate monolith to microservices.
```

### 2.19 Bulkhead Pattern
```
Implement Bulkhead pattern for fault isolation:
- Resource pool separation
- Thread pool per service
- Failure isolation
- Degraded mode operation
- Monitoring and alerting
Example: separate pools for critical vs non-critical operations.
```

### 2.20 Sidecar Pattern
```
Design Sidecar pattern for cross-cutting concerns:
- Logging sidecar
- Monitoring/metrics sidecar
- Service mesh integration
- Configuration management
- Independent deployment
Example: microservice with logging and metrics sidecars.
```

## Category: Architecture Patterns (10 prompts)

### 2.21 Layered Architecture
```
Design a layered architecture application:
- Presentation layer (API/UI)
- Business logic layer
- Data access layer
- Cross-cutting concerns (logging, auth)
- Clear layer dependencies (top-down only)
- Interface definitions between layers
Include folder structure and dependency diagram.
```

### 2.22 Clean Architecture
```
Implement Clean Architecture (Uncle Bob):
- Entities (domain models)
- Use cases (business rules)
- Interface adapters (controllers, presenters)
- Frameworks and drivers (outer layer)
- Dependency rule (inward only)
- Testable business logic
Example: order management system.
```

### 2.23 Hexagonal Architecture (Ports and Adapters)
```
Design system using Hexagonal Architecture:
- Core business logic (hexagon)
- Ports (interfaces)
- Adapters (implementations)
- Primary adapters (driving - API, UI)
- Secondary adapters (driven - DB, external services)
Show multiple adapters for same port.
```

### 2.24 Microservices Architecture
```
Design microservices architecture for e-commerce platform:
- Service boundaries (user, product, order, payment)
- Communication (sync REST, async messaging)
- Service discovery
- API gateway
- Database per service
- Distributed tracing
Include service interaction diagram.
```

### 2.25 Event-Driven Architecture
```
Design event-driven system:
- Event producers
- Event consumers
- Event bus (Kafka, RabbitMQ, SNS/SQS)
- Event schemas
- Eventual consistency handling
- Dead letter queues
Example: order processing with inventory, shipping, notification services.
```

### 2.26 Serverless Architecture
```
Design serverless application on AWS Lambda:
- Lambda functions for business logic
- API Gateway for HTTP endpoints
- DynamoDB for data
- S3 for storage
- EventBridge for events
- Step Functions for workflows
Include IAM policies and deployment config.
```

### 2.27 Monolith to Microservices Migration
```
Plan migration from monolith to microservices:
1. Analyze monolith dependencies
2. Identify bounded contexts
3. Extract services incrementally
4. Data decomposition strategy
5. Testing and rollback plan
6. Timeline and risk assessment
Use Strangler Fig pattern.
```

### 2.28 Multi-Tenant Architecture
```
Design multi-tenant SaaS application:
- Tenant isolation strategies (DB per tenant, schema per tenant, shared schema)
- Tenant identification (subdomain, header, JWT claim)
- Data partitioning
- Performance isolation
- Security considerations
Include pros/cons of each approach.
```

### 2.29 CQRS + Event Sourcing Architecture
```
Design system combining CQRS and Event Sourcing:
- Write model with event sourcing
- Read model with projections
- Event store
- Projection rebuilding
- Eventual consistency
- Scalability benefits
Example: financial trading platform.
```

### 2.30 Service Mesh Architecture
```
Design service mesh for microservices:
- Service-to-service communication
- Load balancing and routing
- Circuit breakers and retries
- mTLS security
- Observability (traces, metrics)
- Istio or Linkerd configuration
Include traffic management examples.
```

## Category: API Design (10 prompts)

### 2.31 RESTful API Design
```
Design RESTful API for blog platform following best practices:
- Resource naming (plural nouns)
- HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Status codes (200, 201, 400, 404, 500)
- Versioning (URL or header)
- Pagination, filtering, sorting
- HATEOAS links
Include OpenAPI/Swagger specification.
```

### 2.32 GraphQL API Design
```
Design GraphQL schema for e-commerce:
- Types, queries, mutations
- Relationships and nested data
- Input types for mutations
- Pagination (cursor-based)
- Error handling
- N+1 query prevention (DataLoader)
Include example queries and mutations.
```

### 2.33 API Versioning Strategy
```
Design API versioning strategy:
- URL versioning (v1, v2)
- Header versioning (Accept-Version)
- Breaking vs non-breaking changes
- Deprecation policy
- Migration guide for clients
- Backward compatibility
Example: migrate /v1/users to /v2/users with new fields.
```

### 2.34 API Authentication & Authorization
```
Implement API security:
- JWT authentication
- OAuth 2.0 flows
- API keys for service-to-service
- Role-based access control (RBAC)
- Refresh token mechanism
- Rate limiting per user/key
Include FastAPI security utilities.
```

### 2.35 API Rate Limiting
```
Implement rate limiting for API:
- Token bucket algorithm
- Rate limit per user/IP/API key
- Response headers (X-RateLimit-*)
- 429 Too Many Requests response
- Redis for distributed rate limiting
- Graceful degradation
Include sliding window implementation.
```

### 2.36 API Error Handling
```
Design comprehensive API error handling:
- Consistent error response format
- Error codes and messages
- Validation errors with field details
- Stack traces (dev vs prod)
- Error logging and monitoring
- Client-friendly error messages
Include RFC 7807 Problem Details format.
```

### 2.37 API Documentation
```
Create comprehensive API documentation:
- OpenAPI 3.0 specification
- Interactive documentation (Swagger UI)
- Authentication guide
- Request/response examples
- Error codes reference
- Rate limiting info
- Changelog and versioning
Use FastAPI or Flask-RESTX.
```

### 2.38 API Pagination
```
Implement pagination strategies:
- Offset-based pagination
- Cursor-based pagination
- Page-based pagination
- Response metadata (total, page, size)
- Link headers (next, prev, first, last)
- Performance considerations
Compare approaches and recommend best fit.
```

### 2.39 API Filtering and Sorting
```
Implement flexible filtering and sorting for REST API:
- Query parameters for filters
- Multiple filter operators (eq, gt, lt, like)
- Sort by multiple fields
- Type-safe query parsing
- SQL injection prevention
- Index recommendations for performance
Example: /api/users?age[gt]=18&sort=-created_at
```

### 2.40 API Caching Strategy
```
Design caching strategy for API:
- Cache-Control headers
- ETag for conditional requests
- Redis for response caching
- Cache invalidation strategies
- Cache key design
- CDN integration for static resources
Include edge cases and cache stampede prevention.
```

## Category: Database Design (10 prompts)

### 2.41 Database Normalization
```
Normalize this database schema to 3NF:
1. Identify functional dependencies
2. Eliminate repeating groups (1NF)
3. Remove partial dependencies (2NF)
4. Remove transitive dependencies (3NF)
5. Create ER diagram
Provide before/after comparison.

[paste your denormalized schema here]
```

### 2.42 Database Indexing Strategy
```
Design indexing strategy for this database:
- Identify query patterns
- Single-column vs composite indexes
- Index on foreign keys
- Covering indexes
- Partial indexes for filtered queries
- Index maintenance overhead
Provide EXPLAIN ANALYZE results.

[paste your schema and queries here]
```

### 2.43 Database Migration Strategy
```
Create database migration plan:
- Version-controlled migrations (Alembic, Flyway)
- Forward and backward migrations
- Data migrations vs schema migrations
- Zero-downtime deployment strategy
- Rollback plan
- Testing migrations
Include migration scripts and CI/CD integration.
```

### 2.44 Sharding Strategy
```
Design database sharding strategy for scalability:
- Shard key selection
- Horizontal vs vertical sharding
- Shard distribution algorithm
- Cross-shard queries handling
- Rebalancing strategy
- Consistency considerations
Example: user data sharding by user_id.
```

### 2.45 Database Replication
```
Design database replication architecture:
- Primary-replica configuration
- Read replicas for scaling reads
- Replication lag monitoring
- Failover strategy
- Write scaling (sharding)
- Consistency guarantees
Include PostgreSQL or MySQL configuration.
```

### 2.46 Query Optimization
```
Optimize these slow database queries:
1. Analyze EXPLAIN plans
2. Identify missing indexes
3. Rewrite inefficient queries
4. Consider materialized views
5. Measure performance improvement
6. Suggest schema changes if needed
Provide before/after execution times.

[paste your slow queries here]
```

### 2.47 Connection Pooling
```
Implement database connection pooling:
- Pool size configuration
- Connection timeout
- Connection validation
- Pool exhaustion handling
- Monitoring and metrics
- Thread safety
Use SQLAlchemy or asyncpg. Include tuning guidelines.
```

### 2.48 Transaction Management
```
Implement proper transaction management:
- ACID properties
- Isolation levels (Read Committed, Serializable)
- Deadlock handling
- Long-running transaction prevention
- Distributed transactions (2PC)
- Rollback strategies
Include SQLAlchemy or Django ORM examples.
```

### 2.49 Data Archival Strategy
```
Design data archival strategy:
- Archive criteria (age, status)
- Hot vs cold storage
- Archive table design
- Partitioning by date
- Archive retrieval mechanism
- Compliance requirements (GDPR)
Include automated archival jobs.
```

### 2.50 NoSQL vs SQL Decision
```
Analyze requirements and recommend database type:
- Structured vs unstructured data
- Query patterns
- Consistency requirements
- Scalability needs
- Transaction requirements
- Schema flexibility
Provide detailed comparison and recommendation.

[provide application requirements]
```

---

# 3. Machine Learning (50 Prompts)

## Category: Data Preprocessing (10 prompts)

### 3.1 Missing Data Handling
```
Analyze and handle missing data in this dataset:
1. Identify missing patterns (MCAR, MAR, MNAR)
2. Calculate missing percentage per feature
3. Recommend strategy: imputation vs deletion
4. Implement:
   - Mean/median/mode imputation
   - KNN imputation
   - Iterative imputer
5. Compare impact on model performance

[provide dataset description or sample]
```

### 3.2 Outlier Detection and Treatment
```
Detect and handle outliers in this dataset:
- Statistical methods: Z-score, IQR
- ML methods: Isolation Forest, LOF
- Visualization: box plots, scatter plots
- Treatment strategies: cap, remove, transform
- Document impact on distribution
- Compare before/after model metrics

[provide dataset]
```

### 3.3 Feature Scaling
```
Implement feature scaling for this dataset:
- StandardScaler (z-score normalization)
- MinMaxScaler (0-1 normalization)
- RobustScaler (median and IQR)
- When to use each method
- Handle mixed data types
- Fit on train, transform on test
Measure impact on model convergence.

[provide features]
```

### 3.4 Categorical Encoding
```
Encode categorical variables using multiple methods:
- Label Encoding for ordinal features
- One-Hot Encoding for nominal features
- Target Encoding for high cardinality
- Binary Encoding
- Handle unknown categories
- Prevent data leakage
Compare memory usage and model performance.

[provide categorical features]
```

### 3.5 Feature Engineering Pipeline
```
Create scikit-learn pipeline for preprocessing:
- Numerical: imputation → scaling
- Categorical: imputation → encoding
- ColumnTransformer for mixed types
- Custom transformers
- Save/load pipeline
- Integration with GridSearchCV
Include production-ready code.
```

### 3.6 Data Imbalance Handling
```
Address class imbalance in this classification dataset:
- Analyze class distribution
- Oversampling: SMOTE, ADASYN
- Undersampling: RandomUnderSampler
- Class weights in model
- Ensemble methods
- Evaluation metrics for imbalanced data
Compare approaches with F1, PR-AUC.

[provide dataset info]
```

### 3.7 Time Series Preprocessing
```
Preprocess time series data for ML:
- Handle missing timestamps
- Resample to regular intervals
- Create lag features
- Rolling window statistics
- Seasonal decomposition
- Stationarity testing (ADF test)
- Train/test split preserving time order
```

### 3.8 Text Preprocessing Pipeline
```
Create text preprocessing pipeline:
- Lowercasing
- Tokenization
- Stop words removal
- Stemming vs Lemmatization
- Special character handling
- TF-IDF or word embeddings
- Handle out-of-vocabulary words
Include spaCy or NLTK implementation.
```

### 3.9 Image Data Augmentation
```
Implement image augmentation pipeline:
- Rotation, flip, zoom, shift
- Color jittering
- Random crop and resize
- Mixup/Cutmix
- Albumentations or torchvision
- Augmentation only for training
- Validation of augmentations
Include PyTorch DataLoader integration.
```

### 3.10 Data Validation and Quality Checks
```
Create data quality validation framework:
- Schema validation (expected columns, types)
- Range checks for numerical features
- Uniqueness constraints
- Cross-field validation
- Data drift detection
- Great Expectations or custom validators
- CI/CD integration
Generate data quality report.
```

## Category: Model Training (10 prompts)

### 3.11 Train-Test Split Strategy
```
Implement proper train-test splitting:
- Random split with stratification
- Time-based split for temporal data
- K-fold cross-validation
- Stratified K-fold for classification
- Group K-fold for grouped data
- Validation set for hyperparameter tuning
Explain when to use each method.

[provide dataset characteristics]
```

### 3.12 XGBoost Model Training
```
Train and optimize XGBoost model:
- Define objective function
- Set hyperparameters: learning_rate, max_depth, n_estimators
- Early stopping with eval_set
- Feature importance analysis
- Handle class imbalance with scale_pos_weight
- Cross-validation for robustness
- Save model with joblib
Include full training pipeline.
```

### 3.13 Random Forest Optimization
```
Train Random Forest with optimal configuration:
- n_estimators, max_depth, min_samples_split tuning
- Feature importance and selection
- OOB score for validation
- Parallel training with n_jobs
- Memory optimization for large datasets
- Comparison with ExtraTrees
Provide before/after performance metrics.
```

### 3.14 Logistic Regression with Regularization
```
Implement Logistic Regression with regularization:
- L1 (Lasso) for feature selection
- L2 (Ridge) for reducing overfitting
- ElasticNet combining L1 and L2
- Cross-validation for C parameter
- Class weights for imbalance
- Probability calibration
- Coefficients interpretation
```

### 3.15 Neural Network with PyTorch
```
Build neural network classifier in PyTorch:
- Define model architecture (fully connected layers)
- Loss function and optimizer
- Training loop with batching
- Validation during training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Evaluation metrics
Include reproducibility (seed setting).
```

### 3.16 CNN for Image Classification
```
Implement CNN for image classification:
- Convolutional layers with ReLU
- Pooling layers
- Batch normalization
- Dropout for regularization
- Flatten and dense layers
- Data augmentation
- Transfer learning with pretrained models
- Fine-tuning strategy
Use PyTorch or TensorFlow.
```

### 3.17 RNN/LSTM for Sequences
```
Build LSTM for sequence prediction:
- Embedding layer for text or categorical sequences
- LSTM layers with hidden states
- Dropout and recurrent dropout
- Bidirectional LSTM
- Attention mechanism
- Sequence padding and masking
- Teacher forcing for training
Example: sentiment analysis or time series forecasting.
```

### 3.18 Transfer Learning
```
Implement transfer learning for image task:
- Load pretrained model (ResNet, VGG, EfficientNet)
- Freeze base layers
- Add custom classification head
- Fine-tuning strategy (gradual unfreezing)
- Learning rate differential
- Data augmentation specific to domain
- Compare from-scratch vs transfer learning
```

### 3.19 Ensemble Methods
```
Create ensemble of models:
- Voting classifier (hard/soft voting)
- Stacking with meta-learner
- Bagging for variance reduction
- Boosting for bias reduction
- Diversity in ensemble (different algorithms)
- Cross-validation for ensemble weights
Measure ensemble improvement.
```

### 3.20 AutoML with Optuna
```
Implement hyperparameter optimization with Optuna:
- Define objective function
- Search space for hyperparameters
- Pruning for early stopping of bad trials
- Parallel optimization
- Visualization of optimization history
- Best parameters and score
- Save study for resume
Example: optimize XGBoost or neural network.
```

## Category: Model Evaluation (10 prompts)

### 3.21 Classification Metrics
```
Calculate comprehensive classification metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC curve and AUC
- Precision-Recall curve
- Classification report
- Multi-class metrics (macro, micro, weighted avg)
- Per-class metrics
Visualize with matplotlib/seaborn.

[provide predictions and labels]
```

### 3.22 Regression Metrics
```
Evaluate regression model with multiple metrics:
- MSE, RMSE, MAE
- R² score
- Adjusted R²
- MAPE (Mean Absolute Percentage Error)
- Residual analysis and plots
- Prediction vs actual scatter plot
- QQ plot for normality
Interpret results and suggest improvements.

[provide predictions and actuals]
```

### 3.23 Cross-Validation Strategy
```
Implement robust cross-validation:
- K-fold cross-validation
- Stratified K-fold for classification
- Time series cross-validation
- Leave-one-out CV for small datasets
- Repeated cross-validation
- Calculate mean and std of metrics
- Statistical significance testing
Report confidence intervals.
```

### 3.24 Learning Curves
```
Generate and analyze learning curves:
- Training and validation loss/accuracy
- Detect overfitting (high variance)
- Detect underfitting (high bias)
- Determine if more data helps
- Plot with matplotlib
- Recommend solutions based on curves
Include both loss and metric curves.
```

### 3.25 Feature Importance Analysis
```
Analyze feature importance using multiple methods:
- Tree-based importance (XGBoost, RF)
- Permutation importance
- SHAP values
- LIME for local interpretability
- Correlation analysis
- Feature ablation
Visualize top features and recommend feature selection.
```

### 3.26 Model Comparison
```
Compare multiple models systematically:
- Train baseline and candidate models
- Cross-validation scores
- Statistical significance testing (t-test, Wilcoxon)
- Training time comparison
- Inference time comparison
- Model complexity comparison
Create comparison table and recommend best model.

[provide models to compare]
```

### 3.27 Hyperparameter Tuning with GridSearch
```
Perform hyperparameter tuning:
- Define parameter grid
- GridSearchCV or RandomizedSearchCV
- Cross-validation strategy
- Scoring metric
- Parallel processing
- Best parameters and score
- Refit on full training data
Include code for multiple models.
```

### 3.28 Error Analysis
```
Perform detailed error analysis:
- Identify misclassified samples
- Analyze error patterns
- Feature distributions for errors
- Confusion between specific classes
- Edge cases and corner cases
- Recommend data collection or feature engineering
Provide actionable insights.

[provide model and test data]
```

### 3.29 Calibration Assessment
```
Assess and improve model calibration:
- Reliability diagram (calibration curve)
- Expected Calibration Error (ECE)
- Calibration with Platt scaling
- Calibration with isotonic regression
- Compare before/after calibration
- Application to probability predictions
Critical for medical/financial applications.
```

### 3.30 A/B Testing Framework
```
Design A/B test for model comparison:
- Hypothesis definition
- Sample size calculation
- Random assignment
- Metric tracking (online metrics)
- Statistical significance testing
- Confidence intervals
- Decision criteria (deploy vs rollback)
Include monitoring dashboard design.
```

## Category: Deep Learning (10 prompts)

### 3.31 CNN Architecture Design
```
Design custom CNN architecture for specific task:
- Input shape considerations
- Convolutional blocks (conv + bn + relu + pool)
- Feature map progression
- Global average pooling vs flatten
- Classification head
- Parameter count optimization
- Receptive field analysis
Include architecture diagram and PyTorch code.

[provide task requirements]
```

### 3.32 Batch Normalization Implementation
```
Implement and explain batch normalization:
- BN between conv/linear and activation
- Training vs inference mode
- Running statistics (mean, var)
- Impact on learning rate
- Batch size considerations
- Compare with Layer Normalization
- Group Normalization for small batches
Demonstrate with experiments.
```

### 3.33 Dropout Regularization
```
Implement dropout for regularization:
- Dropout rate selection (0.2-0.5)
- Placement in network
- Difference in train vs eval mode
- Comparison with other regularization
- Dropout in RNN (recurrent dropout)
- DropConnect variant
Measure impact on overfitting.
```

### 3.34 Learning Rate Scheduling
```
Implement learning rate schedules:
- Step decay
- Exponential decay
- Cosine annealing
- Warmup + decay
- ReduceLROnPlateau
- Cyclical learning rates
- One-cycle policy
Visualize schedules and compare convergence.
```

### 3.35 Data Augmentation for Images
```
Implement comprehensive image augmentation:
- Geometric: rotation, flip, crop, zoom
- Color: brightness, contrast, saturation
- Noise injection
- Cutout, RandomErasing
- Mixup, Cutmix
- AutoAugment policies
Compare impact on validation accuracy.
```

### 3.36 Transfer Learning Fine-tuning
```
Fine-tune pretrained model:
- Load pretrained weights (ImageNet)
- Replace classification head
- Freeze base layers initially
- Train head with high LR
- Unfreeze base layers gradually
- Fine-tune with low LR
- Layer-wise learning rate decay
Measure improvement over from-scratch.
```

### 3.37 Attention Mechanism
```
Implement attention mechanism:
- Self-attention
- Multi-head attention
- Positional encoding
- Attention weights visualization
- Application in NLP (Transformer)
- Application in vision (Vision Transformer)
Include PyTorch implementation.
```

### 3.38 ResNet Skip Connections
```
Implement ResNet-style architecture:
- Residual blocks with skip connections
- Identity vs projection shortcuts
- Bottleneck design
- Batch normalization placement
- Pre-activation vs post-activation
- Explain gradient flow benefits
Compare with plain CNN.
```

### 3.39 Model Compression
```
Compress neural network for deployment:
- Pruning (magnitude-based, structured)
- Quantization (INT8, mixed precision)
- Knowledge distillation
- Low-rank factorization
- Measure size reduction
- Measure speed improvement
- Accuracy trade-off analysis
```

### 3.40 Multi-GPU Training
```
Implement distributed training:
- DataParallel vs DistributedDataParallel
- Gradient synchronization
- Batch size scaling
- Learning rate scaling
- Batch normalization across GPUs
- Model checkpointing
- Speedup measurement
Use PyTorch or TensorFlow.
```

## Category: MLOps & Deployment (10 prompts)

### 3.41 Model Serialization
```
Implement model save/load strategies:
- Pickle for scikit-learn models
- Joblib for large arrays
- torch.save for PyTorch models
- SavedModel for TensorFlow
- ONNX for interoperability
- Model versioning
- Metadata storage (hyperparameters, metrics)
Include production best practices.
```

### 3.42 FastAPI Model Serving
```
Deploy ML model with FastAPI:
- Load model at startup
- POST endpoint for predictions
- Request/response Pydantic models
- Batch prediction support
- Error handling
- Logging predictions
- Health check endpoint
- Docker containerization
```

### 3.43 Model Monitoring
```
Implement model monitoring in production:
- Input data drift detection
- Output drift detection
- Model performance metrics tracking
- Latency monitoring
- Error rate monitoring
- Alerting on anomalies
- Retraining triggers
Use Prometheus, Grafana, or MLflow.
```

### 3.44 Feature Store
```
Design feature store for ML:
- Feature computation and storage
- Online vs offline features
- Feature versioning
- Point-in-time correct joins
- Feature serving API
- Feature monitoring
- Feast or custom implementation
Include batch and real-time features.
```

### 3.45 ML Pipeline with Airflow
```
Create ML pipeline with Airflow:
- Data extraction task
- Preprocessing task
- Model training task
- Model evaluation task
- Model deployment task
- Dependencies and scheduling
- Failure handling and retries
- XCom for task communication
```

### 3.46 Experiment Tracking with MLflow
```
Implement experiment tracking:
- Log parameters, metrics, artifacts
- Compare experiments
- Model registry
- Model staging (dev, staging, prod)
- Model versioning
- Reproducibility (seed, env, code version)
- UI for experiment visualization
```

### 3.47 A/B Testing Deployment
```
Implement gradual model rollout:
- Canary deployment (10% traffic)
- A/B test with random assignment
- Shadow mode (parallel prediction)
- Metric comparison (latency, accuracy)
- Automated rollback on degradation
- Feature flags for control
Use Kubernetes or cloud services.
```

### 3.48 Model Explainability
```
Implement model explainability:
- SHAP values for global and local explanations
- LIME for local interpretability
- Feature importance visualization
- Partial dependence plots
- Individual conditional expectation
- Counterfactual explanations
Critical for regulated industries.
```

### 3.49 CI/CD for ML Models
```
Setup CI/CD pipeline for ML:
- Code quality checks (lint, format, type check)
- Unit tests for preprocessing
- Model training on CI
- Model validation tests
- Performance regression tests
- Automated deployment on merge
- GitHub Actions or Jenkins
```

### 3.50 Model Documentation
```
Create comprehensive model documentation:
- Model card with purpose, limitations, metrics
- Training data description
- Feature descriptions
- Hyperparameters
- Performance metrics (overall and per-group)
- Fairness and bias analysis
- Deployment instructions
- Monitoring and maintenance plan
Follow model card standards.
```

---

# 4. Quantitative Finance (50 Prompts)

## Category: Trading Strategy Development (10 prompts)

### 4.1 Mean Reversion Strategy
```
Implement mean reversion trading strategy:
- Calculate moving average and standard deviation
- Z-score calculation for entry signals
- Entry threshold (e.g., |z| > 2)
- Exit threshold (e.g., |z| < 0.5)
- Position sizing rules
- Stop loss and take profit
- Backtesting with realistic assumptions
- Performance metrics (Sharpe, Calmar, max DD)
Include complete Python implementation.
```

### 4.2 Momentum Strategy
```
Develop momentum trading strategy:
- Lookback period selection
- Momentum calculation (price change, returns)
- Cross-sectional vs time-series momentum
- Entry/exit rules
- Universe selection
- Risk management
- Transaction cost modeling
- Walk-forward optimization
Backtest on multiple assets.
```

### 4.3 Pairs Trading Strategy
```
Implement statistical arbitrage pairs trading:
- Pair selection criteria (cointegration)
- Spread calculation
- Half-life estimation
- Z-score based entry/exit
- Hedge ratio calculation
- Dynamic rebalancing
- Risk limits per pair
- Portfolio of pairs management
Include Johansen test for cointegration.
```

### 4.4 Trend Following Strategy
```
Create trend following system:
- Trend identification (MA crossover, ADX)
- Breakout detection
- Entry signals (trend confirmation)
- Trailing stop loss
- Position sizing (volatility-adjusted)
- Multiple timeframe analysis
- Regime filtering
- Performance across market conditions
```

### 4.5 Statistical Arbitrage Strategy
```
Develop stat arb strategy:
- Factor model for expected returns
- Residual calculation
- Mean reversion on residuals
- Long/short portfolio construction
- Risk factor neutrality
- Turnover control
- Transaction cost optimization
- Alpha decay analysis
Include PCA for factor extraction.
```

### 4.6 Market Making Strategy
```
Design market making algorithm:
- Bid-ask spread calculation
- Inventory management
- Adverse selection handling
- Mid-price estimation
- Order placement strategy
- Profit from spread capture
- Risk limits
- Handling market orders
Simulate with order book data.
```

### 4.7 Options Trading Strategy
```
Develop options trading strategy:
- Implied volatility calculation
- Greeks computation (delta, gamma, vega, theta)
- Volatility arbitrage
- Delta hedging
- Iron condor or straddle strategy
- Risk management
- P&L attribution
- Backtesting with options data
```

### 4.8 Multi-Factor Strategy
```
Implement multi-factor equity strategy:
- Factor selection (value, momentum, quality, size)
- Factor calculation
- Z-score normalization
- Factor combining (equal-weight, optimized)
- Universe screening
- Portfolio construction (long/short)
- Rebalancing frequency
- Risk-adjusted performance
```

### 4.9 Sentiment-Based Strategy
```
Create trading strategy based on sentiment:
- News sentiment extraction (NLP)
- Social media sentiment
- Sentiment aggregation
- Signal generation from sentiment
- Combining with price data
- Timing of trades
- Backtesting challenges (look-ahead bias)
- Risk management
Use Twitter API or news APIs.
```

### 4.10 Machine Learning Strategy
```
Develop ML-enhanced trading strategy:
- Feature engineering (technical, fundamental)
- Label creation (forward returns)
- Model selection (XGBoost, LSTM)
- Walk-forward validation
- Prediction to signal conversion
- Position sizing based on prediction confidence
- Risk management
- Performance comparison with traditional strategies
```

## Category: Backtesting & Validation (10 prompts)

### 4.11 Professional Backtesting Framework
```
Build comprehensive backtesting engine:
- Event-driven architecture
- Market data handling
- Order execution simulation
- Portfolio tracking
- Realistic transaction costs
- Slippage modeling
- Partial fills
- Position reconciliation
- Performance metrics calculation
Include extensible design for multiple strategies.
```

### 4.12 Transaction Cost Modeling
```
Implement realistic transaction cost model:
- Commission structure
- Bid-ask spread cost
- Slippage (linear, sqrt, power law)
- Market impact (temporary, permanent)
- Liquidity considerations
- Volume-dependent costs
- Compare impact on strategy profitability
Calibrate with real execution data.
```

### 4.13 Walk-Forward Optimization
```
Implement walk-forward optimization:
- In-sample optimization window
- Out-of-sample testing window
- Rolling windows
- Parameter stability analysis
- Robustness testing
- Anchored vs rolling windows
- Performance degradation tracking
Compare with in-sample overfitting.
```

### 4.14 Monte Carlo Simulation
```
Perform Monte Carlo simulation for strategy:
- Bootstrap returns
- Random trade permutation
- Parameter perturbation
- Confidence intervals for metrics
- Risk of ruin estimation
- Drawdown distribution
- Optimal position sizing
Visualize distribution of outcomes.
```

### 4.15 Bias Detection in Backtests
```
Identify and eliminate backtesting biases:
- Survivorship bias (delisted stocks)
- Look-ahead bias (using future data)
- Selection bias (cherry-picking)
- Overfitting to historical data
- Data snooping
- Backtest audit checklist
Provide bias-free backtest implementation.
```

### 4.16 Slippage and Fill Modeling
```
Model realistic order fills:
- Immediate or cancel (IOC) orders
- Fill probability based on volume
- Price improvement
- Partial fills
- Queue position in order book
- Aggressive vs passive orders
- Compare market vs limit orders
Calibrate with level 2 data.
```

### 4.17 Performance Attribution
```
Implement performance attribution:
- Alpha vs beta decomposition
- Factor contribution analysis
- Timing vs selection attribution
- Sector/industry attribution
- Long vs short contribution
- Transaction cost attribution
- Visualization of sources of return
Use Brinson model or factor models.
```

### 4.18 Risk-Adjusted Metrics
```
Calculate comprehensive risk-adjusted performance metrics:
- Sharpe ratio (risk-free rate adjustment)
- Sortino ratio (downside deviation)
- Calmar ratio (return / max drawdown)
- MAR ratio (CAGR / max drawdown)
- Information ratio
- Omega ratio
- Tail ratio
Interpret and compare strategies.
```

### 4.19 Drawdown Analysis
```
Perform detailed drawdown analysis:
- Maximum drawdown calculation
- Drawdown duration
- Underwater curve
- Recovery time
- Drawdown distribution
- Conditional drawdown
- Value at Risk (VaR)
- Expected shortfall (CVaR)
Visualize drawdown periods.
```

### 4.20 Out-of-Sample Testing
```
Implement rigorous out-of-sample validation:
- Train/validation/test split
- Temporal holdout set
- No parameter tuning on test set
- Paper trading simulation
- Live-to-backtest tracking
- Performance degradation detection
- Retraining triggers
Document validation methodology.
```

## Category: Risk Management (10 prompts)

### 4.21 Kelly Criterion Position Sizing
```
Implement Kelly Criterion for position sizing:
- Full Kelly formula: f = (p*b - q) / b
- Fractional Kelly for safety
- Estimation of win probability and odds
- Dynamic adjustment based on confidence
- Portfolio-level Kelly
- Comparison with fixed fractional
- Risk of ruin analysis
Handle parameter uncertainty.
```

### 4.22 Value at Risk (VaR)
```
Calculate Value at Risk using multiple methods:
- Historical simulation VaR
- Parametric VaR (variance-covariance)
- Monte Carlo VaR
- Conditional VaR (Expected Shortfall)
- Different confidence levels (95%, 99%)
- Backtesting VaR predictions
- Regulatory capital requirements
Compare methods and recommend.
```

### 4.23 Stop Loss and Take Profit
```
Design stop loss and take profit system:
- Fixed percentage stops
- Volatility-based stops (ATR multiple)
- Trailing stops
- Time-based stops
- Technical level stops
- Take profit targets
- Risk/reward ratio optimization
- Impact on strategy performance
Backtest with and without stops.
```

### 4.24 Portfolio Allocation
```
Implement portfolio allocation strategies:
- Equal weight
- Inverse volatility weighting
- Risk parity allocation
- Mean-variance optimization
- Black-Litterman model
- Hierarchical risk parity
- Constraints (position limits, sector limits)
Compare allocations and rebalancing frequency.
```

### 4.25 Correlation-Based Risk
```
Manage correlation risk in portfolio:
- Correlation matrix calculation
- Rolling correlation analysis
- Correlation breakdown detection
- Diversification ratio
- Maximum diversification portfolio
- Clustering of correlated assets
- Correlation stress testing
Recommend correlation-aware allocation.
```

### 4.26 Leverage Management
```
Design leverage management system:
- Target leverage calculation
- Dynamic leverage adjustment
- Volatility-based deleveraging
- Drawdown-triggered deleveraging
- Margin requirements
- Forced liquidation prevention
- Leverage costs modeling
- Risk-adjusted returns with leverage
```

### 4.27 Scenario Analysis
```
Perform scenario analysis for portfolio:
- Historical stress scenarios (2008, 2020)
- Hypothetical scenarios
- Factor shock scenarios
- Correlation breakdown scenarios
- Portfolio impact estimation
- Hedging strategies for scenarios
- Scenario probability weighting
Document risk mitigation plans.
```

### 4.28 Greeks Management (Options)
```
Manage options portfolio Greeks:
- Delta hedging frequency
- Gamma risk management
- Vega exposure limits
- Theta decay optimization
- Rho interest rate risk
- Portfolio-level Greeks
- Dynamic hedging strategy
- P&L attribution to Greeks
```

### 4.29 Liquidity Risk Management
```
Implement liquidity risk framework:
- Liquidity metrics (bid-ask spread, volume)
- Market impact estimation
- Liquidation time estimation
- Liquidity-adjusted VaR
- Concentration limits
- Emergency liquidation plan
- Alternative liquidity sources
Model illiquid asset holdings.
```

### 4.30 Risk Budgeting
```
Implement risk budgeting framework:
- Total risk budget definition
- Risk allocation across strategies
- Marginal contribution to risk
- Risk-adjusted capital allocation
- Monitoring risk utilization
- Rebalancing based on risk
- Dynamic risk budget adjustment
Optimize for risk-adjusted returns.
```

## Category: Portfolio Optimization (10 prompts)

### 4.31 Mean-Variance Optimization
```
Implement Markowitz mean-variance optimization:
- Expected returns estimation
- Covariance matrix estimation
- Efficient frontier calculation
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Risk-free rate consideration
- Short sales constraints
- Position limits
Visualize efficient frontier.
```

### 4.32 Black-Litterman Model
```
Implement Black-Litterman portfolio optimization:
- Market equilibrium returns
- Investor views incorporation
- View confidence specification
- Posterior expected returns
- Uncertainty adjustment
- Portfolio weights calculation
- Comparison with mean-variance
Handle multiple views and constraints.
```

### 4.33 Risk Parity Portfolio
```
Create risk parity portfolio:
- Equal risk contribution from each asset
- Volatility-based weights
- Risk budgeting
- Leverage to target return
- Rebalancing frequency
- Transaction cost considerations
- Comparison with equal weight and mean-variance
Include hierarchical risk parity.
```

### 4.34 Maximum Diversification
```
Implement maximum diversification portfolio:
- Diversification ratio calculation
- Optimization problem formulation
- Constraint handling
- Comparison with other methods
- Stability of weights
- Rebalancing trigger
- Performance in different market regimes
Explain diversification benefits.
```

### 4.35 Hierarchical Risk Parity
```
Implement Hierarchical Risk Parity (HRP):
- Correlation matrix
- Hierarchical clustering (linkage methods)
- Dendrogram visualization
- Recursive bisection allocation
- Quasi-diagonalization
- Compare with traditional risk parity
- Stability and robustness
Demonstrate with multi-asset portfolio.
```

### 4.36 Robust Portfolio Optimization
```
Implement robust optimization under uncertainty:
- Parameter uncertainty
- Worst-case optimization
- Shrinkage estimators for covariance
- Regularization techniques
- Box constraints on parameters
- Resampled efficient frontier
- Compare with classical optimization
Handle estimation error.
```

### 4.37 Factor-Based Optimization
```
Optimize portfolio using factor models:
- Factor exposure calculation
- Factor risk budgeting
- Style tilts (value, momentum, quality)
- Factor-neutral portfolio
- Factor timing strategy
- Multi-factor optimization
- Performance attribution
Use Fama-French or custom factors.
```

### 4.38 Transaction Cost-Aware Optimization
```
Optimize portfolio considering transaction costs:
- Turnover penalty in objective
- Quadratic transaction costs
- Rebalancing frequency optimization
- Trade-off between optimal weights and costs
- Drift-based rebalancing
- Cost-benefit analysis
- No-trade regions
Measure cost reduction vs performance.
```

### 4.39 Multi-Period Optimization
```
Implement multi-period portfolio optimization:
- Dynamic programming approach
- Rebalancing over time horizon
- Stochastic returns modeling
- Terminal wealth objective
- Intermediate consumption
- Time-varying constraints
- Comparison with single-period
Handle transaction costs across periods.
```

### 4.40 Portfolio Rebalancing Strategy
```
Design portfolio rebalancing strategy:
- Calendar-based rebalancing
- Threshold-based rebalancing
- Volatility-triggered rebalancing
- Tax-aware rebalancing
- Transaction cost minimization
- Rebalancing frequency optimization
- Performance impact analysis
Backtest different rebalancing rules.
```

## Category: Market Microstructure (10 prompts)

### 4.41 Order Book Analysis
```
Analyze limit order book dynamics:
- Order book reconstruction
- Bid-ask spread calculation
- Order book imbalance
- Depth at different levels
- Order book shape
- Quote intensity
- Cancellation rates
- Predictive signals from order book
Use L2/L3 market data.
```

### 4.42 Market Impact Modeling
```
Model market impact of trades:
- Almgren-Chriss model
- Square-root market impact
- Permanent vs temporary impact
- Impact as function of size and volume
- Meta-order impact
- Calibration with execution data
- Optimal execution schedule
Minimize implementation shortfall.
```

### 4.43 TWAP/VWAP Execution
```
Implement TWAP and VWAP execution algorithms:
- Time-weighted average price (TWAP) schedule
- Volume-weighted average price (VWAP) schedule
- Historical volume pattern
- Participation rate
- Adaptive execution
- Performance measurement vs benchmark
- Slippage analysis
Compare TWAP vs VWAP performance.
```

### 4.44 Smart Order Routing
```
Design smart order routing (SOR) system:
- Venue selection (exchanges, dark pools)
- Liquidity aggregation
- Fee structure optimization
- Latency considerations
- Order splitting strategy
- Fill probability estimation
- Routing logic optimization
Measure execution quality improvement.
```

### 4.45 Bid-Ask Spread Analysis
```
Analyze bid-ask spread components:
- Order processing costs
- Inventory holding costs
- Adverse selection costs
- Roll model for spread
- Quoted vs effective spread
- Spread as function of volatility
- Intraday spread patterns
- Spread predictability
Use high-frequency data.
```

### 4.46 Trade Classification (Lee-Ready)
```
Implement trade classification algorithm:
- Lee-Ready algorithm (tick rule, quote rule)
- Trade signing (buy vs sell)
- Order flow imbalance
- Informed vs uninformed trades
- Volume-weighted order flow
- Predictive power for returns
- Application in execution algorithms
Validate with labeled trades.
```

### 4.47 Price Discovery Analysis
```
Analyze price discovery process:
- Contribution of trades vs quotes
- Information share (Hasbrouck, Gonzalo-Granger)
- Lead-lag relationships between venues
- Price efficiency metrics
- Informed trading detection
- Incorporation of news
- High-frequency price dynamics
Multi-venue analysis.
```

### 4.48 Latency Arbitrage Detection
```
Detect and measure latency arbitrage:
- Cross-venue price discrepancies
- Latency measurement
- Quote update speed
- Arbitrage opportunity duration
- Profitability estimation
- Impact on market quality
- Defense mechanisms
Analyze with co-located data.
```

### 4.49 Queue Position Estimation
```
Estimate queue position in order book:
- Historical fill rates
- Queue dynamics modeling
- Order priority rules (price-time, pro-rata)
- Cancellation ahead in queue
- Fill probability estimation
- Impact on order placement strategy
- Optimal limit order pricing
Use for execution optimization.
```

### 4.50 Transaction Cost Analysis (TCA)
```
Implement comprehensive TCA framework:
- Slippage measurement (arrival price, decision price)
- Market impact decomposition
- Timing cost
- Opportunity cost
- Benchmark comparison (VWAP, TWAP, closing)
- Venue analysis
- Broker performance
- Best execution reporting
Dashboard for ongoing monitoring.
```

---

# 5. Senior Quantitative Developer (50 Prompts)

## Category: Low-Latency Infrastructure (10 prompts)

### 5.1 Market Data Feed Optimization
```
Reduce P99 latency of market data handler from 200µs to <50µs:
- Profile with perf/VTune to identify hot paths
- Eliminate heap allocations on critical path
- Use lock-free data structures (SPSC queue)
- Cache-aware memory layout (alignment, false sharing)
- Zero-copy I/O with ring buffers
- SIMD vectorization where applicable
- Generate flamegraph before/after
- Measure with hardware counters (cache misses, branch mispredicts)
Provide P50/P95/P99 latency distribution.
```

### 5.2 Network Stack Optimization
```
Optimize network stack for ultra-low latency:
- Kernel bypass with DPDK or XDP
- NIC tuning (IRQ affinity, RSS, flow steering)
- Busy polling vs interrupt-driven
- TCP vs UDP vs raw sockets trade-offs
- Jumbo frames configuration
- SO_BUSY_POLL socket option
- TSO/GSO/GRO offload settings
- Measure packet drop rate and jitter
Include before/after latency distribution.
```

### 5.3 Order Book Reconstruction Performance
```
Implement high-performance order book reconstruction:
- Incremental updates vs full snapshots
- Cache-friendly data structure design
- Memory pool for order objects
- Price level aggregation optimization
- Gap detection and recovery
- Deterministic replay for testing
- Benchmark: 1M+ updates/second
- Profile memory allocations
Provide throughput and latency metrics.
```

### 5.4 C++ Hot Path Optimization
```
Optimize C++ trading strategy hot path:
- Profile with perf and identify bottlenecks
- Eliminate virtual function calls
- Template metaprogramming for zero-cost abstractions
- [[likely]]/[[unlikely]] branch hints
- Prefetching for cache optimization
- Avoid exception handling on hot path
- -O3 -march=native compiler flags
- Link-time optimization (LTO)
Measure cycles per operation before/after.
```

### 5.5 Lock-Free Data Structures
```
Implement lock-free SPSC and MPSC queues:
- Single producer, single consumer (SPSC) queue
- Multiple producer, single consumer (MPSC) queue
- Atomic operations with memory ordering
- Cache line padding to avoid false sharing
- Backpressure handling (bounded queue)
- Benchmarking against std::mutex
- Correctness testing with ThreadSanitizer
Compare latency and throughput vs locks.
```

### 5.6 NUMA-Aware Allocation
```
Optimize for NUMA (Non-Uniform Memory Access):
- Identify NUMA topology with hwloc
- Thread and memory affinity (numa_set_preferred)
- Local memory allocation strategies
- Measure remote vs local memory access
- CPU pinning for critical threads
- Inter-socket communication minimization
- NUMA balancing policies
Provide access latency measurements per node.
```

### 5.7 Cache Optimization
```
Optimize CPU cache utilization:
- Cache line size awareness (64 bytes)
- Data structure layout for spatial locality
- Loop optimization (blocking, tiling)
- Prefetching (__builtin_prefetch)
- Avoid false sharing (alignas, padding)
- Hot/cold data separation
- Measure with perf: L1/L2/L3 cache misses
Provide cache miss rate before/after.
```

### 5.8 Time Synchronization (PTP)
```
Implement precise time synchronization with PTP:
- PTP daemon configuration (ptp4l, phc2sys)
- Hardware timestamping (NIC support)
- Clock drift monitoring and compensation
- Leap second handling
- End-to-end latency measurement
- Offset and jitter tracking
- NTP fallback mechanism
- Dashboard for time synchronization health
Report drift and offset distributions.
```

### 5.9 Backpressure Handling
```
Implement bounded backpressure for peak load:
- Detect queue fullness
- Drop vs block vs overflow strategies
- Prioritize critical messages
- Flow control signaling
- Buffering dimensioning (queue sizes)
- Graceful degradation under load
- Monitoring queue depth and drops
Test with 2× peak volume and measure behavior.
```

### 5.10 Deterministic Replay System
```
Build deterministic replay system for debugging:
- Capture all inputs (market data, orders)
- Sequence numbers and timestamps
- State snapshots for checkpointing
- Deterministic execution (no time(), random())
- Replay with identical behavior
- Performance profiling during replay
- Automated regression testing
- Integration with CI/CD
Verify bit-exact reproduction.
```

## Category: Performance Engineering (10 prompts)

### 5.11 Flamegraph Analysis
```
Generate and analyze flamegraphs for trading system:
- Capture stack traces with perf
- Generate flamegraph with flamegraph.pl
- Identify hot functions (>1% CPU time)
- Analyze on-CPU vs off-CPU (waiting)
- Differential flamegraphs (before/after optimization)
- Annotate source code with profiling data
- Integrate with CI for performance regression
Provide top 10 optimization opportunities.
```

### 5.12 Latency Budgeting
```
Create latency budget for tick-to-trade pipeline:
- Break down: network, parsing, logic, execution
- Allocate time budget per component
- Measure actual vs budget with tracing
- Identify violators with percentile analysis
- End-to-end latency tracking (P50/P95/P99)
- Dashboard for latency breakdown
- Alerting on budget violations
Target: <100µs total with per-component budgets.
```

### 5.13 Throughput Optimization
```
Optimize system throughput to handle peak volume:
- Batch processing where possible
- Pipeline parallelism
- SIMD for data processing
- Reduce per-message overhead
- Efficient serialization (FlatBuffers, Cap'n Proto)
- Benchmark: messages/second, GBps
- Load testing with realistic workload
- Identify bottlenecks with profiling
Target: 10M+ messages/second.
```

### 5.14 Memory Allocator Optimization
```
Optimize memory allocation strategy:
- Custom allocators for trading objects
- Memory pools (order pool, message pool)
- Arena allocators for temporary data
- jemalloc vs tcmalloc vs custom
- Reduce fragmentation
- Profile with massif (Valgrind)
- Measure allocation overhead (cycles)
Eliminate allocations on hot path.
```

### 5.15 Compiler Optimization Techniques
```
Leverage advanced compiler optimizations:
- Profile-guided optimization (PGO)
- Link-time optimization (LTO)
- Architecture-specific flags (-march=native)
- Function inlining hints (always_inline, noinline)
- Devirtualization opportunities
- Compiler intrinsics for SIMD
- Static analysis (clang-tidy, cppcheck)
- Benchmark with different optimization levels
Compare binary size and performance.
```

### 5.16 Branch Prediction Optimization
```
Optimize for branch prediction:
- Identify mispredicted branches with perf
- [[likely]]/[[unlikely]] attributes (C++20)
- Reorder code to favor common path
- Eliminate branches with branchless code
- Lookup tables instead of conditionals
- Profile-guided reordering
- Measure branch misprediction rate
Target: <1% branch misprediction rate.
```

### 5.17 SIMD Vectorization
```
Vectorize computational kernels with SIMD:
- Identify vectorizable loops
- Compiler auto-vectorization (pragmas, flags)
- Intrinsics (AVX2, AVX-512)
- Data layout for vectorization (SoA vs AoS)
- Alignment requirements
- Benchmark scalar vs vector versions
- Handle remainder (tail processing)
Example: calculate VWAP for 1000 symbols.
```

### 5.18 Instruction-Level Parallelism
```
Maximize instruction-level parallelism (ILP):
- Unroll loops to expose parallelism
- Reorder independent operations
- Reduce dependencies between instructions
- Pipelining considerations
- Measure with hardware counters (IPC)
- Profile with Intel VTune
- Minimize pipeline stalls
Target: IPC (instructions per cycle) > 2.0.
```

### 5.19 System Call Minimization
```
Minimize system calls on critical path:
- Use io_uring for async I/O (Linux)
- Batch operations where possible
- User-space networking (DPDK)
- Memory-mapped files for IPC
- futex for synchronization
- Profile with strace to identify syscalls
- Measure latency overhead per syscall
Target: zero syscalls on hot path.
```

### 5.20 Benchmarking Framework
```
Build comprehensive benchmarking framework:
- Microbenchmarks with Google Benchmark
- Warm-up iterations
- Statistical analysis (mean, median, percentiles)
- Outlier detection and filtering
- Regression detection in CI
- Comparison with baseline
- Automated reporting
- Integration with performance dashboard
Track performance over time (git commits).
```

## Category: Production Hardening (10 prompts)

### 5.21 Circuit Breaker Implementation
```
Implement circuit breaker for external dependencies:
- States: CLOSED, OPEN, HALF_OPEN
- Failure threshold configuration
- Timeout and retry logic
- Exponential backoff
- Health check probes
- Metrics (failure rate, state transitions)
- Graceful degradation
- Dashboard and alerting
Test with fault injection (chaos engineering).
```

### 5.22 Graceful Shutdown
```
Implement graceful shutdown procedure:
- Signal handling (SIGTERM, SIGINT)
- Drain in-flight requests
- Close connections cleanly
- Flush buffers and logs
- Save state for recovery
- Timeout for forced shutdown
- Health check returns unhealthy
- No data loss guarantee
Test with kill -TERM and verify clean shutdown.
```

### 5.23 Gap Detection and Recovery
```
Implement gap detection and recovery for market data:
- Sequence number tracking
- Gap detection algorithm
- Recovery request mechanism
- Replay from snapshot + incremental
- Out-of-order message handling
- Duplicate detection
- Recovery timeout and fallback
- Metrics (gap frequency, recovery time)
Test with simulated gaps.
```

### 5.24 Idempotency and Deduplication
```
Ensure idempotent processing and deduplication:
- Unique message IDs
- Idempotency key for orders
- Deduplication window (time-based, count-based)
- Persistent state for crash recovery
- At-most-once vs at-least-once delivery
- Exactly-once semantics where critical
- Test with duplicate messages
- Measure duplicate detection rate
```

### 5.25 Health Check Endpoints
```
Implement comprehensive health checks:
- Liveness check (process is running)
- Readiness check (ready to serve traffic)
- Dependency health (database, market data)
- Latency SLO check (P99 < threshold)
- Disk space, memory, CPU checks
- HTTP endpoint for Kubernetes probes
- Detailed status in response
- Alerting on health check failures
```

### 5.26 Connection Pooling and Management
```
Implement robust connection pooling:
- Connection pool with size limits
- Connection validation before use
- Automatic reconnection on failure
- Exponential backoff for retries
- Connection lifecycle management
- Idle connection timeout
- Monitoring (active, idle, failed connections)
- Thread-safe implementation
Test with network failures.
```

### 5.27 Resource Limits and Quotas
```
Enforce resource limits and quotas:
- Rate limiting (requests/second, orders/second)
- Memory limits (per component, total)
- CPU throttling for non-critical tasks
- Disk I/O limits
- Network bandwidth limits
- Graceful handling when limits exceeded
- Metrics for quota utilization
- Dynamic adjustment based on load
```

### 5.28 Crash Recovery and Persistence
```
Implement crash recovery mechanism:
- Checkpointing critical state
- Write-ahead log (WAL) for durability
- Replay log on recovery
- Snapshot + incremental log compaction
- Recovery time objective (RTO) < 1 minute
- Data consistency verification
- Test with kill -9 and verify recovery
- Automated recovery testing in CI
```

### 5.29 Logging for Production
```
Implement production-grade logging:
- Structured logging (JSON format)
- Log levels (ERROR, WARN, INFO, DEBUG, TRACE)
- Conditional compilation for hot path
- Async logging (separate thread)
- Log rotation and retention
- Correlation IDs for request tracing
- No PII/sensitive data in logs
- Integration with centralized logging (ELK, Splunk)
Measure logging overhead (<1% CPU).
```

### 5.30 Metrics and Observability
```
Implement comprehensive metrics collection:
- RED metrics (Rate, Errors, Duration)
- USE metrics (Utilization, Saturation, Errors)
- Latency histograms (P50, P95, P99, P99.9)
- Counter for events (orders, fills, errors)
- Gauge for state (queue depth, connections)
- Prometheus exposition format
- Minimal overhead (<0.1% CPU)
- Dashboard with Grafana
```

## Category: CI/CD and Deployment (10 prompts)

### 5.31 Hermetic Build System
```
Setup hermetic build with Bazel or CMake:
- Reproducible builds (fixed dependencies)
- Docker-based build environment
- Dependency management (Conan, vcpkg)
- Build caching (local and remote)
- Parallel builds with -j
- Static analysis in build (clang-tidy)
- Unit tests in build pipeline
- Artifact versioning (git SHA)
Build time target: <5 minutes for incremental.
```

### 5.32 Canary Deployment Strategy
```
Implement canary deployment with automated rollback:
- Deploy to 10% of production traffic
- Monitor key metrics (latency, error rate, PnL)
- Automatic rollback if SLO violated
- Gradual traffic shift (10% → 50% → 100%)
- Feature flags for A/B testing
- Smoke tests post-deployment
- Rollback procedure (one-click)
- Dashboard for canary health
Document rollback criteria and procedure.
```

### 5.33 Blue-Green Deployment
```
Setup blue-green deployment for zero-downtime:
- Maintain two identical environments (blue, green)
- Deploy to inactive environment
- Run acceptance tests
- Switch traffic with load balancer
- Keep old version for instant rollback
- Database migration strategy
- Smoke tests on new version
- Monitoring during cutover
Test rollback procedure regularly.
```

### 5.34 Performance Regression Detection
```
Automate performance regression detection in CI:
- Run benchmarks on every PR
- Compare with baseline (main branch)
- Statistical significance testing
- Fail CI if regression > 5%
- Flamegraph comparison
- Dashboard for performance trends
- Git bisect for regression root cause
- Automated alerts on regression
```

### 5.35 Load Testing in CI
```
Implement load testing in CI pipeline:
- Synthetic market data generator
- Realistic workload (peak volume + 50%)
- Measure latency and throughput
- Detect memory leaks (Valgrind)
- CPU profiling under load
- Pass/fail criteria (SLO compliance)
- Automated report generation
- Integration with performance dashboard
```

### 5.36 Chaos Engineering
```
Implement chaos engineering for resilience:
- Inject network latency and packet loss
- Simulate dependency failures
- Kill random processes
- Fill disk space
- Exhaust memory
- Verify system recovers gracefully
- Automated chaos experiments
- Metrics during chaos (availability, recovery time)
Run in staging before production.
```

### 5.37 Rollback Automation
```
Automate rollback procedure:
- Detect anomalies (error rate spike, latency increase)
- One-click rollback from dashboard
- Automatic rollback based on SLO
- Git revert and redeploy
- Database schema rollback
- Notification to team (Slack, PagerDuty)
- Post-mortem template generation
- Test rollback in staging regularly
```

### 5.38 Configuration Management
```
Implement configuration management system:
- Centralized configuration (etcd, Consul)
- Environment-specific configs (dev, staging, prod)
- Dynamic configuration updates (no restart)
- Configuration validation on startup
- Audit log for config changes
- Rollback to previous config
- Feature flags for gradual rollout
- Documentation for all config parameters
```

### 5.39 Container Orchestration
```
Deploy with Kubernetes or Docker Swarm:
- Containerize with multi-stage Dockerfile
- Resource limits (CPU, memory)
- Health checks (liveness, readiness)
- Horizontal pod autoscaling
- Rolling updates with zero downtime
- Service mesh for observability (Istio)
- Persistent volumes for state
- Helm charts for deployment
```

### 5.40 Incident Response Automation
```
Automate incident response workflow:
- Automated alerting (PagerDuty, Opsgenie)
- Runbooks for common incidents
- Automated diagnostics (logs, metrics, traces)
- Correlation of alerts
- Auto-remediation for known issues
- Incident timeline generation
- Post-mortem template
- Blameless culture documentation
Measure MTTR (Mean Time To Recovery).
```

## Category: Testing and Validation (10 prompts)

### 5.41 Deterministic Testing Framework
```
Build deterministic testing framework:
- Inject dependencies (clock, random, network)
- Replay market data deterministically
- Freeze time for testing
- Seed randomness for reproducibility
- Verify bit-exact results across runs
- Property-based testing (Hypothesis, QuickCheck)
- Fuzz testing for edge cases
- Integration with CI (every commit)
```

### 5.42 Market Data Simulator
```
Implement realistic market data simulator:
- Replay historical data with timing
- Synthetic data generation (random walk, GBM)
- Order book simulation
- Trade and quote generation
- Configurable market conditions (volatile, calm)
- Gap injection for testing recovery
- Performance testing (max throughput)
- Validation against real data statistics
```

### 5.43 Load and Stress Testing
```
Perform load and stress testing:
- Gradually increase load to find breaking point
- Measure latency at different loads
- Identify resource bottlenecks (CPU, memory, network)
- Sustained load test (24+ hours)
- Peak load test (2× expected volume)
- Spike test (sudden load increase)
- Monitoring during test (metrics, logs)
- Report with recommendations
```

### 5.44 Fault Injection Testing
```
Implement fault injection for resilience testing:
- Network failures (partition, latency, packet loss)
- Process crashes (kill -9)
- Disk failures (read-only, full disk)
- Memory exhaustion (OOM)
- Clock skew and NTP failures
- Dependency failures (database, external API)
- Verify system recovers gracefully
- Measure recovery time
```

### 5.45 Concurrency Testing
```
Test concurrent execution for race conditions:
- ThreadSanitizer (TSan) for data races
- AddressSanitizer (ASan) for memory errors
- Stress testing with multiple threads
- Property-based testing for concurrent code
- Deadlock detection
- Liveness testing (no hangs)
- Performance under contention
- Reproduce and fix race conditions
```

### 5.46 End-to-End Integration Testing
```
Implement end-to-end integration tests:
- Test full pipeline (market data → order → fill)
- Mock external dependencies
- Test with realistic data
- Verify P&L calculations
- Test error scenarios (reject, partial fill)
- Automated test suite in CI
- Test coverage >80%
- Performance benchmarking
```

### 5.47 Latency Validation Testing
```
Validate latency meets SLO:
- Inject timestamped messages
- Measure end-to-end latency
- Capture latency distribution (histogram)
- Verify P99 < threshold
- Test under load
- Identify outliers and root cause
- Continuous monitoring in production
- Regression testing in CI
```

### 5.48 Memory Leak Detection
```
Detect and fix memory leaks:
- Valgrind Memcheck for leak detection
- AddressSanitizer (ASan) for use-after-free
- Heap profiling (massif, heaptrack)
- Continuous monitoring in staging
- Long-running test (24+ hours)
- Memory growth analysis
- Fix leaks before production
- Automated leak detection in CI
```

### 5.49 Correctness Validation
```
Validate correctness of trading logic:
- Unit tests for all functions
- Property-based testing (invariants)
- Reference implementation comparison
- Numerical accuracy checks (floating point)
- Edge case testing (zero, negative, overflow)
- Boundary value analysis
- Test with historical data
- Cross-validation with external systems
```

### 5.50 Performance Benchmark Suite
```
Create comprehensive benchmark suite:
- Microbenchmarks for critical functions
- End-to-end latency benchmarks
- Throughput benchmarks
- Memory usage benchmarks
- Comparison with competitors (if data available)
- Automated execution in CI
- Historical tracking (performance over time)
- Regression alerts
Report: P50/P95/P99 latency, throughput, memory.
```

---

# 6. Senior Quantitative Researcher (50 Prompts)

## Category: Alpha Research Pipeline (10 prompts)

### 6.1 Hypothesis-Driven Research
```
Conduct hypothesis-driven alpha research:
- Formulate testable hypothesis (economic intuition, literature)
- Define expected behavior (e.g., "momentum persists 3-12 months")
- Data requirements and sources
- Statistical tests to validate hypothesis
- Success criteria (IC, Sharpe, returns)
- Document research in notebook
- Peer review process
- Decision: proceed to backtest or abandon
Include research log template.
```

### 6.2 Factor Mining and Discovery
```
Discover new alpha factors:
- Literature review (academic papers, Factor Zoo)
- Brainstorm factor ideas (fundamental, technical, alternative)
- Calculate factors on universe
- Screen for predictive power (IC, quintile returns)
- Test for redundancy with existing factors
- Factor decay analysis
- Turnover and capacity estimation
- Document factor construction methodology
Report top 10 candidate factors.
```

### 6.3 Feature Engineering for Quant
```
Engineer features for quantitative trading:
- Price-based: returns, volatility, momentum, reversals
- Volume: VWAP, order imbalance, volume shocks
- Fundamental: P/E, P/B, ROE, earnings growth
- Alternative data: sentiment, satellite, web traffic
- Time-series: lags, rolling statistics, seasonality
- Cross-sectional: rankings, z-scores
- Feature interactions and polynomials
- Dimensionality reduction (PCA, autoencoders)
Create feature library with 100+ features.
```

### 6.4 Information Coefficient Analysis
```
Calculate and analyze Information Coefficient (IC):
- Rank IC (Spearman correlation)
- Pearson IC
- Time-series of IC (stability)
- IC decay over holding periods
- IC by sector, market cap, region
- Statistical significance (t-test)
- IC combining multiple factors
- Optimal IC for portfolio construction
Visualize IC over time and provide insights.
```

### 6.5 Quintile Analysis
```
Perform quintile analysis for factor evaluation:
- Sort universe into 5 quintiles by factor
- Calculate average returns per quintile
- Long-short spread (Q5 - Q1)
- Turnover per quintile
- Monotonicity test
- Risk-adjusted returns (Sharpe per quintile)
- Transaction costs impact
- Comparison across time periods
Visualize cumulative returns by quintile.
```

### 6.6 Factor Orthogonalization
```
Orthogonalize factors to isolate unique alpha:
- Correlation matrix of factors
- Gram-Schmidt orthogonalization
- Regression residuals approach
- PCA for uncorrelated factors
- Measure redundancy reduction
- Compare IC before/after orthogonalization
- Impact on portfolio construction
- Document factor relationships
Provide correlation heatmap before/after.
```

### 6.7 Alpha Combination Strategies
```
Combine multiple alpha factors optimally:
- Equal-weight combination
- IC-weighted combination
- Volatility-weighted combination
- Optimization-based weights (maximize IC)
- Machine learning meta-model
- Time-varying weights
- Robust combination (handling outliers)
- Backtesting combined alpha
Compare performance of combination methods.
```

### 6.8 Alternative Data Integration
```
Integrate alternative data into alpha research:
- Data source evaluation (quality, coverage, cost)
- Data preprocessing and cleaning
- Feature engineering from alt data
- Predictive power analysis
- Combination with traditional signals
- Capacity and scalability assessment
- Regulatory compliance (data usage rights)
- Cost-benefit analysis
Example: satellite imagery, credit card data, web scraping.
```

### 6.9 Sector and Industry Neutrality
```
Implement sector and industry neutrality:
- Sector/industry classification (GICS, ICB)
- Calculate factor exposures by sector
- Demean factors within sectors
- Industry-neutral portfolio construction
- Test for residual sector tilts
- Compare sector-neutral vs unconstrained
- Impact on diversification and risk
- Sector rotation strategy
Verify beta to sector indices ≈ 0.
```

### 6.10 Capacity and Scalability Analysis
```
Estimate strategy capacity and scalability:
- Average daily volume (ADV) analysis
- Market impact modeling
- Liquidity-adjusted capacity
- Turnover and portfolio size constraints
- Marginal impact of increasing capital
- Optimal portfolio size
- Decay curve (capacity vs Sharpe)
- Documentation for portfolio management
Provide capacity estimate in $ AUM.
```

## Category: Backtesting and Validation (10 prompts)

### 6.11 Walk-Forward Validation Framework
```
Implement walk-forward validation:
- Rolling window: train on N months, test on M months
- Anchored window vs rolling window
- Parameter stability across windows
- Out-of-sample performance tracking
- Degradation analysis (IS vs OOS)
- Statistical tests (IS vs OOS returns)
- Optimal train/test ratio
- Retraining frequency decision
Report OOS Sharpe and compare to IS.
```

### 6.12 Bias-Free Backtesting
```
Ensure backtest is free from biases:
- Survivorship bias: include delisted stocks
- Look-ahead bias: point-in-time data only
- Selection bias: no cherry-picking symbols/periods
- Data snooping: limit strategy iterations
- Overfitting: cross-validation, regularization
- Backtest audit checklist
- Document all trials (success and failure)
- Independent validation by another researcher
Provide bias audit report.
```

### 6.13 Transaction Cost Modeling
```
Model realistic transaction costs:
- Bid-ask spread (half-spread)
- Market impact (sqrt or linear in volume)
- Commission structure
- Slippage (price movement during execution)
- Opportunity cost (missed trades)
- Borrow costs for shorts
- Calibrate with real execution data
- Sensitivity analysis (cost ±50%)
Compare strategy with and without costs.
```

### 6.14 Sharpe Ratio Estimation and Significance
```
Estimate Sharpe ratio with statistical rigor:
- Calculate annualized Sharpe
- Standard error of Sharpe estimate
- Confidence interval (bootstrap or analytical)
- T-statistic and p-value
- Sharpe ratio stability over time
- Comparison with benchmark
- Haircut for multiple testing (Bonferroni)
- Minimum track record length for significance
Report: Sharpe, std error, 95% CI, p-value.
```

### 6.15 Maximum Drawdown and Recovery
```
Analyze maximum drawdown and recovery:
- Calculate max drawdown (peak-to-trough)
- Drawdown duration and recovery time
- Drawdown frequency distribution
- Conditional drawdown (CVaR of drawdowns)
- Expected recovery time
- Comparison with benchmark
- Stress testing under scenarios
- Investor psychology (risk of abandonment)
Visualize underwater curve and distribution.
```

### 6.16 Factor Exposure Analysis
```
Analyze factor exposures (Fama-French, Barra):
- Regression on factor returns
- Time-series of factor betas
- Decompose returns into factor contributions
- Alpha after factor adjustment
- Hedging factor exposures
- Factor timing vs constant exposure
- Risk from unintended factor bets
- Factor-neutral portfolio construction
Report: alpha, factor betas, R².
```

### 6.17 Turnover and Capacity
```
Analyze turnover and capacity constraints:
- Calculate portfolio turnover (%)
- Turnover decomposition (factor decay, rebalancing)
- Relationship between turnover and returns
- Transaction cost as % of returns
- Capacity estimate based on ADV
- Optimal rebalancing frequency
- Trade-off: turnover vs performance
- Turnover reduction techniques
Recommend turnover target and capacity.
```

### 6.18 Monte Carlo Simulation
```
Perform Monte Carlo simulation for strategy:
- Bootstrap historical returns
- Simulate 10,000 paths
- Distribution of Sharpe, max DD, returns
- Probability of negative returns
- Risk of ruin estimation
- Confidence intervals for all metrics
- Stress scenarios (bear market, volatility spike)
- Comparison of strategies under uncertainty
Visualize distribution of outcomes.
```

### 6.19 Regime Analysis
```
Analyze strategy performance across market regimes:
- Define regimes (bull/bear, high/low vol, rising/falling rates)
- Identify regime switches (HMM, threshold-based)
- Performance metrics per regime
- Factor behavior in regimes
- Regime-dependent position sizing
- Regime forecasting
- Diversification across regimes
- Portfolio construction with regime awareness
Report performance by regime.
```

### 6.20 Alpha Decay Measurement
```
Measure alpha decay over holding periods:
- Calculate returns at horizons (1-day, 1-week, 1-month)
- IC decay curve
- Optimal holding period
- Rebalancing frequency decision
- Turnover vs decay trade-off
- Comparison with industry benchmarks
- Factor-specific decay patterns
- Documentation for implementation
Visualize IC vs holding period.
```

## Category: Statistical Methods (10 prompts)

### 6.21 Cointegration Testing
```
Test for cointegration in pairs trading:
- Augmented Dickey-Fuller (ADF) test
- Johansen cointegration test
- Engle-Granger two-step method
- Calculate cointegration vector (hedge ratio)
- Half-life of mean reversion
- Rolling cointegration testing
- Cointegration breakdown detection
- Portfolio of cointegrated pairs
Select pairs with p-value < 0.05 and stable hedge ratio.
```

### 6.22 Time Series Modeling
```
Implement time series models for forecasting:
- ARIMA for univariate series
- VAR for multivariate series
- GARCH for volatility modeling
- State-space models (Kalman filter)
- Structural breaks detection
- Model diagnostics (residuals, ACF)
- Walk-forward evaluation
- Comparison with machine learning
Forecast returns and volatility.
```

### 6.23 Hypothesis Testing for Strategies
```
Perform rigorous hypothesis testing:
- Null hypothesis: strategy has no alpha
- T-test for mean returns
- Permutation test (shuffle returns)
- Multiple testing correction (Bonferroni, FDR)
- Power analysis (sample size calculation)
- Effect size (Cohen's d)
- P-hacking awareness
- Publication bias
Report: test statistic, p-value, conclusion.
```

### 6.24 Cross-Sectional Regression
```
Perform cross-sectional regression analysis:
- Fama-MacBeth regression
- Panel regression with fixed effects
- Factor exposures and risk premia
- Time-series of cross-sectional betas
- Heteroskedasticity and autocorrelation (HAC)
- Newey-West standard errors
- Interpretation of coefficients
- Economic vs statistical significance
Estimate factor risk premia.
```

### 6.25 Principal Component Analysis (PCA)
```
Apply PCA for dimensionality reduction:
- Correlation or covariance matrix
- Eigenvalue decomposition
- Scree plot for component selection
- Explained variance per component
- Factor loadings interpretation
- Reconstruct data from components
- Use in portfolio construction
- Comparison with factor models
Reduce 100 factors to top 10 components.
```

### 6.26 Outlier Detection and Treatment
```
Detect and handle outliers in financial data:
- Statistical methods (Z-score, IQR)
- MAD (Median Absolute Deviation)
- Winsorization vs trimming
- Robust statistics (median, MAD)
- Impact on backtest (with/without outliers)
- Event study for extreme events
- Outlier analysis by sector/time
- Documentation of treatment
Test sensitivity to outlier handling.
```

### 6.27 Correlation and Causation
```
Distinguish correlation from causation:
- Granger causality test
- Instantaneous vs lagged correlation
- Lead-lag relationships
- Spurious correlations
- Confounding variables
- Instrumental variables approach
- Natural experiments
- Economic intuition validation
Document causal mechanisms.
```

### 6.28 Stationarity Testing
```
Test for stationarity in time series:
- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Phillips-Perron test
- Differencing to achieve stationarity
- Seasonal adjustment
- Implications for modeling
- Rolling stationarity tests
- Cointegration as alternative
Transform non-stationary series appropriately.
```

### 6.29 Autocorrelation Analysis
```
Analyze autocorrelation in returns:
- ACF and PACF plots
- Ljung-Box test for autocorrelation
- Identify momentum vs mean reversion
- Optimal lookback for strategies
- Seasonality detection
- Autocorrelation in residuals (model check)
- Trading opportunities from autocorrelation
- Comparison across assets
Document patterns and trading rules.
```

### 6.30 Volatility Forecasting
```
Forecast volatility for risk management:
- Historical volatility (rolling window)
- EWMA (Exponentially Weighted Moving Average)
- GARCH family models (GARCH, EGARCH, GJR-GARCH)
- Implied volatility from options
- Realized volatility from intraday data
- Volatility clustering
- Forecast evaluation (MSE, QLIKE)
- Application in position sizing
Provide volatility forecasts and uncertainty.
```

## Category: Machine Learning for Trading (10 prompts)

### 6.31 Feature Selection for ML
```
Perform feature selection for trading ML models:
- Filter methods (correlation, mutual information)
- Wrapper methods (RFE, forward/backward selection)
- Embedded methods (Lasso, tree importance)
- Eliminate multicollinearity (VIF)
- Stability of feature selection
- Cross-validation for selection
- Trade-off: more features vs overfitting
- Documentation of selected features
Reduce from 200 to top 50 features.
```

### 6.32 Time Series Cross-Validation
```
Implement proper CV for time series:
- TimeSeriesSplit (rolling window)
- Purged K-fold (remove leakage)
- Embargoed CV (gap between train and test)
- Walk-forward validation
- Combinatorial purged CV
- Avoid look-ahead bias
- Stratification by volatility regime
- Report performance across folds
Ensure temporal ordering preserved.
```

### 6.33 XGBoost for Return Prediction
```
Train XGBoost for return prediction:
- Feature engineering (technical, fundamental)
- Label: forward returns (classification or regression)
- Hyperparameter tuning (max_depth, learning_rate, n_estimators)
- Early stopping with validation set
- Feature importance analysis
- SHAP values for interpretability
- Walk-forward evaluation
- Comparison with linear models
Report: IC, Sharpe, feature importance.
```

### 6.34 LSTM for Sequence Prediction
```
Implement LSTM for time series forecasting:
- Sequence construction (lookback window)
- Feature normalization
- LSTM architecture (layers, units, dropout)
- Attention mechanism
- Bidirectional LSTM
- Multi-step ahead forecasting
- Walk-forward evaluation
- Comparison with ARIMA and XGBoost
Forecast returns or volatility.
```

### 6.35 Ensemble Methods for Robustness
```
Create ensemble of trading models:
- Diverse base models (linear, tree, NN)
- Stacking with meta-learner
- Voting (average, weighted)
- Boosting and bagging
- Out-of-fold predictions
- Diversity metrics
- Performance improvement
- Robustness across regimes
Combine 5+ models and measure improvement.
```

### 6.36 Hyperparameter Optimization
```
Optimize ML hyperparameters for trading:
- Grid search with time-series CV
- Random search
- Bayesian optimization (Optuna)
- Hyperband for early stopping
- Walk-forward evaluation of tuned model
- Avoid overfitting to validation set
- Ensemble over hyperparameters
- Document optimal settings
Report: best params, CV score, test score.
```

### 6.37 Model Interpretability (SHAP)
```
Interpret ML model predictions with SHAP:
- Global feature importance
- Local explanations for predictions
- Force plots for individual trades
- Dependence plots (feature vs SHAP)
- Interaction effects
- Validate against domain knowledge
- Communication with stakeholders
- Debugging model errors
Provide top features and their impact.
```

### 6.38 Overfitting Detection and Prevention
```
Detect and prevent overfitting in ML models:
- Train vs validation vs test performance
- Learning curves (performance vs data size)
- Complexity metrics (number of parameters)
- Regularization (L1, L2, dropout)
- Early stopping
- Ensemble for variance reduction
- Cross-validation for robust evaluation
- Simplicity bias (Occam's razor)
Report: train/val/test Sharpe, overfitting score.
```

### 6.39 Online Learning and Adaptation
```
Implement online learning for market adaptation:
- Incremental model updates
- Sliding window retraining
- Concept drift detection
- Adaptive learning rate
- Forgetting old data (exponential weighting)
- A/B testing of models
- Trigger for retraining
- Performance tracking over time
Measure improvement with online learning.
```

### 6.40 Label Engineering for Classification
```
Engineer labels for classification ML:
- Binary: positive/negative returns
- Multi-class: strong buy, buy, hold, sell, strong sell
- Quantile-based labels
- Volatility-adjusted returns
- Triple barrier method (profit, stop, time)
- Label balance and class weights
- Purging and embargo for labels
- Impact on model performance
Compare labeling strategies.
```

## Category: Research Documentation (10 prompts)

### 6.41 Research Notebook Template
```
Create standardized research notebook template:
- Executive summary
- Hypothesis and motivation
- Data description and preprocessing
- Methodology (features, model, backtesting)
- Results (metrics, visualizations)
- Sensitivity analysis
- Limitations and risks
- Recommendations (proceed, refine, abandon)
- Appendix (code, tables)
Include version control and peer review.
```

### 6.42 Reproducible Research
```
Ensure research is fully reproducible:
- Random seed setting
- Environment specification (requirements.txt, conda env)
- Data versioning (DVC)
- Code versioning (git)
- Jupyter notebook with clear narrative
- Automated pipeline (Airflow, Prefect)
- Unit tests for critical functions
- Documentation of all parameters
Provide one-click reproduction script.
```

### 6.43 Research Log and Tracking
```
Maintain comprehensive research log:
- Date, researcher, hypothesis
- Data used, time period
- Features and model
- Backtest results (metrics)
- Outcome (success, failure, inconclusive)
- Lessons learned
- Next steps
- Link to notebooks and code
Track all experiments to avoid data snooping.
```

### 6.44 Factor Library Documentation
```
Document factor library comprehensively:
- Factor name and formula
- Data sources and frequency
- Rationale and literature references
- Historical performance (IC, returns)
- Correlation with other factors
- Turnover and capacity
- Implementation complexity
- Usage guidelines
Maintain up-to-date factor catalog.
```

### 6.45 Peer Review Process
```
Establish peer review process for research:
- Review checklist (bias-free, reproducible, documented)
- Statistical rigor verification
- Code review for bugs
- Independent replication
- Constructive feedback
- Iterate based on review
- Approval gate before production
- Document review comments and responses
Require 2+ reviewers for production strategies.
```

### 6.46 Research Presentation
```
Prepare research presentation for stakeholders:
- Executive summary (1 slide)
- Hypothesis and motivation
- Methodology overview
- Key results (Sharpe, IC, drawdown)
- Risk analysis and limitations
- Capacity and scalability
- Implementation plan
- Q&A preparation
Tailor to audience (PM, tech, risk).
```

### 6.47 Sensitivity Analysis Documentation
```
Document sensitivity analysis comprehensively:
- Parameter ranges tested
- Impact on key metrics (Sharpe, turnover)
- Robustness to assumptions
- Break-even analysis
- Worst-case scenarios
- Recommendations for parameter settings
- Monitoring plan for production
- Update frequency
Visualize with tornado diagrams.
```

### 6.48 Handover to Production
```
Prepare research for production handover:
- Production-ready code (refactored, tested)
- Configuration file (parameters, thresholds)
- Data requirements and sources
- Deployment instructions
- Monitoring metrics and thresholds
- Alerting rules
- Runbook for common issues
- Training for operations team
Include transition checklist.
```

### 6.49 Research Archive and Retrieval
```
Archive research for future reference:
- Centralized repository (Confluence, Notion)
- Searchable by keyword, factor, method
- Versioned notebooks and code
- Backtest results storage
- Lessons learned database
- Failed experiments (what not to do)
- Regular cleanup and organization
- Access control and permissions
Enable knowledge sharing across team.
```

### 6.50 Performance Reporting
```
Create comprehensive performance report:
- Monthly/quarterly summary
- P&L attribution (alpha, factors, costs)
- Risk metrics (Sharpe, max DD, VaR)
- Comparison with benchmark
- Factor exposures
- Trade analysis (winners, losers)
- Capacity utilization
- Recommendations for adjustments
Distribute to PM, risk, and executive team.
```

---

## 7. Senior Systematic Trader Prompts (50)

### Category: Live Execution Operations

### 7.1 Pre-Market Checklist
```
Execute comprehensive pre-market operations checklist:
- Verify all market data feeds (primary/backup) connectivity and latency
- Test order routing to all venues (send test orders to QA endpoints)
- Review overnight position reconciliation against broker reports
- Check risk limits (position, exposure, VaR) loaded correctly
- Verify strategy parameters match approved values
- Test emergency kill switches and manual override
- Review market calendar (holidays, early closes, corporate actions)
- Check funding rates, borrow costs updated
- Monitor news for overnight events affecting positions
Document any anomalies in operations log.
```

### 7.2 Live Strategy Monitoring Dashboard
```
Build comprehensive real-time monitoring dashboard:
- Current P&L (realized/unrealized) with 1-second refresh
- Position vs target delta by strategy and aggregate
- Fill rate, rejection rate, latency by venue
- Strategy state (running/paused/error) with indicators
- Risk utilization (% of VaR, position limits)
- Market regime indicators (volatility, spread, volume)
- Alerting panel (critical/warning/info)
- Historical performance charts (intraday, trailing week)
Include drill-down capability to individual symbols.
```

### 7.3 Order Execution Quality Analysis
```
Analyze execution quality for today's trading:
- Implementation shortfall (arrival price vs execution)
- VWAP slippage per strategy
- Fill rate breakdown (complete/partial/rejected)
- Venue performance comparison (fill rate, slippage)
- Time-to-fill distribution
- Adverse selection analysis
- Comparison to TCA benchmarks
- Identify patterns in rejections/failures
Generate daily report with actionable recommendations.
```

### 7.4 Intraday Risk Management
```
Implement real-time risk monitoring and controls:
- Calculate live VaR with 5-minute refresh
- Track drawdown from day high (P&L and Sharpe)
- Monitor concentration risk (top N positions)
- Greeks exposure for options portfolios
- Correlation breakdown vs historical
- Stress scenarios (±2σ moves in key factors)
- Automatic position reduction if limits breached
- Escalation protocol (trader → PM → CRO)
Document all risk interventions in audit log.
```

### 7.5 Market Impact Minimization
```
Optimize order execution to minimize market impact:
- Measure historical price impact coefficients
- Implement adaptive order slicing (volume participation)
- Use dark pools and hidden liquidity for large orders
- Schedule trades around volume peaks
- Avoid trading around news/events
- Implement TWAP/VWAP for large unwinds
- Monitor price reversion after trades
- Compare impact vs theoretical models (Almgren-Chriss)
Target <10 bps impact for typical trade sizes.
```

### 7.6 Emergency Kill Switch Protocol
```
Design and test emergency shutdown procedure:
- One-click cancel all open orders across all venues
- Automatic position flattening (market orders with limits)
- Notification to PM, risk, compliance within 30 seconds
- Lock out new order submissions
- Capture system state (logs, positions, orders) for forensics
- Test quarterly in controlled environment
- Document activation criteria (runaway algo, fat finger, system error)
- Recovery procedure and rollback plan
Ensure <5 second execution time from trigger.
```

### 7.7 Venue Performance Comparison
```
Evaluate execution venue performance:
- Fill rate by venue and order type
- Effective spread comparison
- Latency measurement (order ack, fills)
- Reject rate and reasons
- Adverse selection by venue
- Rebate capture effectiveness
- Hidden liquidity availability
- Correlation of venue outages with P&L
Recommend venue routing optimization changes.
```

### 7.8 Real-Time P&L Attribution
```
Build real-time P&L attribution system:
- Break down P&L into alpha, execution costs, funding
- Separate strategy contributions (gross/net)
- Market impact vs alpha capture
- Fee/commission breakdown by venue
- Borrow costs and funding charges
- Currency effects for multi-currency portfolios
- Unexplained P&L tracking and investigation
- Compare intraday vs EOD official P&L
Refresh every 60 seconds with <10 second staleness.
```

### 7.9 Liquidity Crisis Management
```
Handle liquidity crisis in live trading:
- Detect liquidity drought (spread widening, volume drop)
- Pause aggressive strategies automatically
- Switch to passive order types (limit vs market)
- Extend time horizon for position targets
- Reduce position sizing temporarily
- Monitor correlation breakdown (diversification failure)
- Communicate with market makers for liquidity
- Document incident for post-mortem analysis
Define clear thresholds for automatic intervention.
```

### 7.10 End-of-Day Reconciliation
```
Execute comprehensive EOD reconciliation:
- Match all fills against venue reports
- Reconcile positions with prime broker/custodian
- Verify P&L calculation vs official accounting
- Check for missing fills or duplicate executions
- Resolve breaks within 2 hours of close
- Generate discrepancy report with root causes
- Update audit trail and compliance records
- Prepare handover notes for next trading day
Escalate unresolved breaks to operations manager.
```

### Category: TCA and Execution Optimization

### 7.11 Implementation Shortfall Analysis
```
Measure and minimize implementation shortfall:
- Calculate IS = (execution price - decision price) × shares
- Decompose into delay cost, market impact, timing cost, opportunity cost
- Compare vs theoretical optimal (Almgren-Chriss)
- Analyze by order urgency, size, time of day
- Identify strategies with highest IS
- Test alternative execution tactics (aggressive vs passive)
- Benchmark against peer TCA data
- Optimize target participation rate
Report daily IS statistics to PM.
```

### 7.12 VWAP Performance Tracking
```
Track VWAP execution performance:
- Calculate VWAP slippage for all parent orders
- Decompose by strategy, symbol, order size
- Identify systematic deviations (always early/late vs VWAP)
- Compare scheduled vs opportunistic VWAP
- Analyze volume curve prediction accuracy
- Test improvements to volume forecasting
- Benchmark vs industry VWAP algos
- Adjust participation rates to minimize slippage
Target <5 bps median VWAP slippage.
```

### 7.13 Smart Order Routing Optimization
```
Optimize smart order routing logic:
- Collect historical fill data by venue, symbol, time
- Build venue scoring model (fill rate, spread, latency, rebates)
- Implement dynamic routing based on real-time conditions
- Test A/B routing strategies in simulation
- Monitor routing performance daily
- Incorporate hidden liquidity signals
- Adjust for venue correlations (avoid adverse selection)
- Implement failover for venue outages
Measure 10-20% improvement in effective spreads.
```

### 7.14 Adverse Selection Detection
```
Detect and mitigate adverse selection:
- Measure post-trade price drift (30s, 1min, 5min)
- Identify patterns (time of day, symbol, venue)
- Correlate with order characteristics (size, urgency)
- Detect toxic flow indicators (order book imbalance)
- Adjust limit order pricing to reduce selection
- Avoid stale quotes in fast markets
- Use mid-point pegs where appropriate
- Monitor maker/taker ratio impact
Target neutral or positive post-trade drift.
```

### 7.15 Fill Rate Optimization
```
Improve order fill rates:
- Analyze partial fills and cancellations
- Identify time-to-fill patterns by venue
- Test aggressive vs passive limit order placement
- Optimize limit price (spread capture vs fill probability)
- Implement adaptive timeout logic
- Use IOC vs GTC based on urgency
- Measure fill rate vs market conditions (volatility, spread)
- A/B test different order types
Achieve >90% fill rate for typical orders.
```

### 7.16 Latency Arbitrage Prevention
```
Protect against latency arbitrage:
- Measure time-to-quote (exchange → strategy)
- Detect stale quotes in fast-moving markets
- Implement quote aging and automatic cancellation
- Use faster data feeds (direct exchange connections)
- Co-locate near matching engines where feasible
- Monitor for consistent pick-offs on limit orders
- Adjust quoting strategy in high-frequency regimes
- Calculate cost of latency arbitrage
Reduce pick-offs by 50% through faster cancels.
```

### 7.17 Spread Capture Analysis
```
Analyze bid-ask spread capture:
- Measure percentage of spread captured per trade
- Compare limit vs market order spread capture
- Track by symbol, time of day, market regime
- Identify opportunities for passive liquidity provision
- Test maker-taker economics by venue
- Optimize limit order placement (at/near touch)
- Monitor queue position and adverse selection
- Calculate net spread capture after fees/rebates
Target 40-60% spread capture on passive orders.
```

### 7.18 Execution Cost Breakdown
```
Decompose total execution costs:
- Explicit costs: commissions, fees, taxes
- Implicit costs: spread, impact, delay, opportunity
- Calculate in bps of notional
- Break down by strategy, symbol, venue
- Compare vs budget/expectations
- Identify high-cost outliers
- Test cost reduction tactics
- Benchmark vs industry standards
Report total cost per strategy monthly.
```

### 7.19 Dark Pool Performance
```
Evaluate dark pool execution quality:
- Fill rate comparison across dark venues
- Price improvement vs lit markets
- Information leakage assessment
- Size of fills (full vs partial)
- Latency measurement
- Adverse selection post-dark fill
- Cost-benefit vs exchange rebates
- Optimal dark pool routing strategy
Include top 5 dark pools in evaluation.
```

### 7.20 Execution Algorithm Selection
```
Choose optimal execution algorithm:
- Map order characteristics to algo type (VWAP, TWAP, POV, IS)
- Test algo performance in backtesting
- Measure real-world algo vs benchmark
- Identify when to use algo vs direct trading
- Optimize algo parameters (urgency, participation)
- Monitor algo provider performance
- Detect algo gaming or front-running
- Build internal algos for sensitive orders
Document algo selection criteria.
```

### Category: Strategy Calibration

### 7.21 Daily Parameter Review
```
Review and adjust strategy parameters daily:
- Compare realized vs expected volatility, correlations
- Check if signal decay aligns with assumptions
- Verify position sizing vs risk budget
- Test parameter sensitivity (±10% changes)
- Identify drift in model coefficients
- Review stop-loss trigger levels
- Adjust for changing market microstructure
- Document all parameter changes with rationale
Require PM approval for material changes (>20%).
```

### 7.22 Regime Detection and Adaptation
```
Implement market regime detection:
- Classify markets (trending, mean-reverting, crisis, calm)
- Calculate regime probabilities using HMM or similar
- Adjust strategy parameters per regime
- Scale position sizing based on regime confidence
- Pause strategies in unfavorable regimes
- Monitor regime transitions (whipsaw risk)
- Backtest regime-adaptive strategy
- Compare vs static parameter approach
Target 15-25% Sharpe improvement through adaptation.
```

### 7.23 Signal Decay Analysis
```
Monitor signal decay in production:
- Measure signal autocorrelation over time
- Track hit rate evolution (daily, weekly, monthly)
- Compare recent vs historical performance
- Test if decay is temporary or permanent
- Identify causes (crowding, market change, data issues)
- Decide on signal refresh vs retirement
- A/B test signal modifications
- Document signal lifecycle
Report material decay (>20%) to research team.
```

### 7.24 Turnover Optimization
```
Optimize portfolio turnover:
- Measure current daily/monthly turnover
- Calculate relationship between turnover and costs
- Test impact of trading threshold changes
- Implement transaction cost-aware optimization
- Use buffer zones to reduce whipsaw
- Analyze turnover drivers (signals, rebalancing, risk)
- Compare turnover vs alpha generated
- Find optimal turnover-performance tradeoff
Target turnover reduction of 20-30% with <5% alpha loss.
```

### 7.25 Risk Target Calibration
```
Calibrate risk targets dynamically:
- Monitor realized volatility vs target
- Adjust position sizing to hit vol target
- Scale up in low-vol regimes, down in high-vol
- Incorporate correlation forecasts
- Test different target windows (trailing 20, 60, 120 days)
- Avoid over-leveraging in quiet markets
- Set maximum leverage bounds
- Track deviation from target (tracking error)
Maintain realized vol within 10% of target.
```

### 7.26 Factor Exposure Management
```
Manage unintended factor exposures:
- Calculate daily factor loadings (market, size, value, momentum)
- Set target ranges for each factor
- Implement factor neutralization if needed
- Monitor exposure drift over time
- Rebalance to target exposures
- Test impact on strategy alpha
- Report exposures to risk team
- Compare vs peer strategies
Keep unintended exposures <0.2 per factor.
```

### 7.27 Holding Period Optimization
```
Optimize trade holding periods:
- Analyze P&L by holding period (hours, days, weeks)
- Identify optimal holding window per strategy
- Test dynamic holding based on signal strength
- Calculate turnover-adjusted returns
- Measure alpha decay rate
- Adjust exit rules to match alpha persistence
- Compare fixed vs adaptive holding periods
- Monitor changes in optimal period over time
Document holding period policy per strategy.
```

### 7.28 Stop-Loss Calibration
```
Calibrate stop-loss levels:
- Backtest different stop-loss thresholds (-1%, -2%, -3%)
- Measure impact on Sharpe, max drawdown, turnover
- Test trailing stops vs fixed stops
- Analyze false stops (stopped then reversed)
- Calculate opportunity cost of stops
- Optimize stop width by volatility regime
- Test time-based stops (hold minimum period)
- Compare strategies with/without stops
Document optimal stop policy per strategy.
```

### 7.29 Leverage Adjustment
```
Dynamically adjust strategy leverage:
- Monitor Kelly criterion optimal leverage
- Scale based on recent Sharpe ratio
- Reduce leverage after drawdowns
- Increase leverage after strong performance (cautiously)
- Test leverage ramping strategies
- Set hard maximum leverage bounds
- Monitor correlation with other strategies
- Calculate VaR-based leverage limits
Document leverage policy and adjustment rules.
```

### 7.30 Performance Feedback Loop
```
Implement systematic performance feedback:
- Daily P&L review with root cause analysis
- Weekly strategy performance meeting
- Monthly calibration review
- Track changes made and impact measured
- Maintain decision log (what changed, why, result)
- Learn from both successes and failures
- Share insights across strategies
- Build institutional knowledge base
Create feedback loop with <1 week response time.
```

### Category: Production Governance

### 7.31 Change Control Process
```
Implement rigorous change control:
- Document all proposed changes (parameters, code, data)
- Require business justification and backtest evidence
- Code review by second developer
- QA testing in staging environment
- PM approval for strategy changes
- Risk approval for limit changes
- Staged rollout (paper trading → small size → full size)
- Monitoring period post-change (1-2 weeks)
Maintain audit trail of all changes.
```

### 7.32 Strategy Authorization Matrix
```
Define authorization levels for strategy operations:
- Parameter changes within bounds: Trader discretion
- Parameter changes outside bounds: PM approval
- Code changes: Developer + QA + PM approval
- Risk limit changes: PM + CRO approval
- New strategy launch: Investment committee approval
- Emergency shutdown: Trader discretion (with notification)
- Strategy retirement: PM approval
- Incident escalation: Follow escalation tree
Document in operations manual.
```

### 7.33 Incident Response Protocol
```
Define comprehensive incident response:
- Incident classification (P0: critical, P1: major, P2: minor)
- Response time SLAs (P0: immediate, P1: <15min, P2: <2hrs)
- Escalation chain (trader → PM → CRO → CIO)
- Immediate actions (pause, kill switch, reduce size)
- Communication protocol (internal + external)
- Incident documentation template
- Post-incident review (PIR) process
- Root cause analysis and remediation
Test quarterly with tabletop exercises.
```

### 7.34 Compliance Monitoring
```
Ensure continuous compliance monitoring:
- Pre-trade compliance checks (position limits, prohibited securities)
- Post-trade surveillance (wash trades, layering, spoofing patterns)
- Daily best execution review
- Market abuse detection
- Record all communications (Bloomberg, email, calls)
- Maintain trade audit trail
- Regular compliance attestations
- External audit readiness
Generate daily compliance report.
```

### 7.35 Disaster Recovery Testing
```
Test disaster recovery procedures:
- Simulate primary system failure
- Failover to backup systems within RTO (15 minutes)
- Verify data consistency post-failover
- Test trading from DR site
- Validate position reconciliation
- Check market data feed redundancy
- Test communication channels
- Document gaps and remediate
Conduct full DR test semi-annually.
```

### 7.36 Model Validation and Review
```
Periodic independent model validation:
- Annual review by model validation team
- Replicate backtest results independently
- Review assumptions and limitations
- Test sensitivity to parameter changes
- Assess model risk (overfitting, data snooping)
- Benchmark vs alternative models
- Review any model changes since last validation
- Produce validation report with recommendations
Address any concerns before production use.
```

### 7.37 Audit Trail Maintenance
```
Maintain comprehensive audit trail:
- Log all orders (submitted, modified, canceled, filled)
- Record all parameter changes with timestamp and user
- Capture system state at key events
- Store P&L calculations with inputs
- Archive market data used for decisions
- Maintain version control of code
- Log all manual interventions
- Ensure immutability and tamper-evidence
Retention: 7 years minimum for regulatory compliance.
```

### 7.38 Access Control and Segregation
```
Implement strict access controls:
- Role-based access (trader, developer, PM, risk, compliance)
- Separation of duties (no single person can deploy + approve)
- Multi-factor authentication for production systems
- Audit all privileged access
- Regular access reviews (quarterly)
- Immediate revocation on role change
- Segregate production/QA/dev environments
- Protect sensitive data (PII, positions, P&L)
Zero tolerance for unauthorized access.
```

### 7.39 Vendor and Data Provider SLAs
```
Monitor vendor SLA compliance:
- Market data uptime (target: 99.95%)
- Latency (P99 <10ms for critical feeds)
- Accuracy (zero tolerance for bad data)
- Exchange connectivity uptime
- Order routing system availability
- Cloud infrastructure SLAs
- Track incidents and credits
- Review quarterly with vendors
Escalate SLA breaches immediately.
```

### 7.40 Documentation Standards
```
Maintain comprehensive documentation:
- Strategy specifications (logic, parameters, risk limits)
- Runbooks for operations (startup, shutdown, troubleshooting)
- System architecture diagrams
- Data flow and dependencies
- Alerting rules and escalation
- Change history with rationale
- Known issues and workarounds
- Contact information (on-call, vendors)
Review and update documentation quarterly.
```

### Category: Monitoring and Reconciliation

### 7.41 Real-Time Alerting System
```
Build comprehensive alerting system:
- Critical: Strategy error, connectivity loss, limit breach (SMS + email)
- Warning: High latency, partial fills, unusual P&L (email)
- Info: Daily summary, performance milestones (dashboard)
- Alert fatigue prevention (smart grouping, suppression)
- Escalation for unacknowledged alerts (15 min)
- Alert testing and tuning (false positive <5%)
- Integration with monitoring dashboard
- On-call rotation schedule
Target <2 minute time-to-acknowledge for critical alerts.
```

### 7.42 Position Reconciliation
```
Reconcile positions continuously:
- Real-time vs broker/exchange every 5 minutes
- EOD full reconciliation with prime broker
- Investigate breaks >0.1% of position
- Resolve breaks within 2 hours
- Track reconciliation metrics (break rate, time to resolve)
- Root cause analysis for recurring breaks
- Automated reconciliation where possible
- Manual review for complex instruments
Report unresolved breaks to PM and risk immediately.
```

### 7.43 P&L Reconciliation
```
Reconcile P&L across systems:
- Intraday P&L vs risk system (hourly)
- EOD P&L vs official books (daily)
- Realized vs unrealized P&L breakdown
- Strategy-level vs aggregate P&L
- Investigate discrepancies >$10k or 1%
- Common causes: pricing differences, missed trades, corporate actions
- Track unexplained P&L over time
- Monthly reconciliation with finance
Resolve all material differences within 24 hours.
```

### 7.44 Market Data Quality Monitoring
```
Monitor market data quality continuously:
- Detect stale quotes (no update >5 seconds)
- Identify price outliers (>10σ moves)
- Monitor bid-ask spread anomalies
- Check for negative spreads or crossed quotes
- Validate volume consistency
- Compare primary vs backup feeds
- Alert on feed latency spikes
- Automatic failover to backup feed
Log all data quality issues for vendor review.
```

### 7.45 Order Flow Analysis
```
Analyze order flow patterns daily:
- Order submission rate by strategy
- Fill rate trends over time
- Rejection rate and reasons
- Average order size and count
- Venue distribution
- Order type mix (market, limit, IOC, etc.)
- Canceled vs filled orders
- Identify anomalies or changes in patterns
Detect issues before they impact P&L.
```

### 7.46 System Performance Monitoring
```
Monitor system performance metrics:
- CPU, memory, network utilization
- Order latency (submission to ack)
- Event processing latency
- Database query performance
- Message queue depths
- Thread pool utilization
- Garbage collection pauses
- Disk I/O and capacity
Set alerts at 70% utilization, critical at 85%.
```

### 7.47 Trade Cost Analysis
```
Analyze trading costs comprehensively:
- Commission breakdown by venue and broker
- Exchange fees and rebates
- Borrow costs for short positions
- Funding costs (overnight, intraday)
- Stamp duties and transaction taxes
- Currency conversion costs
- Slippage and market impact
- Compare vs budget and forecasts
Report monthly to PM and CFO.
```

### 7.48 Correlation Monitoring
```
Monitor portfolio correlations:
- Realized correlation vs forecasted
- Correlation breakdown in stress scenarios
- Diversification ratio trends
- Identify correlation regime changes
- Alert on unexpected correlation spikes
- Impact on portfolio risk metrics
- Compare vs historical patterns
- Test portfolio robustness to correlation shifts
Flag correlation >0.7 between "independent" strategies.
```

### 7.49 Capacity Monitoring
```
Track strategy capacity utilization:
- Current AUM vs estimated capacity
- Daily volume as % of market volume
- Market impact trends over time
- Fill rate degradation as size increases
- Compare vs capacity model
- Project capacity headroom
- Alert when approaching capacity (>80%)
- Plan for capacity expansion or strategy closure
Report capacity status monthly to PM.
```

### 7.50 Performance Attribution
```
Attribute performance to sources:
- Alpha (signal-driven returns)
- Execution costs (slippage, commissions, impact)
- Funding costs (borrow, overnight)
- Market exposure (beta, factor tilts)
- Timing (entry/exit efficiency)
- Regime effects (volatility, correlation changes)
- Unexplained residual
- Compare actual vs expected attribution
Produce weekly attribution report.
```

---

## 8. Senior Quantitative Trader Prompts (50)

### Category: Portfolio Management

### 8.1 Multi-Strategy Portfolio Construction
```
Build optimal multi-strategy portfolio:
- Estimate correlation matrix between strategies
- Calculate optimal weights (mean-variance, risk parity, equal Sharpe)
- Incorporate capacity constraints per strategy
- Set diversification targets (max weight, HHI)
- Backtest portfolio vs individual strategies
- Test robustness to correlation estimation error
- Implement dynamic rebalancing rules
- Monitor concentration risk
Target portfolio Sharpe >1.5 with <0.5 correlation between strategies.
```

### 8.2 Capital Allocation Framework
```
Design dynamic capital allocation system:
- Allocate based on recent Sharpe ratio (trailing 60 days)
- Incorporate strategy capacity and scalability
- Penalize strategies in drawdown
- Reward strategies with improving metrics
- Implement minimum/maximum allocation bounds
- Test allocation frequency (daily, weekly, monthly)
- Simulate allocation policy in historical scenarios
- Compare vs static allocation
Document reallocation triggers and governance.
```

### 8.3 Strategy Correlation Management
```
Manage strategy correlations actively:
- Monitor rolling correlation matrix (30/60/90 day)
- Detect correlation regime changes
- Reduce allocations to highly correlated strategies
- Favor decorrelated or negatively correlated pairs
- Test portfolio in correlation stress scenarios
- Calculate effective number of independent bets
- Target diversification ratio >1.5
- Rebalance when correlations exceed thresholds
Alert PM when avg correlation >0.4.
```

### 8.4 Cross-Strategy Risk Budgeting
```
Implement risk budgeting across strategies:
- Allocate risk budget (VaR, volatility) not dollar capital
- Target equal risk contribution or tilted based on Sharpe
- Monitor risk utilization vs budget
- Dynamically adjust leverage per strategy
- Consolidate tail risks and correlation effects
- Test portfolio VaR decomposition
- Ensure no single strategy >30% of total risk
- Rebalance monthly or on material changes
Report risk budget utilization weekly.
```

### 8.5 Portfolio Rebalancing Logic
```
Optimize portfolio rebalancing:
- Define rebalancing triggers (drift thresholds, calendar)
- Calculate optimal trade list (target - current positions)
- Incorporate transaction costs in optimization
- Use tax-loss harvesting where applicable
- Batch small rebalances to reduce costs
- Implement buffer zones (±5% no-trade zone)
- Test rebalancing frequency impact (daily vs weekly vs monthly)
- Monitor tracking error to target portfolio
Target <50 bps annual cost from rebalancing.
```

### 8.6 Drawdown Management
```
Implement systematic drawdown management:
- Monitor max drawdown from peak (strategy and portfolio)
- Define de-risking triggers (-5%, -10%, -15% from peak)
- Scale down leverage or positions proportionally
- Avoid full shutdown unless extreme (-20%+)
- Test recovery time from historical drawdowns
- Communicate drawdown status to stakeholders
- Review strategy fundamentals during drawdown
- Implement recovery plan (gradual ramp-up)
Target max drawdown <15% annually.
```

### 8.7 Leverage Management
```
Manage portfolio leverage dynamically:
- Calculate gross leverage (long + short) and net leverage
- Set leverage limits based on strategy volatility
- Scale leverage inversely with portfolio vol
- Monitor margin utilization vs available capital
- Stress test leverage in adverse scenarios
- Implement automatic deleveraging triggers
- Compare leverage vs peer strategies
- Report leverage metrics daily
Maintain gross leverage <3x for typical strategies.
```

### 8.8 Factor Exposure Hedging
```
Hedge unwanted factor exposures:
- Calculate portfolio factor loadings (market, sector, style)
- Identify significant unintended exposures (|beta| > 0.2)
- Determine hedge instruments (index futures, ETFs)
- Optimize hedge ratios (minimize variance or tail risk)
- Implement static or dynamic hedging
- Monitor hedge effectiveness and costs
- Rebalance hedges on material exposure changes
- Report net factor exposures post-hedge
Target market beta between -0.1 and +0.1 for market-neutral fund.
```

### 8.9 Cash Management and Liquidity
```
Optimize cash management:
- Forecast daily cash needs (margin, redemptions, expenses)
- Maintain liquidity buffer (target 5-10% of AUM)
- Invest excess cash (money market, T-bills, repo)
- Monitor funding costs and optimize financing
- Manage FX cash balances efficiently
- Track cash drag on performance
- Plan for liquidity stress scenarios
- Coordinate with prime broker on funding
Minimize cash drag while ensuring operational liquidity.
```

### 8.10 Multi-Asset Portfolio Integration
```
Integrate strategies across asset classes:
- Trade equities, futures, FX, fixed income, crypto
- Calculate cross-asset correlations and betas
- Optimize portfolio across asset class boundaries
- Manage cross-asset margining and capital efficiency
- Monitor regime-dependent correlations (crisis vs normal)
- Diversify revenue streams across assets
- Test portfolio under asset class stress scenarios
- Report performance by asset class
Target <0.3 correlation between asset class strategies.
```

### Category: Performance Attribution

### 8.11 Daily P&L Attribution
```
Perform detailed daily P&L attribution:
- Decompose into: alpha, execution, funding, market beta
- Attribute to individual strategies and symbols
- Separate realized vs unrealized P&L
- Identify top contributors and detractors
- Explain unexpected P&L movements
- Compare vs forecasted P&L (from backtests)
- Track attribution accuracy over time
- Distribute attribution report to PM and risk
Generate within 2 hours of market close.
```

### 8.12 Risk-Adjusted Return Analysis
```
Calculate comprehensive risk-adjusted metrics:
- Sharpe ratio (daily, monthly, annual)
- Sortino ratio (downside deviation)
- Calmar ratio (return / max drawdown)
- Information ratio vs benchmark
- Omega ratio (probability-weighted returns)
- Treynor ratio (for market exposure)
- Jensen's alpha
- Compare current vs trailing periods
Report monthly with trends and peer comparison.
```

### 8.13 Factor Attribution
```
Attribute returns to systematic factors:
- Estimate factor loadings (market, size, value, momentum, quality)
- Calculate factor returns contribution
- Measure alpha net of factor exposures
- Test stability of factor loadings over time
- Decompose P&L variance by factor
- Identify factor timing effects
- Compare vs factor indices (Fama-French, etc.)
- Report monthly factor attribution
Target alpha >50% of total return.
```

### 8.14 Trade-Level P&L Analysis
```
Analyze P&L at individual trade level:
- Identify highest P&L trades (winners and losers)
- Calculate win rate and average win/loss size
- Measure holding period returns
- Compare entry vs exit decisions quality
- Detect patterns in profitable trades
- Learn from losing trades
- Test trade sizing impact on P&L
- Document lessons learned
Review top 10 winners/losers weekly.
```

### 8.15 Alpha Decay Measurement
```
Measure alpha decay systematically:
- Calculate returns by holding period (1h, 1d, 1w, 1m)
- Plot alpha decay curve
- Identify optimal exit timing
- Compare current vs historical decay rates
- Test if decay is accelerating (signal crowding)
- Attribute decay to market impact vs genuine signal fade
- Adjust strategy holding periods accordingly
- Report alpha half-life
Flag significant changes in decay profile.
```

### 8.16 Cost Attribution
```
Attribute total trading costs:
- Commissions and fees (by venue and broker)
- Market impact and slippage
- Spread costs (bid-ask)
- Financing costs (borrow, repo, margin)
- Taxes and duties
- Operational costs (data, infrastructure)
- Calculate cost per strategy and total
- Compare vs industry benchmarks
Target total costs <50 bps of AUM annually for low-turnover strategies.
```

### 8.17 Regime-Conditional Performance
```
Analyze performance by market regime:
- Classify regimes: bull, bear, high vol, low vol, crisis
- Calculate returns in each regime
- Test if strategy alpha is regime-dependent
- Identify when strategy underperforms
- Optimize strategy mix for all-weather performance
- Hedge regime-specific risks
- Compare current regime vs historical
- Forecast regime transitions
Target positive returns in at least 75% of regimes.
```

### 8.18 Peer Comparison and Benchmarking
```
Benchmark against peer strategies:
- Identify comparable strategies (HF indices, peer funds)
- Compare Sharpe, drawdown, correlation, turnover
- Analyze performance during market stress
- Identify performance gaps and advantages
- Test if outperformance is statistically significant
- Benchmark costs and operational metrics
- Learn from peer best practices
- Report relative performance monthly
Target top quartile performance in peer group.
```

### 8.19 Event Attribution
```
Attribute P&L to specific market events:
- Earnings announcements impact
- Economic data releases (NFP, CPI, Fed)
- Corporate actions (M&A, buybacks, splits)
- Geopolitical events
- Sector rotation and regime shifts
- Quantify event-driven vs baseline P&L
- Test strategy behavior around events
- Adjust strategies to avoid/exploit events
Document major event impacts for pattern recognition.
```

### 8.20 Tax-Adjusted Performance
```
Calculate after-tax performance:
- Track short-term vs long-term gains
- Calculate tax liability by jurisdiction
- Optimize for tax efficiency (holding periods, loss harvesting)
- Report pre-tax and post-tax Sharpe
- Compare tax-adjusted performance vs peers
- Implement tax-loss harvesting strategies
- Monitor wash sale violations
- Coordinate with tax advisors
Particularly important for onshore funds.
```

### Category: Strategy Lifecycle

### 8.21 New Strategy Onboarding
```
Onboard new strategy to production:
- Complete model validation review
- Obtain investment committee approval
- Conduct operational readiness review
- Set up production infrastructure
- Configure risk limits and monitoring
- Paper trade for 2-4 weeks
- Analyze paper trading vs backtest
- Gradual capital allocation (10% → 50% → 100%)
Document onboarding checklist and sign-offs.
```

### 8.22 Strategy Capacity Estimation
```
Estimate strategy capacity rigorously:
- Measure market impact as function of size
- Calculate ADV-based capacity (target <10% ADV)
- Test backtest degradation as AUM scales
- Model slippage and cost increases
- Identify capacity bottlenecks (symbols, times)
- Forecast capacity headroom
- Set soft and hard capacity limits
- Monitor capacity utilization monthly
Conservative estimate better than aggressive.
```

### 8.23 Strategy Performance Review
```
Conduct comprehensive strategy review (monthly):
- P&L vs expectations and budget
- Risk metrics (Sharpe, max DD, volatility)
- Capacity utilization and headroom
- Cost analysis and trends
- Correlation with other strategies
- Market regime impact
- Operational issues and incidents
- Recommended actions (scale, pause, modify, retire)
Present to investment committee.
```

### 8.24 Strategy Modification Process
```
Manage strategy modifications systematically:
- Document proposed change and rationale
- Backtest modification thoroughly
- Estimate impact on P&L, risk, turnover
- Paper trade modified strategy in parallel
- Compare paper vs live baseline strategy
- Obtain PM approval for material changes
- Staged rollout (partial allocation first)
- Monitor for 2-4 weeks post-change
Maintain version control and change log.
```

### 8.25 Underperformance Investigation
```
Investigate strategy underperformance:
- Define underperformance (vs backtest, vs expectations, vs peers)
- Duration and severity of underperformance
- Check for data issues or system bugs
- Analyze market regime changes
- Test if strategy assumptions violated
- Calculate probability of underperformance by chance
- Decide: temporary pause, parameter adjustment, or retirement
- Document findings and recommendations
Act decisively if fundamental issues detected.
```

### 8.26 Strategy Scaling Plan
```
Plan strategy scaling thoughtfully:
- Current AUM and target AUM
- Capacity headroom analysis
- Expected performance degradation curve
- Infrastructure upgrades needed
- Risk limit adjustments
- Operational readiness (staffing, systems)
- Scaling timeline (6-12 months typical)
- Monitoring metrics during scaling
Scale gradually with continuous performance monitoring.
```

### 8.27 Strategy Retirement Decision
```
Retire strategies systematically:
- Retirement criteria: prolonged underperformance, capacity exhausted, alpha decay
- Cost-benefit analysis (maintain vs retire)
- Orderly wind-down plan
- Notify stakeholders (PM, risk, investors)
- Document lessons learned
- Archive code and research for future reference
- Reallocate capital to better opportunities
- Post-retirement monitoring (ensure no residual positions)
Don't hold onto underperforming strategies for sentimental reasons.
```

### 8.28 Strategy Portfolio Optimization
```
Optimize portfolio of strategies:
- Evaluate current strategy mix
- Identify gaps (uncorrelated alphas, diversification)
- Prioritize new strategy development
- Reallocate from crowded to underdeveloped areas
- Balance short-term vs long-term strategies
- Diversify across asset classes and geographies
- Test portfolio under stress scenarios
- Target optimal risk-adjusted returns
Rebalance strategy portfolio annually.
```

### 8.29 Alpha Research Prioritization
```
Prioritize alpha research efforts:
- Evaluate opportunity size (capacity, Sharpe)
- Assess probability of success
- Estimate development time and resources
- Consider strategic fit with existing portfolio
- Avoid over-crowded areas
- Focus on sustainable, non-perishable alpha
- Balance exploration vs exploitation
- Review pipeline quarterly
Focus on high-impact, achievable research.
```

### 8.30 Knowledge Transfer and Documentation
```
Ensure knowledge transfer and documentation:
- Comprehensive strategy documentation
- Code documentation and comments
- Runbooks for operations
- Training for new team members
- Regular knowledge sharing sessions
- Document lessons learned from failures
- Cross-train team on critical strategies
- Maintain institutional memory
Reduce key-person risk.
```

### Category: Risk and Hedging

### 8.31 Value-at-Risk (VaR) Calculation
```
Calculate and monitor VaR:
- Use historical, parametric, and Monte Carlo methods
- Calculate 1-day 95% and 99% VaR
- Decompose VaR by strategy and risk factor
- Backtest VaR model (count exceedances)
- Report VaR daily to PM and risk
- Set VaR limits per strategy and aggregate
- Trigger deleveraging if VaR exceeded
- Stress test beyond VaR (tail risk)
Target <5% VaR exceedance rate for 95% VaR.
```

### 8.32 Stress Testing
```
Conduct comprehensive stress tests:
- Historical scenarios (2008 crisis, COVID, tech bubble)
- Hypothetical scenarios (rates +200bps, market -30%)
- Reverse stress tests (what breaks the portfolio?)
- Multi-factor stress (correlation + volatility shocks)
- Test liquidity stress (wider spreads, lower volume)
- Calculate P&L impact and time to recovery
- Identify vulnerabilities and hedges
- Report stress test results quarterly
Ensure portfolio survives 2008-level events.
```

### 8.33 Tail Risk Management
```
Manage tail risk actively:
- Measure portfolio skewness and kurtosis
- Calculate Expected Shortfall / CVaR (99%)
- Identify fat-tail exposures (options, leverage)
- Implement tail hedges (OTM puts, vol strategies)
- Cost-benefit analysis of tail hedges
- Test portfolio in extreme scenarios (5σ+ events)
- Monitor correlation breakdown in tail events
- Report tail risk metrics monthly
Protect against catastrophic losses (>20% drawdown).
```

### 8.34 Correlation Risk
```
Manage correlation risk:
- Monitor realized correlations vs assumptions
- Test portfolio under correlation stress (all corr → 1)
- Identify correlation regime changes
- Diversify across low-correlation strategies
- Hedge correlation risk (variance swaps, dispersion trades)
- Calculate effective portfolio diversification
- Alert on correlation spikes
- Rebalance when correlations change materially
Target average inter-strategy correlation <0.3.
```

### 8.35 Liquidity Risk Assessment
```
Assess and manage liquidity risk:
- Calculate liquidation time for full portfolio (5-day target)
- Identify illiquid positions (>5% ADV)
- Stress test liquidity (volume drops 50%)
- Maintain liquidity buffer for redemptions
- Diversify across liquid markets
- Set position limits based on ADV
- Monitor bid-ask spreads and depth
- Plan for orderly liquidation in crisis
Test monthly: can portfolio be liquidated in 5 days?
```

### 8.36 Concentration Risk
```
Monitor concentration risk:
- Maximum single position size (target <5% of capital)
- Top 10 positions as % of portfolio
- Sector/industry concentration (max 20% per sector)
- Geographic concentration
- Factor concentration
- Calculate HHI (Herfindahl index)
- Alert on excessive concentration
- Diversify or hedge concentrated exposures
Maintain HHI <0.1 for well-diversified portfolio.
```

### 8.37 Counterparty Risk
```
Manage counterparty risk:
- Monitor exposure to each broker/prime broker
- Diversify across multiple counterparties
- Track counterparty credit ratings
- Maintain ISDA and CSA agreements
- Monitor margin and collateral requirements
- Stress test counterparty default scenarios
- Maintain backup prime broker relationships
- Report counterparty exposure monthly
No single counterparty >40% of total exposure.
```

### 8.38 Operational Risk
```
Identify and mitigate operational risks:
- System failures and downtime
- Data quality issues
- Fat finger errors
- Fraud and insider threats
- Vendor dependencies
- Cybersecurity threats
- Document operational risk register
- Implement controls and monitoring
Conduct operational risk review quarterly.
```

### 8.39 Hedging Strategy Design
```
Design effective hedging strategies:
- Identify primary risk factors to hedge
- Select hedge instruments (futures, options, swaps)
- Calculate optimal hedge ratios
- Choose static vs dynamic hedging
- Measure hedge effectiveness and costs
- Rebalance hedges periodically
- Monitor basis risk (hedge vs underlying)
- Report hedge P&L separately
Target hedge effectiveness >80%.
```

### 8.40 Portfolio Greeks Management
```
Manage portfolio Greeks (for options strategies):
- Calculate Delta, Gamma, Vega, Theta, Rho
- Set limits on each Greek
- Hedge unwanted exposures (e.g., delta-neutral)
- Monitor Greeks in real-time
- Test Greeks sensitivity to large market moves
- Rebalance to stay within Greek limits
- Report Greeks daily
- Stress test option positions
Particularly critical for volatility strategies.
```

### Category: KPI Tracking and Reporting

### 8.41 Daily Performance Flash
```
Produce daily performance flash report:
- P&L (daily, MTD, YTD)
- Sharpe ratio (daily, trailing 20/60/120 days)
- Top strategy and position contributors
- Risk metrics (VaR, leverage, concentration)
- Key alerts and incidents
- Market context (indices, VIX, rates)
- Brief commentary on drivers
- Distribute by 8am following day
Keep concise (1-2 pages max).
```

### 8.42 Weekly Performance Review
```
Conduct weekly performance review:
- Detailed P&L by strategy and asset class
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Performance vs budget and targets
- Strategy health scorecard
- Trade analysis (winners, losers, patterns)
- Cost breakdown
- Upcoming risks and events
- Action items for following week
Present to PM and investment team.
```

### 8.43 Monthly Investor Reporting
```
Prepare monthly investor report:
- Performance summary (returns, Sharpe, drawdown)
- AUM and flow analysis
- Strategy allocation and changes
- Risk metrics and positioning
- Market commentary and outlook
- Top holdings and exposures
- Fee reconciliation
- Compliance attestations
Professional formatting, submit within 10 business days of month-end.
```

### 8.44 Quarterly Performance Attribution
```
Deep-dive quarterly performance attribution:
- Return decomposition (alpha, beta, factors, costs)
- Strategy-by-strategy analysis
- Regime analysis (bull, bear, high vol, low vol)
- Peer comparison and benchmarking
- Hit rate and trade quality metrics
- Capacity and scalability review
- Forward-looking changes and initiatives
- Lessons learned
Present to investment committee.
```

### 8.45 Annual Strategy Review
```
Comprehensive annual strategy review:
- Full-year performance vs objectives
- Multi-year track record and consistency
- Risk and return evolution
- Capacity progression
- Major changes and their impact
- Competitive positioning
- Strategic priorities for next year
- Team and resource needs
Inform annual planning and budgeting.
```

### 8.46 Real-Time Risk Dashboard
```
Build real-time risk dashboard:
- Live VaR and stress test results
- Position concentrations
- Factor exposures
- Leverage and margin utilization
- Correlation matrix heatmap
- Alerts and limit breaches
- P&L and drawdown from peak
- Sharpe ratio trends
Accessible to PM, risk, and senior management.
```

### 8.47 Trade Analytics
```
Comprehensive trade analytics:
- Total trades (count and volume)
- Average trade size and holding period
- Win rate and profit factor
- Best and worst trades
- Trade costs breakdown
- Execution quality (vs VWAP, arrival)
- Turnover analysis
- Impact of trade sizing on outcomes
Review weekly, optimize trade execution.
```

### 8.48 Capacity Utilization Report
```
Monitor capacity utilization:
- Current AUM vs estimated capacity by strategy
- ADV utilization (% of market volume)
- Impact on execution quality over time
- Capacity headroom projections
- Risks of overcapacity
- Scaling plans or capacity expansion
- New strategy pipeline to absorb growth
- Report monthly to PM
Plan ahead when approaching 70% capacity.
```

### 8.49 Cost Analysis Dashboard
```
Track all costs systematically:
- Trading costs (commissions, impact, spread)
- Data and technology costs
- Personnel costs (allocated to strategies)
- Infrastructure and overhead
- Cost per strategy (in bps of AUM)
- Cost trends over time
- Benchmark vs industry standards
- Cost optimization opportunities
Target industry-competitive cost structure.
```

### 8.50 Executive Summary for Leadership
```
Monthly executive summary for CIO/CEO:
- One-page overview of firm performance
- Key wins and challenges
- Material risks and mitigants
- Strategic initiatives progress
- Competitive intelligence
- Regulatory or market developments
- Resource needs or bottlenecks
- Forward-looking priorities
Focus on decision-relevant information.
```

---

**End of Document**
Total: 400 prompts across 8 Claude Code Skills

Generated for comprehensive prompt engineering reference.
