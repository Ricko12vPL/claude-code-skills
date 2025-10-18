---
name: software-engineering
description: Comprehensive software engineering practices covering system design, architecture patterns, SOLID principles, clean code, testing strategies, CI/CD, and development workflows. Use when designing systems, reviewing code, implementing best practices, or planning software architecture.
---

# Software Engineering

Expert guidance for building robust, scalable, and maintainable software systems using industry-standard practices and architectural patterns.

## Core Software Engineering Principles

### SOLID Principles

#### Single Responsibility Principle (SRP)
Each class should have only one reason to change.

```python
# Bad - Multiple responsibilities
class User:
    def __init__(self, name):
        self.name = name
    
    def save_to_database(self):
        # Database logic
        pass
    
    def send_email(self):
        # Email logic
        pass

# Good - Separated responsibilities
class User:
    def __init__(self, name):
        self.name = name

class UserRepository:
    def save(self, user: User):
        # Database logic
        pass

class EmailService:
    def send_welcome_email(self, user: User):
        # Email logic
        pass
```

#### Open/Closed Principle (OCP)
Software entities should be open for extension but closed for modification.

```python
from abc import ABC, abstractmethod

# Good - Extension through inheritance
class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

class CreditCardProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        # Credit card processing
        return True

class PayPalProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        # PayPal processing
        return True
```

#### Liskov Substitution Principle (LSP)
Subtypes must be substitutable for their base types.

```python
class Rectangle:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    def area(self) -> int:
        return self._width * self._height

# Bad - Square violates LSP
class Square(Rectangle):
    def __init__(self, side: int):
        super().__init__(side, side)
    
    @Rectangle.width.setter
    def width(self, value: int):
        self._width = value
        self._height = value  # Violates LSP

# Good - Separate implementations
class Shape(ABC):
    @abstractmethod
    def area(self) -> int:
        pass

class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def area(self) -> int:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: int):
        self.side = side
    
    def area(self) -> int:
        return self.side * self.side
```

#### Interface Segregation Principle (ISP)
Clients should not depend on interfaces they don't use.

```python
from abc import ABC, abstractmethod

# Bad - Fat interface
class Worker(ABC):
    @abstractmethod
    def work(self):
        pass
    
    @abstractmethod
    def eat(self):
        pass

# Good - Segregated interfaces
class Workable(ABC):
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    @abstractmethod
    def eat(self):
        pass

class Human(Workable, Eatable):
    def work(self):
        return "Working"
    
    def eat(self):
        return "Eating"

class Robot(Workable):
    def work(self):
        return "Working"
```

#### Dependency Inversion Principle (DIP)
Depend on abstractions, not concretions.

```python
from abc import ABC, abstractmethod

# Good - Dependency injection with abstraction
class Database(ABC):
    @abstractmethod
    def save(self, data: dict):
        pass

class PostgreSQL(Database):
    def save(self, data: dict):
        # PostgreSQL specific implementation
        pass

class MongoDB(Database):
    def save(self, data: dict):
        # MongoDB specific implementation
        pass

class UserService:
    def __init__(self, database: Database):
        self.database = database
    
    def create_user(self, user_data: dict):
        self.database.save(user_data)
```

## Architectural Patterns

### Layered Architecture
```
┌─────────────────────────────────┐
│     Presentation Layer          │  (API, UI, Controllers)
├─────────────────────────────────┤
│     Business Logic Layer        │  (Services, Domain Models)
├─────────────────────────────────┤
│     Data Access Layer           │  (Repositories, ORMs)
├─────────────────────────────────┤
│     Database Layer              │  (SQL, NoSQL)
└─────────────────────────────────┘
```

```python
# Presentation Layer
class UserController:
    def __init__(self, user_service: UserService):
        self.user_service = user_service
    
    def create_user(self, request: dict) -> dict:
        return self.user_service.create_user(request)

# Business Logic Layer
class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    def create_user(self, data: dict) -> User:
        user = User(**data)
        return self.user_repository.save(user)

# Data Access Layer
class UserRepository:
    def __init__(self, db: Database):
        self.db = db
    
    def save(self, user: User) -> User:
        return self.db.insert('users', user.to_dict())
```

### Microservices Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Service A  │────▶│  API Gateway │◀────│   Service B  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
  ┌─────────┐          ┌─────────┐          ┌─────────┐
  │  DB A   │          │ Message │          │  DB B   │
  └─────────┘          │  Queue  │          └─────────┘
                       └─────────┘
```

Key Principles:
- Single responsibility per service
- Independent deployment
- Decentralized data management
- Infrastructure automation
- Design for failure

### Event-Driven Architecture
```python
from abc import ABC, abstractmethod
from typing import List, Callable

class Event:
    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.data = data

class EventBus:
    def __init__(self):
        self._subscribers: dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

# Usage
event_bus = EventBus()

def on_user_created(event: Event):
    print(f"Sending welcome email to {event.data['email']}")

event_bus.subscribe('user.created', on_user_created)
event_bus.publish(Event('user.created', {'email': 'user@example.com'}))
```

### CQRS (Command Query Responsibility Segregation)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Commands (write operations)
@dataclass
class CreateUserCommand:
    username: str
    email: str

class CommandHandler(ABC):
    @abstractmethod
    def handle(self, command):
        pass

class CreateUserCommandHandler(CommandHandler):
    def __init__(self, repository: UserRepository):
        self.repository = repository
    
    def handle(self, command: CreateUserCommand):
        user = User(command.username, command.email)
        return self.repository.save(user)

# Queries (read operations)
@dataclass
class GetUserQuery:
    user_id: str

class QueryHandler(ABC):
    @abstractmethod
    def handle(self, query):
        pass

class GetUserQueryHandler(QueryHandler):
    def __init__(self, read_repository: UserReadRepository):
        self.read_repository = read_repository
    
    def handle(self, query: GetUserQuery):
        return self.read_repository.find_by_id(query.user_id)
```

## Design Patterns

### Creational Patterns

#### Factory Method
```python
class Document(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class PDFDocument(Document):
    def render(self) -> str:
        return "Rendering PDF"

class WordDocument(Document):
    def render(self) -> str:
        return "Rendering Word"

class DocumentFactory:
    @staticmethod
    def create(doc_type: str) -> Document:
        if doc_type == 'pdf':
            return PDFDocument()
        elif doc_type == 'word':
            return WordDocument()
        raise ValueError(f"Unknown document type: {doc_type}")
```

#### Builder Pattern
```python
class Pizza:
    def __init__(self):
        self.size = None
        self.cheese = False
        self.pepperoni = False
        self.mushrooms = False

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size: str) -> 'PizzaBuilder':
        self.pizza.size = size
        return self
    
    def add_cheese(self) -> 'PizzaBuilder':
        self.pizza.cheese = True
        return self
    
    def add_pepperoni(self) -> 'PizzaBuilder':
        self.pizza.pepperoni = True
        return self
    
    def build(self) -> Pizza:
        return self.pizza

# Usage
pizza = (PizzaBuilder()
         .set_size('large')
         .add_cheese()
         .add_pepperoni()
         .build())
```

### Structural Patterns

#### Adapter Pattern
```python
class EuropeanSocket:
    def voltage(self) -> int:
        return 230
    
    def live(self) -> int:
        return 1
    
    def neutral(self) -> int:
        return -1

class USASocket:
    def voltage(self) -> int:
        return 120

class Adapter:
    def __init__(self, socket: EuropeanSocket):
        self.socket = socket
    
    def voltage(self) -> int:
        return 110  # Convert voltage
```

#### Decorator Pattern
```python
from abc import ABC, abstractmethod

class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass

class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 2.0
    
    def description(self) -> str:
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    
    def cost(self) -> float:
        return self._coffee.cost()
    
    def description(self) -> str:
        return self._coffee.description()

class Milk(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5
    
    def description(self) -> str:
        return self._coffee.description() + ", milk"

# Usage
coffee = SimpleCoffee()
coffee = Milk(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
```

### Behavioral Patterns

#### Strategy Pattern
```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass

class QuickSort(SortStrategy):
    def sort(self, data: list) -> list:
        # Quick sort implementation
        return sorted(data)

class MergeSort(SortStrategy):
    def sort(self, data: list) -> list:
        # Merge sort implementation
        return sorted(data)

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data: list) -> list:
        return self._strategy.sort(data)

# Usage
sorter = Sorter(QuickSort())
result = sorter.sort([3, 1, 4, 1, 5, 9])
```

#### Observer Pattern
```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject: 'Subject'):
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
```

## Clean Code Principles

### Meaningful Names
```python
# Bad
def calc(d):
    return d * 24 * 60 * 60

# Good
def calculate_seconds_from_days(days: int) -> int:
    SECONDS_PER_DAY = 86400
    return days * SECONDS_PER_DAY
```

### Functions Should Do One Thing
```python
# Bad
def process_user(user_data):
    # Validate
    if not user_data.get('email'):
        raise ValueError("Email required")
    # Create user
    user = User(user_data)
    # Save to DB
    db.save(user)
    # Send email
    send_welcome_email(user.email)

# Good
def validate_user_data(user_data: dict):
    if not user_data.get('email'):
        raise ValueError("Email required")

def create_user(user_data: dict) -> User:
    return User(user_data)

def process_user(user_data: dict):
    validate_user_data(user_data)
    user = create_user(user_data)
    user_repository.save(user)
    email_service.send_welcome_email(user.email)
```

### Don't Repeat Yourself (DRY)
```python
# Bad
def calculate_circle_area(radius):
    return 3.14159 * radius * radius

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

# Good
PI = 3.14159

def calculate_circle_area(radius: float) -> float:
    return PI * radius ** 2

def calculate_circle_circumference(radius: float) -> float:
    return 2 * PI * radius
```

## Testing Strategies

### Test Pyramid
```
        ┌─────────┐
        │   E2E   │  (Few, slow, expensive)
        ├─────────┤
        │ Integration│ (Some, moderate speed)
        ├─────────┤
        │   Unit  │  (Many, fast, cheap)
        └─────────┘
```

### Unit Testing Best Practices
```python
import pytest
from unittest.mock import Mock, patch

class TestUserService:
    @pytest.fixture
    def user_repository(self):
        return Mock()
    
    @pytest.fixture
    def user_service(self, user_repository):
        return UserService(user_repository)
    
    def test_create_user_success(self, user_service, user_repository):
        # Arrange
        user_data = {'name': 'John', 'email': 'john@example.com'}
        user_repository.save.return_value = User(**user_data)
        
        # Act
        result = user_service.create_user(user_data)
        
        # Assert
        assert result.name == 'John'
        user_repository.save.assert_called_once()
    
    def test_create_user_invalid_email(self, user_service):
        # Arrange
        user_data = {'name': 'John', 'email': 'invalid'}
        
        # Act & Assert
        with pytest.raises(ValidationError):
            user_service.create_user(user_data)
```

### Integration Testing
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope='module')
def db_session():
    engine = create_engine('sqlite:///:memory:')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Setup
    Base.metadata.create_all(engine)
    
    yield session
    
    # Teardown
    session.close()

def test_user_crud_operations(db_session):
    # Create
    user = User(name='John', email='john@example.com')
    db_session.add(user)
    db_session.commit()
    
    # Read
    retrieved_user = db_session.query(User).filter_by(email='john@example.com').first()
    assert retrieved_user is not None
    assert retrieved_user.name == 'John'
    
    # Update
    retrieved_user.name = 'Jane'
    db_session.commit()
    
    # Delete
    db_session.delete(retrieved_user)
    db_session.commit()
```

## CI/CD Best Practices

### GitHub Actions Example
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with ruff
      run: ruff check .
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Run tests with coverage
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Deployment commands here
        echo "Deploying to production"
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Code Review Checklist

### Functionality
- [ ] Code does what it's supposed to do
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs or logic errors

### Code Quality
- [ ] Follows SOLID principles
- [ ] DRY - no repeated code
- [ ] Functions are small and focused
- [ ] Clear and meaningful names
- [ ] Appropriate comments for complex logic

### Testing
- [ ] Unit tests cover new code
- [ ] Integration tests if needed
- [ ] Tests are clear and maintainable
- [ ] Edge cases are tested

### Security
- [ ] No sensitive data exposed
- [ ] Input validation present
- [ ] SQL injection prevented
- [ ] XSS protection in place
- [ ] Authentication/authorization checked

### Performance
- [ ] No obvious performance issues
- [ ] Database queries optimized
- [ ] Caching used where appropriate
- [ ] Memory usage considered

### Documentation
- [ ] Code is self-documenting
- [ ] Complex logic has comments
- [ ] Public APIs documented
- [ ] README updated if needed

## API Design Best Practices

### RESTful API Design
```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

# GET /api/v1/users - List all users
@app.get("/api/v1/users", response_model=list[UserResponse])
async def list_users(skip: int = 0, limit: int = 100):
    return get_users(skip, limit)

# POST /api/v1/users - Create user
@app.post("/api/v1/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    return create_user_in_db(user)

# GET /api/v1/users/{user_id} - Get specific user
@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# PUT /api/v1/users/{user_id} - Update user
@app.put("/api/v1/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user: UserCreate):
    return update_user_in_db(user_id, user)

# DELETE /api/v1/users/{user_id} - Delete user
@app.delete("/api/v1/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    delete_user_from_db(user_id)
```

### API Versioning
```python
# URL versioning (recommended)
@app.get("/api/v1/users")
@app.get("/api/v2/users")

# Header versioning
@app.get("/api/users")
async def get_users(accept_version: str = Header(default="v1")):
    if accept_version == "v1":
        return get_users_v1()
    elif accept_version == "v2":
        return get_users_v2()
```

## Database Design Principles

### Normalization
- **1NF**: Eliminate repeating groups
- **2NF**: Remove partial dependencies
- **3NF**: Remove transitive dependencies

### Indexing Strategy
```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Composite index for multiple columns
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
```

### Migration Best Practices
```python
# alembic migration
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_users_email', 'users', ['email'])

def downgrade():
    op.drop_index('idx_users_email', 'users')
    op.drop_table('users')
```

## Monitoring and Observability

### Logging
```python
import logging
import structlog

# Structured logging
logger = structlog.get_logger()

logger.info(
    "user_created",
    user_id=user.id,
    email=user.email,
    source="api"
)
```

### Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration'
)

# Use metrics
@request_duration.time()
def handle_request():
    result = process_request()
    request_count.labels(method='GET', endpoint='/users', status=200).inc()
    return result
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": check_database_connection(),
        "cache": check_cache_connection(),
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Security Best Practices

### Input Validation
```python
from pydantic import BaseModel, EmailStr, constr, validator

class UserInput(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    password: constr(min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v
```

### Authentication & Authorization
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return get_user(user_id)
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": f"Hello {current_user.username}"}
```

## Performance Optimization

### Caching Strategy
```python
from functools import lru_cache
import redis

# In-memory cache
@lru_cache(maxsize=128)
def get_user_by_id(user_id: int):
    return db.query(User).get(user_id)

# Redis cache
redis_client = redis.Redis()

def cached_get_user(user_id: int):
    cache_key = f"user:{user_id}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    user = db.query(User).get(user_id)
    redis_client.setex(cache_key, 3600, json.dumps(user.to_dict()))
    return user
```

### Database Query Optimization
```python
# Bad - N+1 query problem
users = User.query.all()
for user in users:
    print(user.orders)  # Separate query for each user

# Good - Use eager loading
users = User.query.options(joinedload(User.orders)).all()
for user in users:
    print(user.orders)  # No additional queries
```

## Documentation Best Practices

### API Documentation
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="API for managing users and orders",
    version="1.0.0"
)

@app.post("/users", 
    response_model=UserResponse,
    summary="Create a new user",
    description="Create a new user with the provided information",
    response_description="The created user"
)
async def create_user(user: UserCreate):
    """
    Create a new user with all the required information:
    
    - **name**: User's full name
    - **email**: Valid email address
    - **password**: Strong password (min 8 characters)
    """
    return create_user_in_db(user)
```

### Architecture Documentation
```markdown
# System Architecture

## Overview
Brief description of the system

## Components
- API Gateway: Routes requests to appropriate services
- User Service: Handles user management
- Order Service: Processes orders
- Payment Service: Handles payments

## Data Flow
1. Client sends request to API Gateway
2. Gateway authenticates and routes to service
3. Service processes request
4. Response sent back through gateway

## Technology Stack
- FastAPI for API framework
- PostgreSQL for database
- Redis for caching
- Docker for containerization
```

## Best Practices Summary

1. **Follow SOLID principles** religiously
2. **Write clean, readable code** over clever code
3. **Test extensively** - aim for >80% coverage
4. **Document your decisions** - especially non-obvious ones
5. **Review code thoroughly** before merging
6. **Monitor in production** - logs, metrics, alerts
7. **Secure by default** - validate input, sanitize output
8. **Optimize for readability first**, performance second
9. **Use version control effectively** - meaningful commits
10. **Keep dependencies updated** and minimal

## Tools and Resources

### Essential Tools
- **Git**: Version control
- **Docker**: Containerization
- **Jenkins/GitHub Actions**: CI/CD
- **SonarQube**: Code quality
- **Prometheus**: Monitoring
- **Grafana**: Visualization
- **ELK Stack**: Logging

### Recommended Reading
- Clean Code by Robert C. Martin
- Design Patterns by Gang of Four
- Domain-Driven Design by Eric Evans
- The Pragmatic Programmer
- Building Microservices by Sam Newman
