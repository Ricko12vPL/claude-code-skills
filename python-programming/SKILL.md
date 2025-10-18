---
name: python-programming
description: Expert Python programming guidance covering modern best practices, design patterns, testing, type hints, and performance optimization. Use when writing Python code, refactoring, debugging, or implementing Python projects.
---

# Python Programming

Expert guidance for writing clean, efficient, and maintainable Python code following modern best practices and industry standards.

## Core Principles

### Code Style and Standards
- Follow PEP 8 style guide for code formatting
- Use PEP 257 for docstring conventions
- Implement type hints (PEP 484) for better code clarity
- Maximum line length: 88 characters (Black formatter standard)
- Use meaningful variable and function names that convey intent

### Project Structure
```
project_name/
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       └── modules/
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── docs/
├── pyproject.toml
├── README.md
├── .gitignore
└── requirements.txt
```

## Modern Python Features

### Type Hints
Always use type hints for function signatures and class attributes:

```python
from typing import List, Dict, Optional, Union, Callable
from collections.abc import Sequence

def process_data(
    items: List[str],
    config: Dict[str, int],
    callback: Optional[Callable[[str], None]] = None
) -> List[int]:
    """Process items according to configuration.
    
    Args:
        items: List of strings to process
        config: Configuration dictionary
        callback: Optional callback function
        
    Returns:
        List of processed integers
    """
    results = []
    for item in items:
        if callback:
            callback(item)
        results.append(len(item) * config.get('multiplier', 1))
    return results
```

### Dataclasses
Use dataclasses for simple data containers:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name: str
    email: str
    age: int
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age must be positive")
```

### Context Managers
Always use context managers for resource management:

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def managed_resource() -> Generator[Resource, None, None]:
    """Context manager for automatic resource cleanup."""
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)

# Usage
with managed_resource() as resource:
    resource.use()
```

## Design Patterns

### Singleton Pattern
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Factory Pattern
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        animals = {
            'dog': Dog,
            'cat': Cat
        }
        animal_class = animals.get(animal_type.lower())
        if not animal_class:
            raise ValueError(f"Unknown animal type: {animal_type}")
        return animal_class()
```

### Decorator Pattern
```python
from functools import wraps
import time
from typing import Callable, Any

def timing_decorator(func: Callable) -> Callable:
    """Measure function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"
```

## Error Handling

### Custom Exceptions
```python
class ApplicationError(Exception):
    """Base exception for application errors."""
    pass

class ValidationError(ApplicationError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid."""
    pass

# Usage
def validate_email(email: str) -> None:
    if '@' not in email:
        raise ValidationError(f"Invalid email format: {email}")
```

### Exception Handling Best Practices
```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers with error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        logger.error(f"Attempted to divide {a} by zero")
        return None
    except TypeError as e:
        logger.error(f"Type error in division: {e}")
        raise
```

## Testing

### Pytest Framework
```python
import pytest
from typing import List

# Fixtures
@pytest.fixture
def sample_data() -> List[int]:
    return [1, 2, 3, 4, 5]

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input: int, expected: int):
    assert input ** 2 == expected

# Test with fixtures
def test_sum(sample_data: List[int]):
    assert sum(sample_data) == 15

# Exception testing
def test_invalid_input():
    with pytest.raises(ValueError, match="Invalid input"):
        raise ValueError("Invalid input")
```

### Mocking
```python
from unittest.mock import Mock, patch, MagicMock

def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'key': 'value'}
        
        response = make_api_call()
        
        assert response['key'] == 'value'
        mock_get.assert_called_once()
```

## Performance Optimization

### List Comprehensions vs Loops
```python
# Efficient - use list comprehension
squares = [x**2 for x in range(1000)]

# Even better - use generator for large datasets
squares_gen = (x**2 for x in range(1000000))

# Dictionary comprehension
word_lengths = {word: len(word) for word in words}
```

### Using Built-in Functions
```python
from operator import itemgetter
from itertools import groupby, islice

# Efficient sorting
sorted_items = sorted(items, key=itemgetter('priority'))

# Grouping data
grouped = {k: list(g) for k, g in groupby(sorted_data, key=lambda x: x['category'])}

# Memory-efficient iteration
for item in islice(large_iterator, 1000):
    process(item)
```

### Profiling
```python
import cProfile
import pstats
from functools import wraps

def profile(func):
    """Profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        return result
    return wrapper
```

## Async Programming

### Asyncio Basics
```python
import asyncio
from typing import List

async def fetch_data(url: str) -> str:
    """Async function to fetch data."""
    await asyncio.sleep(1)  # Simulate network call
    return f"Data from {url}"

async def main():
    urls = ['url1', 'url2', 'url3']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Run async code
if __name__ == "__main__":
    results = asyncio.run(main())
```

### Async Context Managers
```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        await asyncio.sleep(0.1)
    
    async def disconnect(self):
        await asyncio.sleep(0.1)

# Usage
async def use_resource():
    async with AsyncResource() as resource:
        # Use resource
        pass
```

## Dependencies Management

### Using pyproject.toml
```toml
[project]
name = "my-project"
version = "0.1.0"
description = "Project description"
authors = [{name = "Your Name", email = "you@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

## Logging

### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Setup
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

## Best Practices Checklist

- [ ] Use type hints for all function signatures
- [ ] Write docstrings for all public functions and classes
- [ ] Handle exceptions explicitly and log errors
- [ ] Use context managers for resource management
- [ ] Write unit tests with >80% coverage
- [ ] Use virtual environments (venv or conda)
- [ ] Format code with Black or Ruff
- [ ] Run type checking with mypy
- [ ] Use logging instead of print statements
- [ ] Keep functions small and focused (single responsibility)
- [ ] Use descriptive variable names
- [ ] Avoid global variables
- [ ] Use pathlib for file operations
- [ ] Implement proper error messages
- [ ] Document complex algorithms and business logic

## Common Pitfalls to Avoid

1. **Mutable Default Arguments**
   ```python
   # Wrong
   def append_to(element, target=[]):
       target.append(element)
       return target
   
   # Correct
   def append_to(element, target=None):
       if target is None:
           target = []
       target.append(element)
       return target
   ```

2. **Not Using Generators for Large Data**
   ```python
   # Memory inefficient
   def get_all_lines(filename):
       return [line for line in open(filename)]
   
   # Efficient
   def get_all_lines(filename):
       with open(filename) as f:
           for line in f:
               yield line.strip()
   ```

3. **Catching Too Broad Exceptions**
   ```python
   # Wrong
   try:
       risky_operation()
   except Exception:
       pass
   
   # Correct
   try:
       risky_operation()
   except SpecificError as e:
       logger.error(f"Operation failed: {e}")
       raise
   ```

## Tools and Libraries

### Essential Development Tools
- **Black**: Code formatter
- **Ruff**: Fast Python linter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pre-commit**: Git hooks for code quality

### Command Line Usage
```bash
# Format code
black src/

# Lint code
ruff check src/

# Type check
mypy src/

# Run tests with coverage
pytest --cov=src tests/

# Install dev dependencies
pip install -e ".[dev]"
```

## References

- PEP 8: Style Guide for Python Code
- PEP 20: The Zen of Python
- PEP 257: Docstring Conventions
- Python Type Hints: PEP 484, 526, 544
- Official Python Documentation: https://docs.python.org
