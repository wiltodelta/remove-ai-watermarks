---
name: python-code-review
description: Performs comprehensive code reviews for Python files following PEP 8 and Google Python Style Guide standards. Checks code quality, best practices, security, performance, maintainability, and style compliance. Use when reviewing Python code or when asked to check, audit, or improve Python code quality.
---

# Python Code Review (PEP 8 + Google Style Guide)

Perform systematic code reviews of Python files following PEP 8 and Google Python Style Guide standards.

## Review Philosophy

> "Code is read much more often than it is written." - Guido van Rossum

**Key Principle**: A foolish consistency is the hobgoblin of little minds. Consistency within a project is more important than rigid adherence to rules. When in doubt, prioritize:
1. Consistency within one function/module (most important)
2. Consistency within the project
3. Consistency with PEP 8/Google Style Guide

Know when to be inconsistent:
- When applying the guideline makes code less readable
- To match surrounding code style (but consider refactoring)
- When code predates the guideline
- For backwards compatibility

## Review Process

When reviewing Python code:

1. **Read the entire file** to understand its purpose, structure, and context

2. **Analyze against these standards** in order of priority:

### 1. Style & Formatting (PEP 8)

**Line Length**
- Maximum 80 characters (Google)
- Docstrings/comments: 72 characters max
- Use implicit line continuation (parentheses/brackets) over backslashes

**Indentation**
- Always 4 spaces per level, never tabs
- Continuation lines: align vertically or use 4-space hanging indent
- Closing brackets: align under first non-whitespace or under opening bracket

**Blank Lines**
- 2 blank lines between top-level functions/classes
- 1 blank line between methods
- Sparingly within functions for logical sections
- No blank line after `def` line

**Whitespace Rules**
- No whitespace inside parentheses/brackets/braces: `spam(ham[1], {eggs: 2})`
- No whitespace before comma/semicolon/colon (except in slices)
- No whitespace before function call parentheses: `spam(1)` not `spam (1)`
- No whitespace before indexing brackets: `dct['key']` not `dct ['key']`
- Single space around binary operators: `i = i + 1`
- No spaces around `=` in keyword args: `complex(real, imag=0.0)`
- BUT use spaces when combining annotation + default: `def munge(input: AnyStr = None)`
- Don't align operators vertically across lines (maintenance burden)

**String Quotes**
- Be consistent: pick `'` or `"` and stick with it in a file
- Use the other quote to avoid backslashes: `"He said 'hello'"`
- Always use `"""` for docstrings (never `'''`)

**Trailing Commas**
- Recommended for multi-line structures when closing bracket is on new line
- Mandatory for single-element tuples: `FILES = ('setup.cfg',)`
- Not on same line as closing bracket (except singleton tuples)

### 2. Imports (PEP 8 + Google)

**Import Order** (with blank line between groups):
1. `from __future__` imports
2. Standard library imports
3. Third-party imports
4. Local application/library imports

**Import Style**
- Separate lines: `import os` and `import sys` (not `import os, sys`)
- Exception: OK to import multiple items from one module: `from subprocess import Popen, PIPE`
- Exception: Typing imports: `from typing import Any, NewType`
- Use absolute imports (recommended): `import mypkg.sibling`
- Relative imports acceptable for complex packages: `from . import sibling`
- Never use wildcard imports: `from module import *`
- Import full package paths (Google): `from doctor.who import jodie` not `import jodie`

**Import Format**
- `import x` for packages and modules
- `from x import y` where x is package prefix, y is module name
- `from x import y as z` if y conflicts or is inconveniently long
- `import y as z` only for standard abbreviations: `import numpy as np`

**Module Dunders**
- After module docstring, before imports (except `__future__`)
- Order: `__all__`, `__version__`, `__author__`, etc.

### 3. Naming Conventions (PEP 8 + Google)

| Type | Convention | Examples |
|------|------------|----------|
| **Modules** | `lower_with_under.py` | `my_module.py` |
| **Packages** | `lower_with_under` | `my_package` |
| **Classes** | `CapWords` | `MyClass`, `HTTPServerError` |
| **Exceptions** | `CapWords` + `Error` suffix | `ValueError`, `ConnectionError` |
| **Functions** | `lower_with_under()` | `my_function()` |
| **Methods** | `lower_with_under()` | `my_method()` |
| **Constants** | `CAPS_WITH_UNDER` | `MAX_OVERFLOW`, `TOTAL` |
| **Global/Class Variables** | `lower_with_under` | `global_var` |
| **Instance Variables** | `lower_with_under` | `instance_var` |
| **Parameters** | `lower_with_under` | `function_param` |
| **Local Variables** | `lower_with_under` | `local_var` |
| **Type Variables** | `_T`, `_P` (leading underscore) | `_T = TypeVar("_T")` |

**Special Prefixes/Suffixes**
- `_single_leading`: weak "internal use" indicator (not imported by `from M import *`)
- `single_trailing_`: avoid keyword conflicts (`class_`)
- `__double_leading`: name mangling in classes (discouraged by Google - impacts testability)
- `__double_leading_and_trailing__`: magic methods (never invent these)

**Names to Avoid**
- Never use `l` (lowercase L), `O` (uppercase o), `I` (uppercase i) as single-char names
- No dashes in any package/module name
- Avoid abbreviations unless well-known
- No offensive terms
- No needless type info: `id_to_name_dict` → `id_to_name`

**Descriptive Names**
- Names should be descriptive and clear
- Descriptiveness proportional to scope (wider scope = more descriptive)
- Single-char names OK for: counters (`i`, `j`, `k`), exceptions (`e`), file handles (`f`), type vars (`_T`, `_P`)
- Avoid vague names: `thing`, `stuff`, `data` (without context)

### 4. Documentation (PEP 257 + Google)

**Module Docstrings** (required)
```python
"""One-line summary ending with period.

Longer description of the module or program. May include usage
examples, exported classes/functions, etc.

Typical usage example:
  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

**Function/Method Docstrings** (required for complex/public functions)
```python
def fetch_data(
    table: str,
    keys: Sequence[str],
    require_all: bool = False,
) -> Mapping[str, tuple[str, ...]]:
    """Fetches rows from database.

    Retrieves rows pertaining to given keys. Longer description
    of what the function does and any important details.

    Args:
        table: Name of the database table.
        keys: List of row keys to fetch. Strings will be UTF-8 encoded.
        require_all: If True, only return rows with all keys present.

    Returns:
        Dict mapping keys to row data. Each row is a tuple of strings.
        Returns empty dict if no rows found.

    Raises:
        IOError: Error accessing the database.
        ValueError: Invalid table name provided.
    """
```

**Docstring Sections** (Google style):
- **Summary line**: One physical line (≤80 chars), ends with period
- **Blank line** after summary (if more content follows)
- **Args**: Describe each parameter (with type if not annotated)
- **Returns**: Describe return value (omit for None, generators use "Yields")
- **Raises**: Document exceptions that callers should handle
- **Yields**: For generators, document yielded values

**Class Docstrings**
```python
class SampleClass:
    """Summary of class here.

    Longer class information describing what the class represents
    (not that it "is a class").

    Attributes:
        likes_spam: A boolean indicating spam preference.
        eggs: An integer count of eggs we have.
    """

    def __init__(self, likes_spam: bool = False):
        """Initializes the instance.

        Args:
            likes_spam: Defines instance preference.
        """
        self.likes_spam = likes_spam
        self.eggs = 0
```

**Comments**
- Block comments: Full sentences, capitalized, period at end
- Inline comments: 2+ spaces from code, used sparingly
- Tricky code: Comment before the operation
- Non-obvious code: Comment at end of line
- TODO format: `# TODO: bug-reference - Description` or `# TODO(username): Description`
- Keep comments up-to-date with code changes!
- Comments in English unless 120% sure code never read by non-speakers

**Override Methods**
- Use `@override` decorator (from `typing_extensions`) when overriding
- No docstring needed if behavior unchanged
- Add docstring if behavior differs or side effects added

### 5. Type Hints (PEP 484 + Google)

**Basic Rules**
- Strongly encouraged for function signatures
- Use for complex functions, public APIs, when types aren't obvious
- Don't annotate `self` or `cls` (except when needed for proper type info - use `Self`)
- Don't annotate `__init__` return (always `None`)

**Type Hint Style**
```python
def my_method(
    self,
    first_var: int,
    second_var: Foo,
    third_var: Bar | None,
) -> int:
    ...
```

**Modern Syntax (Python 3.10+)**
- Use `|` for unions: `str | None` (not `Optional[str]` or `Union[str, None]`)
- Use built-in types: `list[int]`, `dict[str, int]` (not `List[int]`, `Dict[str, int]`)
- Use `collections.abc` for parameters: `Sequence`, `Mapping` (not concrete types)

**Specific Guidelines**
- Use explicit `X | None` not implicit (`a: str = None` is wrong)
- Specify generic parameters: `list[int]` not bare `list`
- Use `Any` when best type unknown (but prefer `TypeVar` when possible)
- Type aliases: `CapWords` naming, use `TypeAlias` annotation
- Forward references: use `from __future__ import annotations` or string quotes
- Conditional imports: use `if TYPE_CHECKING:` for type-only imports

**Type Variable Naming**
```python
_T = TypeVar("_T")  # Good: leading underscore, descriptive
_P = ParamSpec("_P")  # Good: leading underscore
AddableType = TypeVar("AddableType", int, float, str)  # Good: descriptive
```

### 6. Code Quality & Best Practices

**Implicit False (PEP 8)**
```python
# Good
if not seq:
if foo is None:
if not x:

# Bad
if len(seq) == 0:
if foo == None:
if x == False:
```

**Comparisons**
- Singletons: use `is`/`is not`: `if x is None:`
- Use `is not` rather than `not ... is`
- Don't compare booleans to True/False: `if greeting:` not `if greeting == True:`
- Type checking: use `isinstance(obj, int)` not `type(obj) is int`
- String prefixes/suffixes: use `.startswith()`/`.endswith()` not slicing

**Sequences**
- Use empty sequence truth value: `if seq:` not `if len(seq):`
- Works for strings, lists, tuples

**Exception Handling**
- Never use bare `except:` (catches SystemExit/KeyboardInterrupt!)
- Use specific exceptions: `except ValueError:` not `except Exception:`
- Minimize try block scope (avoid masking bugs)
- Use `finally` for cleanup or prefer context managers
- Exception chaining: `raise X from Y` or `raise X from None`
- Derive from `Exception` not `BaseException`
- Exception names end in `Error` (if they are errors)

**String Formatting**
```python
# Good - Modern (preferred)
x = f'name: {name}; score: {n}'

# Good - Classic
x = 'name: %s; score: %d' % (name, n)
x = 'name: {}; score: {}'.format(name, n)

# Bad - Don't concatenate in loops
employee_table = '<table>'
for last, first in employees:
    employee_table += f'<tr><td>{last}, {first}</td></tr>'  # BAD!

# Good - Use list + join
items = ['<table>']
for last, first in employees:
    items.append(f'<tr><td>{last}, {first}</td></tr>')
employee_table = ''.join(items)
```

**Logging**
```python
# Good - Use %-style (not f-strings!)
logger.info('TensorFlow version: %s', tf.__version__)

# Bad - Don't use f-strings (prevents lazy evaluation)
logger.info(f'TensorFlow version: {tf.__version__}')
```

**Resource Management**
```python
# Good - Always use context managers
with open('file.txt') as f:
    data = f.read()

# Good - For non-context objects
import contextlib
with contextlib.closing(urllib.urlopen("http://...")) as page:
    for line in page:
        print(line)

# Bad - Don't rely on __del__ or manual close
f = open('file.txt')
data = f.read()
f.close()  # May not run if exception occurs!
```

**Function Defaults**
```python
# Good - No mutable defaults
def foo(a, b=None):
    if b is None:
        b = []

# Good - Immutable default
def foo(a, b: Sequence = ()):
    ...

# Bad - Mutable default (shared across calls!)
def foo(a, b=[]):
    ...
```

**Comprehensions**
```python
# Good
result = [mapping_expr for value in iterable if filter_expr]

# Bad - Multiple for clauses
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# If complex, use regular loop
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))
```

**Lambdas & Operators**
```python
# OK for simple cases
sorted_list = sorted(items, key=lambda x: x.name)

# Better - Use operator module
from operator import attrgetter
sorted_list = sorted(items, key=attrgetter('name'))

# Bad - Multi-line lambda
complicated = lambda x: x.filter(something).map(
    another_thing).reduce(final_thing)

# Good - Use def
def complicated(x):
    return x.filter(something).map(another_thing).reduce(final_thing)
```

**Statements**
```python
# Good
if foo == 'blah':
    do_blah_thing()
do_one()
do_two()

# Bad - Compound statements
if foo == 'blah': do_blah_thing()
do_one(); do_two(); do_three()
```

**Return Statements**
- Be consistent: all return expressions or all return None
- Explicit is better: use `return None` not bare `return` if other returns have values

**Properties**
- Use `@property` decorator (not manual descriptors)
- Only for trivial computations (cheap, straightforward)
- Don't use for expensive operations or complex logic
- Don't use just to wrap simple attribute access (make it public)

**Decorators**
- Use judiciously when clear advantage
- Never use `@staticmethod` (use module-level function - Google)
- Use `@classmethod` sparingly (named constructors, class-specific state)

**Global State**
- Avoid mutable global state
- Module-level constants OK: `MAX_TIMEOUT = 30`
- Name private globals with leading underscore: `_internal_cache`

**Power Features (Avoid)**
- No custom metaclasses
- No bytecode manipulation
- No `__del__` for cleanup
- No reflection hacks (some `getattr` use is OK)
- No import hacks
- Standard library can use these (e.g., `abc.ABCMeta`, `dataclasses`, `enum`)

### 7. Security

**SQL Injection**
```python
# Bad - SQL injection risk!
query = f"SELECT * FROM users WHERE id = {user_id}"
query = "SELECT * FROM users WHERE id = " + user_id

# Good - Use parameterized queries
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

**Input Validation**
- Validate all external input
- Use allowlists not denylists
- Sanitize before using in system commands

**Hardcoded Secrets**
- Never hardcode passwords, API keys, tokens
- Use environment variables or secret management
- Check for: `password = "..."`, `api_key = "..."`, etc.

**Unsafe Functions**
- Avoid: `eval()`, `exec()`, `compile()`, `__import__()`
- Be careful with: `pickle`, `yaml.load()` (use `safe_load`)

### 8. Performance

**String Concatenation**
- Never use `+` or `+=` in loops (quadratic time!)
- Use `''.join(items)` or `io.StringIO`

**Generators**
- Use generators for large sequences (memory efficient)
- Use comprehensions over `map()`/`filter()` with lambda

**Default Iterators**
```python
# Good
for key in adict:
for line in afile:
for k, v in adict.items():

# Bad
for key in adict.keys():
for line in afile.readlines():
```

### 9. Maintainability

**Function Length**
- Prefer < 40 lines (Google guideline, not hard limit)
- Break up long functions unless it harms structure
- If >40 lines, consider if it can be split

**Main Guard**
```python
# Good
def main():
    ...

if __name__ == '__main__':
    main()

# Or with absl
from absl import app

def main(argv):
    ...

if __name__ == '__main__':
    app.run(main)
```

**Assertions**
- Don't use `assert` for critical logic (can be disabled with `-O`)
- OK for validating test expectations
- Use `if` + `raise` for preconditions

## Output Format

Structure your review as:

### Summary
- **Overall Assessment**: Excellent/Good/Fair/Needs Improvement
- **PEP 8 Compliance**: High/Medium/Low
- **Google Style Compliance**: High/Medium/Low
- **Key Strengths**: 2-4 well-implemented aspects
- **Critical Issues**: Issues requiring immediate attention (if any)

### Detailed Findings

Group by category. For each issue:

**[Category: Style/Documentation/Quality/Security/Performance/Maintainability]**

**Issue #**: Brief title
- **Severity**: Critical/High/Medium/Low
- **Lines**: Specific line numbers
- **PEP 8/Google Reference**: Section reference (if applicable)
- **Description**: Clear explanation of the issue
- **Current Code**:
  ```python
  # Problematic code excerpt
  ```
- **Recommended Fix**:
  ```python
  # Corrected code
  ```
- **Rationale**: Why this matters (readability/safety/performance/maintainability)

### Positive Highlights
- Well-implemented patterns worth noting
- Good adherence to standards
- Exemplary practices

### Recommendations
- Priority-ordered list of improvements
- Consider quick wins vs. larger refactors
- Balance consistency with practical constraints

### References
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 257 Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 Type Hints](https://peps.python.org/pep-0484/)

## Enforcement Tools

**Recommended**:
- **pylint**: Comprehensive linter ([Google's pylintrc](https://google.github.io/styleguide/pylintrc))
- **pytype**: Type checker (Google's tool)
- **mypy**: Alternative type checker
- **Black** or **Pyink**: Auto-formatters (Google uses these)
- **flake8**: Alternative linter
- **isort**: Import sorting

**Suppression**:
- Use `# pylint: disable=rule-name` with explanation
- Use `# type: ignore` for type checking (sparingly)
- Document why suppression is needed

## Review Guidelines

**Be Constructive**
- Explain *why* something matters
- Provide specific, actionable recommendations
- Include code examples for fixes
- Acknowledge good practices

**Context Matters**
- Consider project conventions
- Match surrounding code style when editing
- Balance improvement with backwards compatibility
- Know when rules have valid exceptions

**Prioritize**
1. **Critical**: Security issues, correctness bugs
2. **High**: Significant readability/maintainability issues
3. **Medium**: Style violations, minor best practices
4. **Low**: Nitpicks, suggestions

**Pragmatic Approach**
- Focus on changes being made (not rewriting entire codebase)
- Suggest incremental improvements
- Consider team capacity and priorities
- "Perfect is the enemy of good"

## Special Cases

**Legacy Code**
- Focus on new/modified code
- Don't require full refactor to meet standards
- Suggest incremental modernization

**Mathematical/Scientific Code**
- Short variable names OK if match notation: `i`, `j`, `x`, `y`
- Reference paper/algorithm in comments
- Use `# pylint: disable=invalid-name` if needed

**Test Files**
- PEP 8 compliant names: `test_<method>_<state>`
- Or legacy style: `testMethodUnderTest_state`
- Less strict docstring requirements

**Backwards Compatibility**
- Don't break compatibility just to comply with PEP 8
- Consider deprecation path for API changes

---

This skill combines the authoritative standards of PEP 8 with Google's battle-tested practices to provide comprehensive, practical Python code reviews.
