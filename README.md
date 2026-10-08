# Pymoliath

A robust Python library for Monadic Functional Programming.

# About

Pymoliath provides a comprehensive suite of Monads for functional programming in Python. Inspired by Haskell's type system and established functional programming principles, it aims to provide type-safe, composable abstractions for side effects, state management, and error handling.

This project draws inspiration from:
- [Pymonad](https://github.com/jasondelaat/pymonad) by jasondelaat
- [OSlash](https://github.com/dbrattli/OSlash) by dbrattli
- [typesafe-monads](https://github.com/correl/typesafe-monads) by correl
- [Result](https://doc.rust-lang.org/std/result/index.html) from the Rust standard library

# Getting Started

Pymoliath uses [uv](https://docs.astral.sh/uv/) for high-performance dependency management and project orchestration. To set up the environment and install all dependencies (including development tools), run:

```bash
uv sync
```

> This command initializes a virtual environment and ensures all required packages are installed.

Pymoliath requires Python 3.12 or newer.

## Typing

The sum types (`Result`, `Either`, `Option`, `Maybe`, `Try`) are sealed (`@final` variants) and
Rust-like: both variants carry all type parameters, and a `match` over the variants is exhaustive.
Lambdas passed to `map`/`bind`/... are inferred from the receiver:

```python
def parse(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err("not a number")


parse("4").bind(lambda x: Ok(x / 2) if x else Err("zero"))  # Result[float, str]
```

The type parameters are covariant and the side a variant does not use is `Never`: a bare `Ok(10)`
is an `Ok[int, Never]`, a bare `Err("e")` an `Err[Never, str]`, and `Nil()`/`Nothing()` are
singletons of type `Option[Never]`/`Maybe[Never]`. All of them are assignable to a matching
`Result[int, str]` or `Option[int]` without annotations.

Like in Rust, the receiver fixes the types a method accepts. `bind` on a `Result[int, str]` needs a
function returning a `Result[U, str]`, and a bare `Ok(10)` (error type `Never`) has to be annotated
before it can be bound to fallible code:

```python
value: Result[int, str] = Ok(10)
value.bind(parse_more)  # OK
Ok(10).bind(parse_more)  # type error: the error type of a bare Ok is Never
```

Further notes:

- **Values:** the variants are frozen, slotted dataclasses with value equality and hashing (`{Ok(1),
  Ok(1)}` has one element, and an unhashable payload makes the container unhashable). `Failure`
  compares exceptions by type and `args`. `repr` shows the payload's `repr` (`Ok('1')`), and `str` stays
  human-readable (`Ok(1)`).
- **`unwrap`** on `Err`/`Left`/`Nil`/`Nothing` raises `pymoliath.errors.UnwrapError`, which keeps the
  container in `.container` and chains a wrapped exception as `__cause__`. `Failure.unwrap()` re-raises
  the original exception.
- **`match`** takes keyword-only callbacks: `result.match(ok=..., err=...)`, `either.match(left=...,
  right=...)`, `option.match(some=..., nil=...)`, `maybe.match(just=..., nothing=...)`,
  `try_.match(success=..., failure=...)`.
- **Applicatives:** `value.apply(function)` and its mirror `function.apply2(value)` take a wrapped
  one-argument function; curried functions can be applied argument by argument
  (`Ok(add).apply2(x).apply2(y)`). For uncurried functions of several arguments use the module-level
  `map2`/`map3`. When several values are errors, the function side (or the first argument) wins.
- **Narrowing:** the aliases (`Result`, `Option`, ...) are not classes, so use `is_result(x)` or
  `isinstance(x, RESULT_TYPES)` at runtime, and `is_ok`/`is_err`/`is_some`/... (which return `TypeIs`)
  to narrow a value to a variant.
- **`safe` helpers** (`result_safe`, `either_safe`, `option.safe`, `maybe.safe`, `exception.safe`)
  catch `Exception` by default; pass `exceptions=(ValueError,)` to narrow both what is caught and the
  error type.

# Testing

Execute the test suite using `pytest` to ensure correctness across all modules:

```bash
uv run pytest
```

## Documentation Tests

Verify that all docstrings and examples are valid:

```bash
uv run pytest --doctest-modules
```

# Linting & Formatting

Pymoliath maintains high code quality standards using [ruff](https://docs.astral.sh/ruff/).

```bash
uv run --no-sync ruff check          # Run linting
uv run --no-sync ruff check --fix    # Run linting and apply safe fixes
uv run --no-sync ruff format --check # Check formatting without making changes
uv run --no-sync ruff format         # Apply formatting changes
```

# Type Checking

Static type checking is enforced using [pyright](https://microsoft.github.io/pyright/) (configured via
`pyrightconfig.json`) and [mypy](https://mypy.readthedocs.io/) in strict mode (configured in
`pyproject.toml`, covering the sum-type modules). `test/test_typing.py` holds the `assert_type`
regression tests for pyright.

```bash
uv run --no-sync pyright
uv run --no-sync mypy
```

# Benchmarks

Micro-benchmarks for the Result Monad use [pyperf](https://pyperf.readthedocs.io/):

```bash
uv run --no-sync python bench/bench_result.py
```

# Documentation

Documentation is automatically generated using [pdoc](https://pdoc.dev/docs/pdoc.html) with a custom theme inspired by Rust's Cargo.

```bash
uv run pdoc -t .pdoc/rust pymoliath
```

# Components

| Component        | Description                                                     | Docs                                        |
| ---------------- | --------------------------------------------------------------- | ------------------------------------------- |
| `Maybe`          | Optional value, Haskell-style (`Just`/`Nothing`)                | [maybe](./pymoliath/maybe.py)               |
| `Option`         | Optional value, Rust-style (`Some`/`Nil`)                       | [option](./pymoliath/option.py)             |
| `Either`         | Two-track value, Haskell-style (`Left`/`Right`)                 | [either](./pymoliath/either.py)             |
| `Result`         | Two-track value, Rust-style (`Ok`/`Err`)                        | [result](./pymoliath/result.py)             |
| `Try`            | Computation that might raise, Scala-style (`Success`/`Failure`) | [exception](./pymoliath/exception.py)       |
| `IO`             | Deferred side-effecting computation                             | [io](./pymoliath/io.py)                     |
| `ListMonad`      | Eager list with Rust `Iterator`-inspired combinators            | [list](./pymoliath/list.py)                 |
| `Sequence`       | Lazily-evaluated counterpart of `ListMonad`                     | [lazy](./pymoliath/lazy.py#sequence.py)     |
| `Reader`         | Read-only shared environment                                    | [reader](./pymoliath/reader.py)             |
| `Writer`         | Computation with an accumulated log/monoid                      | [writer](./pymoliath/writer.py)             |
| `State`          | Computation that threads state through                          | [state](./pymoliath/state.py)               |
| `LazyMonad`      | Deferred, memoized computation                                  | [lazy](./pymoliath/lazy/README.md.py)       |
| `Continuation`   | Continuation-passing style (CPS) computation                    | [continuation](./pymoliath/continuation.py) |
| `pymoliath.util` | FP prelude helpers (`compose`, `curry`, `pipe`, ...)            | [util](./pymoliath/util.py)                 |
| `UnwrapError`    | Raised by `unwrap` on `Err`/`Left`/`Nil`/`Nothing`               | [errors](./pymoliath/errors.py)             |
