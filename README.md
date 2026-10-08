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
Rust-like, and a `match` over the variants is exhaustive. In the two-track types (`Result`, `Either`)
both variants carry both type parameters; `Nil` and `Nothing` carry none (they are `Never`-typed
singletons). Lambdas passed to `map`/`bind`/... are inferred from the receiver:

```python
def parse(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err("not a number")


parse("4").bind(lambda x: Ok(x / 2) if x else Err("zero"))  # Result[float, str]
```

The type parameters are covariant and the side a variant does not use is `Never`: a bare `Ok(10)`
is an `Ok[int, Never]`, a bare `Err("e")` an `Err[Never, str]`, a bare `Right(10)` a
`Right[Never, int]` (both `Left` and `Right` take `[L, R]`, like `Either`), and `Nil()`/`Nothing()`
are singletons of type `Option[Never]`/`Maybe[Never]`. All of them are assignable to a matching
`Result[int, str]`, `Either[str, int]` or `Option[int]` without annotations.

Like in Rust, the receiver fixes the types a method accepts. `bind` on a `Result[int, str]` needs a
function returning a `Result[U, str]`, and a bare `Ok(10)` (error type `Never`) has to be annotated
before it can be bound to fallible code:

```python
value: Result[int, str] = Ok(10)
value.bind(parse_more)  # OK
Ok(10).bind(parse_more)  # type error: the error type of a bare Ok is Never
```

Recovering from an error or empty value works with any type, so `Err("e").unwrap_or(10)`,
`Left("e").or_(Right(1))` and `Nil().unwrap_or(10)` need no annotation.

Further notes:

- **Values:** the variants are frozen, slotted dataclasses with value equality and hashing (`{Ok(1),
  Ok(1)}` has one element, and an unhashable payload makes the container unhashable). `Failure`
  compares exceptions by type and `args`. `repr` shows the payload's `repr` (`Ok('1')`), and `str` stays
  human-readable (`Ok(1)`).
- **`unwrap`** on `Err`/`Left`/`Nil`/`Nothing` raises `pymoliath.errors.UnwrapError`, which keeps the
  container in `.container` and chains a wrapped exception as `__cause__`. `Failure.unwrap()` re-raises
  the original exception.
- **Narrowing:** `isinstance` narrows a value to a variant, in an `if`/`else` or after an early exit,
  just like `match`:

  ```python
  def describe(text: str) -> str:
      result = parse(text)
      if isinstance(result, Err):
          return result.error
      return str(result.value)  # result is an Ok[int, str] here, so .value is an int
  ```

  The predicate methods (`is_ok`/`is_err`, `is_some`/`is_nil`, ...) return a plain `bool`; a method
  can't narrow its own receiver, so use `isinstance` when the type checker should know the variant.
  The aliases (`Result`, `Option`, ...) are not classes, so check "any Result" with
  `isinstance(x, (Ok, Err))`.
- **`match`** takes keyword-only callbacks: `result.match(ok=..., err=...)`, `either.match(left=...,
  right=...)`, `option.match(some=..., nil=...)`, `maybe.match(just=..., nothing=...)`,
  `try_.match(success=..., failure=...)`.
- **Applicatives:** `value.apply(function)` and its mirror `function.apply2(value)` take a wrapped
  one-argument function. Functions of several arguments are curried and applied argument by argument
  (`Ok(add).apply2(x).apply2(y)`). When both sides are errors, the function side wins.
- **Parameter names** are uniform: predicates are `predicate`, callbacks are `function` (and
  `default_function` for the fallback of `map_or_else`), default values are `default_value`, and
  error values are `error` on the Rust-style side (`Result`, `Option`) and `left_value` on the
  Haskell-style side (`Either`, `Maybe`).
- **Constructors** are static methods, available on either variant: `Ok.safe(f)`/`Ok.from_option(o,
  error)`, `Right.safe(f)`/`Right.from_maybe(m, left)`, `Some.safe(f)`/`Some.from_optional(x)`,
  `Just.safe(f)`/`Just.from_optional(x)` and `Success.safe(f)`. `safe` catches `Exception` by
  default; pass `exceptions=(ValueError,)` to narrow both what is caught and the error type.

## Async

Every monad except `ListMonad`, `Sequence`, `LazyMonad` and `Continuation` has an async counterpart in
`pymoliath.aio` (`AsyncResult`, `AsyncEither`, `AsyncOption`, `AsyncMaybe`, `AsyncTry`, `AsyncIO`,
`AsyncReader`, `AsyncState`, `AsyncWriter`). An `Async*` value wraps a deferred computation: `map`,
`bind`, ... only build the pipeline, and awaiting it runs the pipeline and resolves to the sync monad
(`Ok`/`Err`, `Some`/`Nil`, ...). The callbacks may be plain or `async` functions.

```python
from pymoliath.aio import AsyncResult


async def main() -> None:
    result = await AsyncResult.from_ok(4).map(lambda x: x / 2)  # Ok(2.0)
```

The sum types have one constructor per variant, named after it: `AsyncResult.from_ok`/`from_err`,
`AsyncEither.from_right`/`from_left`, `AsyncOption.from_some`/`from_nil`,
`AsyncMaybe.from_just`/`from_nothing` and `AsyncTry.from_success`/`from_failure`. Each type also has a
constructor from the sync monad (`from_result`, `from_either`, `from_option`, `from_maybe`, `from_try`)
and one from an async callable (`from_coroutine`). `from_err`, `from_left` and `from_failure` leave the
success type open, so annotate the target, as with a bare `Err`:

```python
failed: AsyncResult[int, str] = AsyncResult.from_err("not found")
```

The async classes wrap callables, so unlike the sync types they are invariant.

# Testing

Execute the test suite using `pytest` to ensure correctness across all modules:

```bash
uv run --no-sync pytest
```

## Documentation Tests

Verify that all docstrings and examples are valid:

```bash
uv run --no-sync pytest --doctest-modules
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
`pyproject.toml`, covering the whole package). `test/test_typing.py` holds the `assert_type`
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
uv run --no-sync pdoc -t .pdoc/rust pymoliath
```

# Components

| Component        | Description                                                     | Docs                                            |
| ---------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| `Maybe`          | Optional value, Haskell-style (`Just`/`Nothing`)                | [maybe](./pymoliath/maybe.py)                   |
| `Option`         | Optional value, Rust-style (`Some`/`Nil`)                       | [option](./pymoliath/option.py)                 |
| `Either`         | Two-track value, Haskell-style (`Left`/`Right`)                 | [either](./pymoliath/either.py)                 |
| `Result`         | Two-track value, Rust-style (`Ok`/`Err`)                        | [result](./pymoliath/result.py)                 |
| `Try`            | Computation that might raise, Scala-style (`Success`/`Failure`) | [exception](./pymoliath/exception.py)           |
| `IO`             | Deferred side-effecting computation                             | [io](./pymoliath/io.py)                         |
| `ListMonad`      | Eager list with Rust `Iterator`-inspired combinators            | [list](./pymoliath/list.py)                     |
| `Sequence`       | Lazily-evaluated counterpart of `ListMonad`                     | [lazy](./pymoliath/lazy.py)                     |
| `Reader`         | Read-only shared environment                                    | [reader](./pymoliath/reader.py)                 |
| `Writer`         | Computation with an accumulated log/monoid                      | [writer](./pymoliath/writer.py)                 |
| `State`          | Computation that threads state through                          | [state](./pymoliath/state.py)                   |
| `LazyMonad`      | Deferred, memoized computation                                  | [lazy](./pymoliath/lazy.py)                     |
| `Continuation`   | Continuation-passing style (CPS) computation                    | [continuation](./pymoliath/continuation.py)     |
| `AsyncResult`    | Async `Result` (resolves to `Ok`/`Err`)                         | [async_result](./pymoliath/aio/async_result.py) |
| `AsyncEither`    | Async `Either` (resolves to `Left`/`Right`)                     | [async_either](./pymoliath/aio/async_either.py) |
| `AsyncOption`    | Async `Option` (resolves to `Some`/`Nil`)                       | [async_option](./pymoliath/aio/async_option.py) |
| `AsyncMaybe`     | Async `Maybe` (resolves to `Just`/`Nothing`)                    | [async_maybe](./pymoliath/aio/async_maybe.py)   |
| `AsyncTry`       | Async `Try` (resolves to `Success`/`Failure`)                   | [async_try](./pymoliath/aio/async_try.py)       |
| `AsyncIO`        | Async `IO`                                                      | [async_io](./pymoliath/aio/async_io.py)         |
| `AsyncReader`    | Async `Reader`                                                  | [async_reader](./pymoliath/aio/async_reader.py) |
| `AsyncState`     | Async `State`                                                   | [async_state](./pymoliath/aio/async_state.py)   |
| `AsyncWriter`    | Async `Writer`                                                  | [async_writer](./pymoliath/aio/async_writer.py) |
| `pymoliath.util` | FP prelude helpers (`compose`, `curry`, `pipe`, ...)            | [util](./pymoliath/util.py)                     |
| `UnwrapError`    | Raised by `unwrap` on `Err`/`Left`/`Nil`/`Nothing`              | [errors](./pymoliath/errors.py)                 |
