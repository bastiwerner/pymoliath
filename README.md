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

Static type checking is enforced using [pyright](https://microsoft.github.io/pyright/), configured via `pyrightconfig.json`.

```bash
uv run --no-sync pyright
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
| `Option`         | Optional value, Rust-style (`Some`/`None`)                       | [option](./pymoliath/option.py)             |
| `Either`         | Two-track value, Haskell-style (`Left`/`Right`)                 | [either](./pymoliath/either.py)             |
| `Result`         | Two-track value, Rust-style (`Ok`/`Err`)                        | [result](./pymoliath/result.py)             |
| `Try`            | Computation that might raise, Scala-style (`Success`/`Failure`) | [try](./pymoliath/try.py)                   |
| `IO`             | Deferred side-effecting computation                             | [io](./pymoliath/io.py)                     |
| `ListMonad`      | Eager list with Rust `Iterator`-inspired combinators            | [list](./pymoliath/list.py)                 |
| `Sequence`       | Lazily-evaluated counterpart of `ListMonad`                     | [lazy](./pymoliath/lazy.py#sequence.py)     |
| `Reader`         | Read-only shared environment                                    | [reader](./pymoliath/reader.py)             |
| `Writer`         | Computation with an accumulated log/monoid                      | [writer](./pymoliath/writer.py)             |
| `State`          | Computation that threads state through                          | [state](./pymoliath/state.py)               |
| `LazyMonad`      | Deferred, memoized computation                                  | [lazy](./pymoliath/lazy/README.md.py)       |
| `Continuation`   | Continuation-passing style (CPS) computation                    | [continuation](./pymoliath/continuation.py) |
| `pymoliath.util` | FP prelude helpers (`compose`, `curry`, `pipe`, ...)            | [util](./pymoliath/util.py)                 |
