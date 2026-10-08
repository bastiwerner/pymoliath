# Changelog

## 3.0.0

A hardening release for the sum types (`Result`, `Either`, `Option`, `Maybe`, `Try`) and their async
counterparts: value semantics instead of string comparison, covariant typing, a faster
per-variant implementation, and a uniform API. It contains many breaking changes; see below.

### Breaking changes

**Typing**

- The type parameters are covariant, and the side a variant does not use defaults to `Never`. A bare
  `Ok(1)` is an `Ok[int, Never]` and a bare `Err("e")` an `Err[Never, str]`; both are assignable to a
  `Result[int, str]`. Like in Rust, `bind` still requires the receiver's error type, so a bare `Ok(10)`
  has to be annotated before it can be bound to fallible code.
- `Right` takes its parameters in the same order as `Left` and `Either`: `Right[L, R]` (was
  `Right[R, L]`). A bare `Right(10)` is a `Right[Never, int]`.
- `Nil` and `Nothing` are non-generic singletons of type `Option[Never]` / `Maybe[Never]`:
  `type Option[T] = Some[T] | Nil`. `Nil[int]()` no longer works; write `Nil()`.

**Values**

- Equality and hashing compare the wrapped values (frozen, slotted dataclasses) instead of
  `type` + `str()`. For example, `Ok(Decimal("1.0")) == Ok(Decimal("1.00"))`, distinct NaNs are unequal,
  and an unhashable payload makes the container unhashable. Instances are hashable (previously
  `__hash__ = None`).
- `Failure` compares and hashes its exception by `(type(exception), exception.args)`. `Err`/`Left`
  holding exceptions compare them by identity.
- `repr` shows the payload's `repr` (`Ok('1')` vs `Ok(1)`, `Failure(ValueError('boom'))`). `str` is
  unchanged.
- `Failure(...)` raises `TypeError` (instead of failing an `assert`) when given a non-exception.

**Methods**

- `unwrap` on `Err`, `Left`, `Nil` and `Nothing` raises the new `pymoliath.errors.UnwrapError`
  (instead of a bare `Exception`). It keeps the container in `.container` and chains a wrapped
  exception as `__cause__`. `Failure.unwrap()` still re-raises the original exception.
- `match` takes keyword-only callbacks: `match(*, ok=, err=)`, `match(*, left=, right=)`,
  `match(*, some=, nil=)`, `match(*, just=, nothing=)`, `match(*, success=, failure=)`.
- `apply` takes a wrapped one-argument function; it no longer auto-curries multi-argument functions.
  `apply2` is kept but typed the same way (`function.apply2(value)`), so functions of several
  arguments are curried by hand and applied one argument at a time
  (`Ok(lambda a: lambda b: a + b).apply2(x).apply2(y)`). When both sides fail, the function side
  wins. This applies to the `Async*` sum types too.
- `Option.is_nothing()` is renamed to `Option.is_nil()`, matching `Nil`.
- Parameter names are uniform:
  - `filter(predicate)` (was `filter_function`);
  - the callbacks of `unwrap_or_else`, `ok_or_else` and `right_or_else` are `function` (was
    `err_function` / `left_function` / `nothing_function` / `failure_function`);
  - error values are `error` on the Rust-style side (`Result.filter`, `Result.from_option`,
    `Option.ok_or`) and `left_value` on the Haskell-style side (`Either.filter`,
    `Either.from_maybe`, `Maybe.right_or`).

**Constructors**

- `result_safe` / `either_safe` and the module-level `safe` / `from_optional` functions are replaced
  by static methods on the classes, reachable through either variant: `Ok.safe`, `Right.safe`,
  `Some.safe`, `Just.safe`, `Success.safe`, `Some.from_optional`, `Just.from_optional`. They are joined
  by the new `Ok.from_option` and `Right.from_maybe`.
- `AsyncOption.from_value` is renamed to `from_some` and `AsyncMaybe.from_value` to `from_just`,
  giving every async sum type one constructor per variant.

**Removed helpers**

- The module-level `map2` / `map3` functions (sync and async), the `TypeIs` functions (`is_ok`,
  `is_err`, `is_some`, `is_nil`, `is_just`, `is_nothing`, `is_left`, `is_right`, `is_success`,
  `is_failure`, `is_result`, `is_option`, ...) and the `*_TYPES` tuples. Narrow with `isinstance`
  (`isinstance(result, Ok)`) or `match`; both narrow in pyright and mypy.

**util**

- `curry` is fully curried (`curry(f)(1)(2)(3)`); previously it only applied one level.
- `compose` unpacks a tuple into functions that take two or more positional parameters; one-parameter
  and `*args` functions receive the tuple as one argument.

### New features

- Covariant typing: bare variants are assignable without annotations, and conditional lambdas such
  as `lambda x: Ok(x) if x else Err("e")` are inferred as `Result[int, str]`.
- Recovering from an error or empty value works with any type: `Err("e").unwrap_or(10)`,
  `Left("e").or_(Right(1))`, `Failure(e).unwrap_or_else(...)` and `Nil().unwrap_or(10)` need no
  annotation.
- Methods called directly on a variant keep the precise type (`Ok(1).map(str)` is an
  `Ok[str, Never]`).
- New methods:
  - all two-track types: `swap`, `map_or_else`, `filter`, `merge`, `transpose`, and `and_then` /
    `or_else` aliases of `bind` / `bind_err` (`bind_left`, `bind_failure`);
  - `Option` / `Maybe`: `map_or_else`, `xor`, `transpose`, `and_then`, `or_else`;
  - `Try`: `map_or_else`, `filter` (a failing predicate yields a `Failure(ValueError(...))`),
    `and_then`, `or_else`.
- `safe(..., exceptions=(ValueError,))` narrows both what is caught and the error type; other
  exceptions propagate.
- Async sum types: `from_nil` / `from_nothing` constructors, `and_then` / `or_else` aliases,
  `filter` on `AsyncResult` / `AsyncEither` / `AsyncTry`, and a `run()` coroutine method (also on
  `AsyncIO` / `AsyncWriter`) for use with `asyncio.run`.
- `pymoliath.errors.UnwrapError`, exported from the package, as is `Continuation`.
- The package ships a `py.typed` marker (PEP 561), so type checkers use its inline types.

### Fixes

- Equality no longer goes through `str()`, which made different values equal (custom `__str__`, NaN)
  and equal values unequal (`Decimal("1.0")` vs `Decimal("1.00")`, reordered dicts).
- `unwrap` keeps the original exception type reachable (`__cause__`) instead of wrapping it in a
  generic `Exception`.
- `apply` / `apply2` no longer go through `cast(Any)` and `Callable[..., U]`, so wrong argument types
  are reported.
- `util.compose` and `util.curry` no longer catch `TypeError`s raised inside the user's functions
  (which could re-run a function and hide the original error).
- `AsyncWriter.bind` accepts callbacks returning a plain `(value, monoid)` tuple again.
- The docstring examples run on Python 3.12 and 3.13 (`asyncio.run` before 3.14 only accepts
  coroutines).

### Performance

Benchmarks (`bench/bench_result.py`, pyperf, Python 3.14) against 2.0.0:

| Benchmark                          | 2.0.0   | 3.0.0   |
| ---------------------------------- | ------- | ------- |
| 10-step `map`/`bind` chain on `Ok` | 3.99 µs | 1.48 µs |
| 10-step `map`/`bind` chain on `Err`| 4.01 µs | 289 ns  |
| `==` on a 10,000-element payload   | 426 µs  | 6.56 µs |

- The behaviour lives in the variants, so methods no longer re-dispatch through a `match`, and
  short-circuit paths return the instance itself instead of allocating.
- `Nil()` / `Nothing()` are cached singletons on the hot paths.
- The variants set their field directly in a hand-written `__init__` instead of the frozen
  dataclass one (about 30% faster construction; still frozen).
- `compose` reads each function's signature once, when composing; `curry` reads a plain function's
  arity straight from its code object.
- Instance size is unchanged (40 bytes, no `__dict__`).

### Tooling

- mypy in strict mode covers the whole package (`uv run --no-sync mypy`), next to pyright strict.
- `test/test_typing.py` holds `assert_type` regression tests, including negative cases.
- `bench/bench_result.py` (pyperf) and pyright coverage of `bench/`.
