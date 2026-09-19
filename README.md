# Pymoliath

Python Monads library for Functional Programming

# About

Pymoliath is a python library containing Monads for monadic functional programming. This library is inspired by Haskell
as well as existing Monad implementations in python.

* [Pymonad](https://github.com/jasondelaat/pymonad) by jasondelaat
* [OSlash](https://github.com/dbrattli/OSlash) by dbrattli
* [typesafe-monads](https://github.com/correl/typesafe-monads) by correl
* [std::result::Result](https://doc.rust-lang.org/std/result/enum.Result.html#) of rustlang

# Getting started

The pymoliath repository uses [uv](https://docs.astral.sh/uv/) as dependency management tool. The project can be
installed using the following command.

```
uv sync
```

> uv will create a virtual environment and install all dependencies, including the dev dependency group.

# Test

Run pymoliath tests using pytest. This command will execute all unittest in the virtual environment.

```
uv run pytest
```

# Lint & Format

Pymoliath uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```
uv run --no-sync ruff check          # lint
uv run --no-sync ruff check --fix    # lint, applying safe fixes
uv run --no-sync ruff format --check # format check (no changes made)
uv run --no-sync ruff format         # format, applying changes
```

# Type Checking

Pymoliath uses [pyright](https://microsoft.github.io/pyright/) for static type checking, configured in
`pyrightconfig.json`.

```
uv run --no-sync pyright
```

# Monads

Every Monad implementation of Pymoliath has the typical haskell Monad interface.

* `map` (`>>`):
    * Monad(a).map(f a -> b) => Monad(b)
* `bind` (`>>=`):
    * Monad(a).bind(f a -> Monad(b)) => Monad(b)
* `apply` (`<*>`):
    * Monad(a).apply(Monad(f a -> b)) => Monad(b)
* `apply2` (`<*>`):
    * Monad(f a -> b).apply2(Monad(a)) => Monad(b)
* `run`: (IO, Reader, Writer, State, Lazy, Sequence)

`ListMonad` and `Sequence` additionally expose a larger set of Rust `Iterator`-inspired combinators
(`fold`, `zip`, `find`, `partition`, `sort_by`, ...) beyond this core Monad interface - see their sections below.

`Maybe`, `Option`, `Either`, `Result`, `Try`, `IO`, `Writer`, `Reader` and `State` additionally have a directly
awaitable `AsyncX` counterpart for use in async code - see [Async Monads](#async-monads) below.

# Structural Pattern Matching

`Maybe`, `Either`, `Result`, `Try` and `Option` are modeled as closed union types (e.g. `Maybe = Just[T] | Nothing`),
so they work directly with Python's `match` statement — including exhaustiveness checking by static type checkers
like pyright, without needing a `case _:` fallback.

```python
match maybe_value:
    case Just(x):
        print(f"just {x}")
    case Nothing():
        print("nothing")
```

The same works for `Left`/`Right`, `Ok`/`Err`, `Success`/`Failure` and `Some`/`Nil`.

## Maybe

Haskell : [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

```python
# Maybe[TypeSource]
just: Maybe[int] = Just(10)
nothing: Maybe[int] = Nothing()
applicative = Just(lambda x: x + 1)

just.is_just()
just.is_nothing()

# Monad functions
just.map(mapping_function)  # Maybe maps the function (if not nothing)
just.bind(bind_function)  # Maybe binds the function (if not nothing)
just.apply(applicative)  # Apply the just value to the applicative (if not nothing)
applicative.apply2(just)  # Apply the applicative to the just value (if not nothing)

just.unwrap()  # Returns the just value or raises an Exception
just.unwrap_or(20)  # Returns the just value or else a default value
just.unwrap_or_else(
    lambda: 20
)  # Return the just value or the result from the passed fucntion

just.filter(
    filter_function
)  # Just if filter_function(value) is True, otherwise Nothing
just.match(just_function, default_function)

just.is_just_and(filter_function)  # True if Just and filter_function(value) is True
just.map_or(0, mapping_function)  # Maps the function, or returns the default if Nothing
just.and_(Just(20))  # Just(20) if just is a Just, otherwise Nothing
just.or_(Just(20))  # just if it is a Just, otherwise Just(20)
just.zip(Just("a"))  # Just((value, "a")), or Nothing if either side is Nothing
Just(Just(10)).flatten()  # Just(10)

just.right_or("error")  # Right(value), or Left("error") if nothing
just.right_or_else(lambda: "error")  # Right(value), or Left(error_function()) if nothing

from_optional(value_or_none)  # from pymoliath.maybe import from_optional
```

## Option

An alternative to `Maybe` with Rust-style naming (`Option`/`Some`/`Nil` instead of `Maybe`/`Just`/`Nothing`). Same
interface otherwise.

```python
# Option[TypeSource]
some: Option[int] = Some(10)
nil: Option[int] = Nil()
applicative = Some(lambda x: x + 1)

some.is_some()
some.is_nothing()

# Monad functions
some.map(mapping_function)  # Option maps the function (if not nil)
some.bind(bind_function)  # Option binds the function (if not nil)
some.apply(applicative)  # Apply the some value to the applicative (if not nil)
applicative.apply2(some)  # Apply the applicative to the some value (if not nil)

some.unwrap()  # Returns the some value or raises an Exception
some.unwrap_or(20)  # Returns the some value or else a default value
some.unwrap_or_else(
    lambda: 20
)  # Return the some value or the result from the passed function

some.filter(filter_function)  # Some if filter_function(value) is True, otherwise Nil
some.match(some_function, default_function)

some.is_some_and(filter_function)  # True if Some and filter_function(value) is True
some.map_or(0, mapping_function)  # Maps the function, or returns the default if Nil
some.and_(Some(20))  # Some(20) if some is a Some, otherwise Nil
some.or_(Some(20))  # some if it is a Some, otherwise Some(20)
some.zip(Some("a"))  # Some((value, "a")), or Nil if either side is Nil
Some(Some(10)).flatten()  # Some(10)

some.ok_or("error")  # Ok(value), or Err("error") if nil
some.ok_or_else(lambda: "error")  # Ok(value), or Err(error_function()) if nil

from_optional(value_or_none)  # from pymoliath.option import from_optional
```

## Either

Haskell : [Data.Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)

```python
# Either[TypeLeft, TypeRight]
right: Either[str, int] = Right(10)
left: Either[str, int] = Left("error")
applicative = Right(lambda x: x + 1)

right.is_right()
right.is_left()

# Monad functions
right.map(right_map_function)  # Maps only the right value with the function
right.map_left(left_map_function)  # Maps only the left value with the function
right.bind(right_bind_function)  # Binds only the right value with the function
right.bind_left(left_bind_function)  # Binds only the left value with the function
right.apply(applicative)  # Apply the right value to the applicative (if not left)
applicative.apply2(right)  # Apply the applicative to the right value (if not left)

right.unwrap()  # Returns the right value or raises an Exception.
right.unwrap_or(default_value_of_type_right)
right.unwrap_left_or(default_value_of_type_left)
right.unwrap_or_else(left_function)

right.match(left_function, right_function)

right.is_right_and(filter_function)  # True if Right and filter_function(value) is True
right.is_left_and(filter_function)  # True if Left and filter_function(value) is True
right.map_or(0, mapping_function)  # Maps the function, or returns the default if Left
right.and_(Right(20))  # Right(20) if right is a Right, otherwise the Left unchanged
right.or_(Right(20))  # right if it is a Right, otherwise Right(20)
right.zip(Right("a"))  # Right((value, "a")), or the Left if either side is Left
Right(Right(10)).flatten()  # Right(10)

right.right()  # Maybe[TypeRight]: Just(value) if Right, otherwise Nothing
right.left()  # Maybe[TypeLeft]: Just(error) if Left, otherwise Nothing

either_safe(
    unsafe_function
)  # Either[Exception, TypeRight], from pymoliath.either import either_safe
```

## Result

* Rust: [std::result::Result](https://doc.rust-lang.org/std/result/enum.Result.html#)

```python
# Result[TypeOk, TypeErr]
result: Result[int, str] = Ok(10)
error: Result[int, str] = Err("error")
applicative = Ok(lambda x: x + 1)

result.is_ok()
result.is_err()

# Monad functions
result.map(ok_map_function)  # Maps only the ok value with the function
result.map_err(err_map_function)  # Maps only the err value with the function
result.bind(ok_bind_function)  # Binds only the ok value with the function
result.bind_err(err_bind_function)  # Binds only the err value with the function
result.apply(applicative)  # Apply the ok value to the applicative (if not err)
applicative.apply2(result)  # Apply the applicative to the ok value (if not err)

result.unwrap()  # Returns the result Ok value or raises an Exception containing the Err value.
result.unwrap_or(default_value_of_type_ok)
result.unwrap_or_else(err_function)
result.unwrap_err_or(default_exception_value)

result.match(err_function, ok_function)

result.is_ok_and(filter_function)  # True if Ok and filter_function(value) is True
result.is_err_and(filter_function)  # True if Err and filter_function(value) is True
result.map_or(0, mapping_function)  # Maps the function, or returns the default if Err
result.and_(Ok(20))  # Ok(20) if result is an Ok, otherwise the Err unchanged
result.or_(Ok(20))  # result if it is an Ok, otherwise Ok(20)
result.zip(Ok("a"))  # Ok((value, "a")), or the Err if either side is Err
Ok(Ok(10)).flatten()  # Ok(10)

result.ok()  # Option[TypeOk]: Some(value) if Ok, otherwise Nil
result.err()  # Option[TypeErr]: Some(error) if Err, otherwise Nil

result_safe(
    unsafe_function
)  # Result[TypeOk, Exception], from pymoliath.result import result_safe
```

## Try

It represents and handles computations which might be fail. The return value is either the specified result type or a
python exception type.

* Scala: [scala.util.Try](https://www.scala-lang.org/api/2.12.4/scala/util/Try.html)

```python
# Try[TypeSource]
success: Try[int] = Success(10)
failure: Try[int] = Failure(Exception("error"))
applicative = Success(lambda x: x + 1)

success.is_success()
success.is_failure()

# Monad functions
success.map(ok_map_function)  # Maps only the success value with the function
success.map_failure(err_map_function)  # Maps only the failure value with the function
success.bind(ok_bind_function)  # Binds only the success value with the function
success.bind_failure(
    err_bind_function
)  # Binds only the failure value with the function
success.apply(
    applicative
)  # Apply the success value to the applicative (if not failure)
applicative.apply2(
    success
)  # Apply the applicative to the success value (if not failure)

success.unwrap()  # Returns the success value or raises the Exception contained in the Failure.
success.unwrap_or(default_value_of_type_ok)
success.unwrap_or_else(err_function)
success.unwrap_failure_or(default_exception_value)

success.match(err_function, ok_function)
success.to_either()  # Either[Exception, TypeSource]
success.to_result()  # Result[TypeSource, Exception]

success.is_success_and(
    filter_function
)  # True if Success and filter_function(value) is True
success.is_failure_and(
    filter_function
)  # True if Failure and filter_function(exception) is True
success.map_or(
    0, mapping_function
)  # Maps the function, or returns the default if Failure
success.and_(
    Success(20)
)  # Success(20) if success is a Success, otherwise the Failure unchanged
success.or_(Success(20))  # success if it is a Success, otherwise Success(20)
success.zip(
    Success("a")
)  # Success((value, "a")), or the Failure if either side is a Failure
Success(Success(10)).flatten()  # Success(10)

safe(unsafe_function)  # Try[TypeSource], from pymoliath.exception import safe
```

## IO

Haskell [System.IO](https://hackage.haskell.org/package/base-4.16.0.0/docs/System-IO.html#t:IO)

```python
# IO[TypeSource]
print_io: IO[None] = lambda x: IO(lambda: print(x))
print_io(10).run()
io_value = IO[int] = IO(lambda: 10)
applicative = IO[Callable[[int], int]] = IO(lambda: lambda x: x + 1)

# Monad functions
io_value.map(map_function).run()  # Maps the resulting io value with the function
io_value.bind(bind_function).run()  # Binds the resulting io value with the function
io_value.apply(applicative).run()  # Apply the resulting io value to the applicative
applicative.apply2(io_value).run()  # Apply the resulting applicative to the io value

io_value.run()  # Returns the value 10
```

## ListMonad

Haskell: [Data.List](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-List.html)

Besides the Monad interface, `ListMonad` offers a set of Rust `Iterator`-inspired combinators. All of them return a
new `ListMonad`/`Option`/tuple - none mutate the underlying list in place.

```python
# ListMonad[TypeSource]
list_monad: ListMonad[int] = ListMonad([1, 2, 3, 4, 5, 6])
list_monad.take(4)  # ListMonad([1, 2, 3, 4])
list_monad.filter(lambda v: v > 4)  # ListMonad([5, 6])
list_monad.to_list()  # List[TypeSource]
applicative = ListMonad([lambda x: x + 1])

# Monad functions
list_monad.map(map_function)  # Maps all values with the function
list_monad.bind(bind_function)  # Binds all values the function
list_monad.apply(applicative)  # Applies all values to the applicative
applicative.apply2(list_monad)  # Applies the applicative to all values

# Iterator-style combinators
list_monad.fold(0, lambda acc, v: acc + v)  # Left fold with a seed value
list_monad.reduce(lambda acc, v: acc + v)  # Option[TypeSource]: Nil if empty
list_monad.enumerate()  # ListMonad[(index, value)]
list_monad.zip(other)  # ListMonad[(value, other_value)], stops at the shorter one
list_monad.zip_with(other, combine_function)  # zip + combine in one step
list_monad.chain(other)  # Concatenates this ListMonad with another iterable
ListMonad(
    [[1, 2], [3, 4]]
).flatten()  # ListMonad([1, 2, 3, 4]): flattens one level of nesting
list_monad.any(filter_function)  # True if any element matches
list_monad.all(filter_function)  # True if all elements match
list_monad.find(filter_function)  # Option[TypeSource]: first match, or Nil
list_monad.position(filter_function)  # Option[int]: index of the first match, or Nil
list_monad.min()  # Option[TypeSource], Nil if empty
list_monad.max()  # Option[TypeSource], Nil if empty
list_monad.min_by_key(key_function)  # Option[TypeSource], Nil if empty
list_monad.max_by_key(key_function)  # Option[TypeSource], Nil if empty
list_monad.partition(filter_function)  # (matching, non_matching) ListMonads
list_monad.take_while(filter_function)  # Leading elements while the predicate holds
list_monad.skip_while(filter_function)  # Remainder from the first non-matching element
list_monad.step_by(2)  # Every 2nd element, starting with the first
list_monad.rev()  # New ListMonad in reverse order
list_monad.sum(0)  # Sums all elements, starting from an initial value
list_monad.dedup()  # Removes consecutive duplicates only (Rust Vec::dedup semantics)
list_monad.distinct()  # Removes all duplicates, preserving first-occurrence order (requires hashable elements)
list_monad.sorted()  # New sorted ListMonad
list_monad.sort_by(key_function)  # New ListMonad sorted by a key function
```

## Sequence (lazy list)

Evaluates a given list in a lazy way. Every intermediate operation (`map`/`bind`/`filter`/`take`/`skip`/`enumerate`/
`zip`/`chain`/`flatten`/`take_while`/`skip_while`/`step_by`/`dedup`/`distinct`) builds a new pipeline without
consuming the source - it is only pulled from element by element once a terminal operation is called
(`run`/`fold`/`reduce`/`any`/`all`/`find`/`position`/`min`/`max`/`sum`/`count`/`partition`, or iterating the
`Sequence` directly). Predicate-based terminal operations (`any`, `all`, `find`, `position`) genuinely short-circuit,
so they work on infinite sources, e.g. `Sequence(itertools.count()).find(lambda x: x > 100)` terminates. `rev()`,
`sorted()` and `sort_by()` need to see every element to produce their first result, so they fully materialize the
Sequence and return a `ListMonad` instead of a `Sequence` - they will hang on an infinite source.

Note: constructing a `Sequence` directly from a raw generator/iterator (rather than from a list, or a callable
returning a fresh iterable) makes the `Sequence` single-use, since the underlying generator is exhausted the first
time it is consumed.

```python
# Sequence[TypeSource]
sequence: Sequence[int] = Sequence([1, 2, 3, 4, 5, 6])
sequence.take(4)  # Sequence([1, 2, 3, 4])
sequence.filter(lambda v: v > 4)  # Sequence([5, 6])
sequence.run()  # List[TypeSource]
applicative = Sequence([lambda x: x + 1])

# Monad functions
sequence.map(map_function).run()  # Lazy mapping of all values with the function
sequence.bind(bind_function).run()  # Lazy binding of all values with the function
sequence.apply(applicative).run()  # Lazy applies all values to the applicative
applicative.apply2(sequence).run()  # Lazy applies the applicative to all values

# Iterator-style combinators - same semantics as ListMonad, see above
sequence.enumerate()  # Sequence[(index, value)]
sequence.zip(other)  # Sequence[(value, other_value)]
sequence.chain(other)  # Sequence concatenation
sequence.take_while(filter_function)
sequence.skip_while(filter_function)
sequence.step_by(2)
sequence.dedup()
sequence.distinct()

# Terminal operations
sequence.fold(0, lambda acc, v: acc + v)
sequence.reduce(lambda acc, v: acc + v)  # Option[TypeSource]
sequence.any(filter_function)  # Short-circuits
sequence.all(filter_function)  # Short-circuits
sequence.find(filter_function)  # Option[TypeSource], short-circuits
sequence.position(filter_function)  # Option[int], short-circuits
sequence.min()  # Option[TypeSource]
sequence.max()  # Option[TypeSource]
sequence.min_by_key(key_function)  # Option[TypeSource]
sequence.max_by_key(key_function)  # Option[TypeSource]
sequence.sum(0)
sequence.count()  # int
sequence.partition(filter_function)  # (ListMonad, ListMonad)
sequence.rev()  # ListMonad, terminates the pipeline
sequence.sorted()  # ListMonad, terminates the pipeline
sequence.sort_by(key_function)  # ListMonad, terminates the pipeline
```

## Reader

Haskell: [Control.Monad.Reader](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Reader.html)

```python
# Reader[TypeEnvironment, TypeSource]
reader: Reader[int, str] = Reader(lambda env: str(env))
reader.run(12)  # '12'
reader.local(change_environment_function)
# Reader[TypeEnvironment, TypeEnvironment]
Reader.ask()  # Reader(lambda env: env)

applicative = Reader(lambda env: lambda x: x + env)

# Monad functions
reader.map(map_function).run(10)  # Map resulting value with the function
reader.bind(bind_function).run(10)  # Bind resulting value with the function
reader.apply(applicative).run(10)  # Apply resulting value to the applicative
applicative.apply2(reader).run(10)  # Apply resulting applicative to the value
```

## Writer

Haskell: [Control.Monad.Writer.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Writer-Lazy.html)

```python
# Writer[TypeSource, TypeMonoid]
writer: Writer[int, str] = Writer(10, "hello")
writer.run()  # (10, 'hello')

writer.tell(" world")  # Writer(10, 'hello world')
writer.listen()  # Writer((10, 'hello world'), 'hello world')
Writer((10, lambda w: w + " world"), "hello").pass_()  # Writer((10, "hello world")

applicative = Writer(lambda x: x, "")

# Monad functions
writer.map(map_function).run()  # Map resulting value with the function
writer.bind(bind_function).run()  # Bind resulting value with the function
writer.apply(applicative).run()  # Apply resulting value to the applicative
applicative.apply2(writer).run()  # Apply resulting applicative to the value
```

## State

Haskell: [Control.Monad.State.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-State-Lazy.html)

```python
# State[TypeState, TypeSource]
state: State[str, int] = State(lambda state: (state, 10))
state.run("hi")  # (hi, 10)

State.get()  # State(lambda state: (state, state))
State.put(new_state)  # State(lambda state: (new_state, ()))

applicative = State(lambda state: (state, lambda x: x * 42))

# Monad functions
state.map(map_function).run("hi")  # Map resulting value with the function
state.bind(bind_function).run("hi")  # Bind resulting value with the function
state.apply(applicative).run("hi")  # Apply resulting value to the applicative
applicative.apply2(state).run("hi")  # Apply resulting applicative to the value
```

## LazyMonad

* [LazyMonad](https://www.philliams.com/monads-in-python/)

```python
# LazyMonad[TypeSource]
lazy: LazyMonad[int] = LazyMonad(lambda: 10)
lazy.run()  # TypeSource: 10

applicative = LazyMonad(lambda: lambda x: x * 42)

# Monad functions
lazy.map(map_function).run()  # Map resulting value with the function
lazy.bind(bind_function).run()  # Bind resulting value with the function
lazy.apply(applicative).run()  # Apply resulting value to the applicative
applicative.apply2(lazy).run()  # Apply resulting applicative to the value
```

## Continuation

Represents computations in continuation-passing style (CPS). Instead of returning a result, a computation receives a
callback and passes its result to it.

```python
# Continuation[TypeSource, TypeReturn]
continuation: Continuation[int, str] = Continuation(lambda callback: callback(10))
continuation.run(lambda x: str(x))  # '10'

# Monad functions
continuation.map(map_function).run(
    callback
)  # Maps the resulting value before it reaches the callback
continuation.bind(bind_function).run(
    callback
)  # Binds the resulting value to a new continuation

applicative: Continuation[Callable[[int], str], str] = Continuation(
    lambda callback: callback(lambda x: str(x))
)
continuation.apply(applicative).run(
    callback
)  # Apply the resulting value to the applicative
applicative.apply2(continuation).run(
    callback
)  # Apply the resulting applicative to the value
```

# Async Monads

`Async*` counterparts exist for `Maybe`, `Option`, `Either`, `Result`, `Try`, `IO`, `Writer`, `Reader` and `State`
(`ListMonad`, `Sequence`, `LazyMonad` and `Continuation` stay sync-only). Each `AsyncX` wraps a deferred computation
instead of a value: building a chain of `map`/`bind`/... never executes anything - only awaiting the final `AsyncX`
does.

* **Directly awaitable.** `await AsyncMaybe.from_value(1).map(f)` runs the whole pipeline and resolves to the
  underlying sync monad (`Just`/`Nothing`, `Left`/`Right`, `Ok`/`Err`, `Some`/`Nil`, `Success`/`Failure`), so
  short-circuiting and pattern matching work exactly like the sync versions once awaited. `AsyncIO` and
  `AsyncWriter` have no failure state, so they resolve directly to the raw value / `(value, monoid)` tuple instead,
  matching their sync `run()`. `AsyncReader` and `AsyncState` need an `env`/`state` argument to run, so instead of
  a bare `await`, call `await an_async_reader.run(env)` / `await an_async_state.run(state)`.
* **Sync or async callbacks, auto-detected.** `map`/`bind`/`filter`/`inspect` (and their `_left`/`_err`/`_failure`
  counterparts) accept a plain sync function or an `async def` function interchangeably - whichever is returned is
  detected at the point it's called, so real async I/O composes freely with plain transforms in the same chain.
  `bind` (and its channel-specific counterparts) similarly accepts a callback returning another `AsyncX`, a plain
  sync monad, or an awaitable resolving to one.
* **Construction.** Every `AsyncX` can be built from an already-resolved value or sync monad (`from_value`,
  `from_maybe`, `from_either`, `from_result`, ...) or from an async callable (`from_coroutine`), and stays
  re-awaitable/re-runnable as long as that callable produces a fresh awaitable each call.
* **Same MVP scope as the sync terminal methods.** `unwrap*`, `match`, `is_*`, and the `Maybe`/`Result`/`Either`
  cross-conversions aren't duplicated on the `Async*` classes - awaiting already hands back the full sync monad,
  so e.g. `(await an_async_maybe).unwrap_or(0)` works with zero extra API surface.

## AsyncMaybe / AsyncOption

```python
# AsyncMaybe[TypeSource] - AsyncOption[TypeSource] is the same interface (Some/Nil naming)
await AsyncMaybe.from_value(10).map(lambda x: x + 1)  # Just(11)
await AsyncMaybe.from_maybe(Nothing()).map(lambda x: x + 1)  # Nothing(), map is never called


async def fetch(x: int) -> int: ...


await AsyncMaybe.from_value(10).map(fetch)  # async callback, auto-detected
await AsyncMaybe.from_value(10).bind(lambda x: AsyncMaybe.from_value(x + 1))  # Just(11)
await AsyncMaybe.from_value(10).filter(lambda x: x > 5)  # Just(10)


async def fetch_ten() -> int:
    return 10


await AsyncMaybe.from_coroutine(fetch_ten)  # Just(10)
```

## AsyncEither

```python
# AsyncEither[TypeLeft, TypeRight]
await AsyncEither.from_right(10).map(lambda x: x + 1)  # Right(11)
await AsyncEither.from_left("error").map(lambda x: x + 1)  # Left("error"), map is never called
await AsyncEither.from_right(10).map_left(lambda e: e.upper())  # Right(10), map_left is never called
await AsyncEither.from_left("error").map_left(str.upper)  # Left("ERROR")

await AsyncEither.from_right(10).bind(lambda x: AsyncEither.from_right(x + 1))  # Right(11)
await AsyncEither.from_either(Right(10))  # lifts an existing sync Either
```

## AsyncResult

```python
# AsyncResult[TypeOk, TypeErr]
await AsyncResult.from_ok(10).map(lambda x: x + 1)  # Ok(11)
await AsyncResult.from_err("error").map(lambda x: x + 1)  # Err("error"), map is never called
await AsyncResult.from_err("error").map_err(str.upper)  # Err("ERROR")

await AsyncResult.from_ok(10).bind(lambda x: AsyncResult.from_ok(x + 1))  # Ok(11)
await AsyncResult.from_result(Ok(10))  # lifts an existing sync Result
```

## AsyncTry

```python
# AsyncTry[TypeSource]
await AsyncTry.from_success(10).map(lambda x: x + 1)  # Success(11)
await AsyncTry.from_failure(Exception("error")).map(lambda x: x + 1)  # Failure(...), map never called


def boom(x: int) -> int:
    raise ValueError("boom")


await AsyncTry.from_success(10).map(boom)  # Failure(ValueError("boom")) - exceptions are caught, like sync Try
await AsyncTry.from_success(10).bind(lambda x: AsyncTry.from_success(x + 1))  # Success(11)
await AsyncTry.from_try(Success(10))  # lifts an existing sync Try
```

## AsyncIO

```python
# AsyncIO[TypeSource] - resolves directly to the raw value, no wrapper (IO always "succeeds")
await AsyncIO.from_value(10).map(lambda x: x + 1)  # 11
await AsyncIO.from_value(10).bind(lambda x: AsyncIO.from_value(x + 1))  # 11
await AsyncIO.from_io(IO(lambda: 10))  # lifts an existing sync IO
await AsyncIO.from_coroutine(fetch_ten)  # 10
```

## AsyncWriter

```python
# AsyncWriter[TypeSource, TypeMonoid] - resolves directly to (value, monoid), no wrapper
await AsyncWriter.from_value(10, "hello")  # (10, 'hello')
await AsyncWriter.from_value(10, "hello").map(lambda x: x + 1)  # (11, 'hello')
await AsyncWriter.from_value(10, "hello").tell(" world")  # (10, 'hello world')
await AsyncWriter.from_value(10, "hello").bind(
    lambda x: AsyncWriter.from_value(x + 1, " world")
)  # (11, 'hello world') - monoids combine via +
```

## AsyncReader

```python
# AsyncReader[TypeEnvironment, TypeSource] - NOT bare-awaitable, run() needs the environment
reader = AsyncReader.from_value(10).map(lambda x: x + 1)
await reader.run("env")  # 11 - the environment is ignored here, from_value always resolves to 10 -> 11

await AsyncReader.ask().run(12)  # 12
await AsyncReader.from_reader(Reader(lambda env: str(env))).run(12)  # '12'
```

## AsyncState

```python
# AsyncState[TypeState, TypeSource] - NOT bare-awaitable, run() needs the initial state
await AsyncState.from_value(10).map(lambda x: x + 1).run("hi")  # ('hi', 11)

await AsyncState.get().run("hi")  # ('hi', 'hi')
await AsyncState.put("new").run("hi")  # ('new', ())
```

# Util

`pymoliath.util` provides a small set of functional-programming prelude helpers used internally (`curry` powers
every Monad's `apply`/`apply2`) and available for general use.

```python
from pymoliath.util import compose, curry, pipe, identity, const, flip

compose(f, g)(x)  # f(g(x)): composes functions right to left
pipe(f, g)(x)  # g(f(x)): composes functions left to right, the mirror of compose

curry(
    function
)  # Returns a curried version of function, applying arguments until all are provided

identity(x)  # x: returns its argument unchanged
const(10)("ignored")  # 10: always returns 10 regardless of the argument
flip(lambda a, b: a - b)(2, 10)  # 8: swaps the argument order of a 2-arg function
```
