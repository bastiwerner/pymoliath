"""
# Async Monads

Async monads provide a unified way to compose computations that involve side effects or
deferred execution (like I/O, network requests, or database queries) while maintaining
the declarative structure of functional programming. They allow you to chain operations
that may fail, produce side effects, or require environmental context without manually
managing `await` calls or nested error handling at every step.

The `Async*` counterparts exist for `Maybe`, `Option`, `Either`, `Result`, `Try`,
`IO`, `Writer`, `Reader`, and `State` (`ListMonad`, `Sequence`, `LazyMonad`, and
`Continuation` remain sync-only). Unlike their synchronous counterparts, each `AsyncX`
wraps a deferred computation rather than a concrete value. Consequently, building a
chain of `map`, `bind`, and other transformations performs no actual execution;
the computation is only triggered when the final `AsyncX` is awaited.

Usage examples for each `AsyncX` are provided alongside their sync counterparts
in each monad's `docs/` page (see the Components section below).

- **Directly awaitable.** Awaiting an `AsyncX` (e.g., `await AsyncMaybe.from_just(1).map(f)`)
  executes the entire pipeline and resolves to the underlying sync monad
  (`Just`/`Nothing`, `Left`/`Right`, `Ok`/`Err`, `Some`/`Nil`, `Success`/`Failure`).
  This ensures that short-circuiting and pattern matching behave identically to the
  sync versions once the result is yielded.
  * Note: `AsyncIO` and `AsyncWriter` do not have failure states and resolve directly
  to the raw value or `(value, monoid)` tuple.
  * Note: `AsyncReader` and `AsyncState` require environment/state arguments; use
  `await an_async_reader.run(env)` or `await an_async_state.run(state)`.
- **Seamless Sync/Async Integration.** The `map`, `bind`, `filter`, and `inspect`
  methods (including `_left`, `_err`, and `_failure` variants) automatically
  detect whether they are receiving a plain sync function or an `async def` function.
  This allows real async I/O to compose fluently with pure transformations in the
  same chain. Similarly, `bind` accepts callbacks returning another `AsyncX`,
  a plain sync monad, or an awaitable resolving to one.
- **Flexible Construction.** Every sum-type `AsyncX` has one constructor per variant, named
  after it (`AsyncResult.from_ok`/`from_err`, `AsyncEither.from_right`/`from_left`,
  `AsyncOption.from_some`/`from_nil`, `AsyncMaybe.from_just`/`from_nothing`,
  `AsyncTry.from_success`/`from_failure`), one from the sync monad (`from_result`,
  `from_either`, `from_option`, `from_maybe`, `from_try`) and one from an async callable
  (`from_coroutine`). The instance remains re-awaitable and re-runnable as long as the
  underlying callable produces a fresh awaitable on each call.
- **Running.** `await an_async_x` resolves it inside a coroutine. `an_async_x.run()` returns a
  coroutine, for APIs such as `asyncio.run` that require one (before Python 3.14):
  `asyncio.run(AsyncResult.from_ok(1).run())`.
- **Minimalist API Surface.** To keep the API clean, terminal methods like
  `unwrap*`, `match`, `is_*`, and cross-conversions between `Maybe`/`Result`/`Either`
  are not duplicated on the `Async*` classes. Since awaiting an `AsyncX`
  returns the full sync monad, you can simply call the sync methods on the
  result (e.g., `(await an_async_maybe).unwrap_or(0)`).
"""

from . import (
    async_either,
    async_io,
    async_maybe,
    async_option,
    async_reader,
    async_result,
    async_state,
    async_try,
    async_writer,
)
from .async_either import AsyncEither
from .async_io import AsyncIO
from .async_maybe import AsyncMaybe
from .async_option import AsyncOption
from .async_reader import AsyncReader
from .async_result import AsyncResult
from .async_state import AsyncState
from .async_try import AsyncTry
from .async_writer import AsyncWriter

__all__ = [
    "async_either",
    "async_io",
    "async_maybe",
    "async_option",
    "async_reader",
    "async_result",
    "async_state",
    "async_try",
    "async_writer",
    "AsyncEither",
    "AsyncResult",
    "AsyncTry",
    "AsyncIO",
    "AsyncMaybe",
    "AsyncOption",
    "AsyncReader",
    "AsyncState",
    "AsyncWriter",
]
