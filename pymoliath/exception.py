"""
# Try Monad

The Try Monad is a container used to represent computations that either succeed with a value or
fail by raising an exception. It encapsulates the outcome of code that might raise, allowing for a
functional approach to exception handling by chaining operations without wrapping every step in
its own `try`/`except` block. Unlike `Result`, whose `Ok`/`Err` values are constructed explicitly
by the caller, `Try`'s `map`/`bind` automatically catch any `Exception` raised by the function they
are given and turn it into a `Failure`.

* Scala: [Try](https://www.scala-lang.org/api/current/scala/util/Try.html)

This implementation is heavily inspired by Scala's `Try` type.

The `Try` type is a sealed sum type that can be either `Success` or `Failure`. Like Rust's
`Result`, both variants carry the type parameter:

```python
type Try[T] = Success[T] | Failure[T]
```

The type parameter is covariant and a bare `Failure(e)` is a `Failure[Never]`, so it is assignable
to every `Try[T]` and a conditional lambda such as
`lambda x: Success(x) if x > 0 else Failure(ValueError("negative"))` is inferred as a `Try[int]`.

Two Failures are equal if their exceptions have the same type and `args`, e.g.
`Failure(ValueError("x")) == Failure(ValueError("x"))`.

## Practical Examples and Benefits:

The Try Monad is particularly useful in scenarios where a chain of operations might each raise an
exception (e.g. parsing a string, calling out to a library that isn't exception-free, or performing
several arithmetic operations that could divide by zero), and you want the first exception to short
-circuit the rest of the chain without a `try`/`except` around each step.

### Benefits:
1. Declarative Code: `map`/`bind` chain operations together while automatically catching any
   exception the passed function raises, rather than requiring a `try`/`except` per step.
2. Error Propagation: If any step in a chain of operations produces a `Failure`, the subsequent
   operations are skipped automatically, and the final result stays a `Failure`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly in the
   return type, rather than having exceptions propagate implicitly and invisibly through the call
   stack.

#### Example: Parsing and transforming a value that might raise.

```python
# Without Try (Imperative)
try:
    parsed = int(raw_value)
    doubled = parsed * 2
    print(doubled)
except ValueError as e:
    print(f"Error: {e}")

# With Try (Functional)
(Success.safe(lambda: int(raw_value))
    .map(lambda parsed: parsed * 2)
    .match(success=print, failure=lambda e: print(f"Error: {e}")))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Try`
monad. `Success` and `Failure` are the only variants (both are `@final`), so a `match` over both is
exhaustive and type checkers narrow the value in each branch, without manually checking
`is_success()`/`is_failure()`.

```python
match try_value:
    case Success(x):
        # This block executes if the computation succeeded
        print(f"Success value: {x}")
    case Failure(e):
        # This block executes if the computation raised
        print(f"Exception encountered: {e}")
```

`Try` is a type alias, so check it at runtime with `isinstance(x, (Success, Failure))`. Both a `match`
over `Success`/`Failure` and `isinstance(x, Success)` narrow the type to the variant.

## Which methods catch exceptions

The methods that build a new Try catch any `Exception` raised by the function they are given and
turn it into a `Failure`: `map`, `bind`/`and_then`, `apply`, `apply2`, `map_failure`,
`bind_failure`/`or_else`, `filter` and `safe` (which catches the given `exceptions`).

The methods that leave the Try (or only observe it) let exceptions propagate to the caller:
`map_or`, `map_or_else`, `is_success_and`, `is_failure_and`, `inspect`, `inspect_failure`, `match`,
`unwrap_or_else` and `unwrap` (which re-raises the stored exception).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, Self, final, overload

from typing_extensions import Generic, TypeVar

from pymoliath.either import Either, Left, Right
from pymoliath.result import Err, Ok, Result

# Covariant, so `Success[bool]` is a `Try[int]` and a bare `Failure(e)` (a `Failure[Never]`) is
# every `Try[T]`. Like in Rust the receiver fixes the types some methods accept
# (`unwrap_or(default: T)`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `type: ignore` (misc).
#
# Short-circuit paths return the instance itself instead of allocating a new one. Only the unused
# (phantom) type parameter changes there, so they are typed through `Any` in that one slot - a
# `self: Failure[Any]` annotation or a `failed: Failure[Any] = ...` local - which costs nothing at
# runtime.
T = TypeVar("T", covariant=True)

# The type parameter of `Success` and `Failure`. It defaults to Never (PEP 696), so a bare
# `Failure(e)` is a `Failure[Never]` and assignable to every `Try[T]`.
SuccessT = TypeVar("SuccessT", covariant=True, default=Never)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")


class _TryImpl(Generic[T]):
    """Public interface of the Try Monad. The behaviour lives in `Success` and `Failure`, its only
    subclasses."""

    __slots__ = ()

    def map(self, function: Callable[[T], U]) -> Try[U]:
        """Calls function on a wrapped Success value, otherwise leaving the Failure untouched.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        try: Try[U]
            Returns a Success with the function result or otherwise a Failure.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map(lambda x: x + 1)
        Success(6)
        >>> def risky(x: int) -> int:
        ...     raise ValueError("boom")
        >>> val.map(risky)
        Failure(ValueError('boom'))
        """
        raise NotImplementedError

    def map_failure(self, function: Callable[[Exception], Exception]) -> Try[T]:
        """Calls function on a wrapped Failure exception, otherwise leaving the Success untouched.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[Exception], Exception]
            Function which takes an Exception and returns an Exception.

        Returns
        -------
        try: Try[T]
            Returns a Failure with the function result or otherwise the Success.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map_failure(lambda e: ValueError("wrapped"))
        Success(5)
        >>> failed: Try[int] = Failure(ValueError("boom"))
        >>> failed.map_failure(lambda e: ValueError(f"wrapped {e}"))
        Failure(ValueError('wrapped boom'))
        """
        raise NotImplementedError

    def bind(self, function: Callable[[T], Try[U]]) -> Try[U]:
        """Calls function if the Try Monad is Success, otherwise returns the Failure.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[T], Try[U]]
            Function which takes a value of T and returns a new Try Monad.

        Returns
        -------
        try: Try[U]
            Returns the function result if Success, otherwise a Failure.

        Examples
        --------
        >>> def get_next(x: int) -> Try[int]:
        ...     return Success(x + 1) if x < 10 else Failure(ValueError("too big"))
        >>> Success(5).bind(get_next)
        Success(6)
        >>> Success(10).bind(get_next)
        Failure(ValueError('too big'))
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[T], Try[U]]) -> Try[U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[T], Try[U]]
            Function which takes a value of T and returns a new Try Monad.

        Returns
        -------
        try: Try[U]
            Returns the function result if Success, otherwise a Failure.

        Examples
        --------
        >>> Success(5).and_then(lambda x: Success(x * 2))
        Success(10)
        """
        raise NotImplementedError

    def bind_failure(self, function: Callable[[Exception], Try[T]]) -> Try[T]:
        """Calls function if the Try Monad is Failure, otherwise returns the Success.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[Exception], Try[T]]
            Function which takes the Exception and returns a new Try Monad.

        Returns
        -------
        try: Try[T]
            Returns the function result if Failure, otherwise the Success.

        Examples
        --------
        >>> failed: Try[int] = Failure(ValueError("boom"))
        >>> failed.bind_failure(lambda e: Success(0))
        Success(0)
        """
        raise NotImplementedError

    def or_else(self, function: Callable[[Exception], Try[T]]) -> Try[T]:
        """Alias of `bind_failure`, named like in Rust.

        Parameters
        ----------
        function: Callable[[Exception], Try[T]]
            Function which takes the Exception and returns a new Try Monad.

        Returns
        -------
        try: Try[T]
            Returns the function result if Failure, otherwise the Success.

        Examples
        --------
        >>> failed: Try[int] = Failure(ValueError("boom"))
        >>> failed.or_else(lambda e: Success(len(str(e))))
        Success(4)
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> Try[T]:
        """Keeps the Success if the predicate holds for its value (Scala: `filter`).

        A Success whose value fails the predicate becomes a Failure of a `ValueError`. An exception
        raised by the predicate is caught and turned into a Failure. A Failure is returned unchanged.

        Parameters
        ----------
        predicate: Callable[[T], bool]
            Predicate function applied to the Success value.

        Returns
        -------
        try: Try[T]
            Returns the Success if the predicate holds, otherwise a Failure.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.filter(lambda x: x > 5)
        Success(10)
        >>> val.filter(lambda x: x > 50)
        Failure(ValueError('predicate does not hold for 10'))
        """
        raise NotImplementedError

    def apply(self, function: Try[Callable[[T], U]]) -> Try[U]:
        """Applies the function wrapped in `function` to the Success value if both are Success.
        Any exception raised by the function is caught and turned into a Failure.

        If both are Failure, the Failure of `function` takes precedence. For functions of several
        arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: Try[Callable[[T], U]]
            Try Monad which contains a function.

        Returns
        -------
        try: Try[U]
            Returns Success of the function result if both are Success, otherwise a Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val: Try[int] = Success(10)
        >>> val.apply(func)
        Success(20)
        """
        raise NotImplementedError

    def apply2(self: _TryImpl[Callable[[U], V]], value: Try[U]) -> Try[V]:
        """Applies the function wrapped in this Try Monad to the value wrapped in `value`.

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both are
        Failure, this (the function side) takes precedence. Any exception raised by the function is
        caught and turned into a Failure. Functions of several arguments can be applied one
        argument at a time when they are curried, e.g.
        `Success(lambda a: lambda b: a + b).apply2(x).apply2(y)`.

        Parameters
        ----------
        value: Try[U]
            Try Monad which contains the argument.

        Returns
        -------
        try: Try[V]
            Returns Success of the function result if both are Success, otherwise a Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], float]] = Success(lambda y: 10 / y)
        >>> func.apply2(Success(5))
        Success(2.0)
        >>> func.apply2(Success(0))
        Failure(ZeroDivisionError('division by zero'))
        """
        raise NotImplementedError

    def is_success_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Try Monad is Success and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Success value.

        Returns
        -------
        result: bool
            Returns the predicate result if Success, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.is_success_and(lambda x: x > 5)
        True
        """
        raise NotImplementedError

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        """Returns True if the Try Monad is Failure and the predicate returns True for the
        exception.

        Parameters
        ----------
        function: Callable[[Exception], bool]
            Predicate function applied to the Failure exception.

        Returns
        -------
        result: bool
            Returns the predicate result if Failure, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_failure_and(lambda e: isinstance(e, ValueError))
        True
        """
        raise NotImplementedError

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Success value, or returns the default value if Failure.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Try Monad is Failure.
        function: Callable[[T], U]
            Function applied to the Success value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[Exception], U], function: Callable[[T], U]
    ) -> U:
        """Applies the function to the Success value, or the default function to the exception.

        Parameters
        ----------
        default_function: Callable[[Exception], U]
            Function applied to the Failure exception.
        function: Callable[[T], U]
            Function applied to the Success value.

        Returns
        -------
        result: U
            Returns the result of whichever function was applied.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.map_or_else(lambda e: str(e), lambda x: str(x * 2))
        'boom'
        """
        raise NotImplementedError

    def and_(self, other: Try[U]) -> Try[U]:
        """Returns `other` if the Try Monad is Success, otherwise the Failure.

        Parameters
        ----------
        other: Try[U]
            Try Monad to be returned if this Try Monad is Success.

        Returns
        -------
        try: Try[U]
            Returns `other` or the Failure.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.and_(val2)
        Success(2)
        """
        raise NotImplementedError

    def or_(self, other: Try[T]) -> Try[T]:
        """Returns this Try Monad if it is Success, otherwise `other`.

        Parameters
        ----------
        other: Try[T]
            Try Monad to be returned if this Try Monad is Failure.

        Returns
        -------
        try: Try[T]
            Returns the Success or `other`.

        Examples
        --------
        >>> val1: Try[int] = Failure(ValueError("boom"))
        >>> val2: Try[int] = Success(2)
        >>> val1.or_(val2)
        Success(2)
        """
        raise NotImplementedError

    def zip(self, other: Try[U]) -> Try[tuple[T, U]]:
        """Combines this Try Monad with another into a Try Monad of a tuple, or the first Failure.

        Parameters
        ----------
        other: Try[U]
            Try Monad to be zipped with this Try Monad.

        Returns
        -------
        try: Try[tuple[T, U]]
            Returns Success of a tuple of both values, or Failure.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.zip(val2)
        Success((1, 2))
        """
        raise NotImplementedError

    def flatten(self: _TryImpl[Try[U]]) -> Try[U]:
        """Flattens a nested Try Monad by one level.

        Returns
        -------
        try: Try[U]
            Returns the nested Try Monad if Success, otherwise the Failure.

        Examples
        --------
        >>> Success(Success(1)).flatten()
        Success(1)
        """
        raise NotImplementedError

    def unwrap(self) -> T:
        """Returns the Success value, or otherwise raises the wrapped exception.

        Returns
        -------
        result: T
            Returns the Success value.

        Raises
        ------
        Exception
            The wrapped exception if the Try Monad is Failure.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap()
        1
        """
        raise NotImplementedError

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        """Returns the Success value, or otherwise the provided default value.

        On a `Try[T]` the default must be a T. A bare `Failure(e)` accepts a default of any type.

        Parameters
        ----------
        default_value: T
            Default value used if the Try Monad is Failure.

        Returns
        -------
        result: T
            Returns the Success value or the default value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or(0)
        0
        """
        raise NotImplementedError

    def unwrap_or_else(self, function: Callable[[Exception], T]) -> T:
        """Returns the Success value, or otherwise the result of `function` called with the exception.

        Parameters
        ----------
        function: Callable[[Exception], T]
            Function which will be called if the Try Monad is Failure.

        Returns
        -------
        result: T
            Returns the Success value or the function result.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or_else(lambda e: len(str(e)))
        4
        """
        raise NotImplementedError

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        """Returns the Failure exception, or otherwise a provided default exception.

        Parameters
        ----------
        default_value: Exception
            Default exception

        Returns
        -------
        result: Exception
            Returns the Failure exception or the default value.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap_failure_or(ValueError("default"))
        ValueError('default')
        """
        raise NotImplementedError

    def inspect(self, function: Callable[[T], None]) -> Self:
        """Calls function with the Success value (if any) and returns the Try Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Success value of the Try Monad

        Returns
        -------
        try: Try[T]
            Returns the Try Monad unchanged.

        Examples
        --------
        >>> val: Try[int] = Success(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Success(42)
        """
        raise NotImplementedError

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
        """Calls function with the Failure exception (if any) and returns the Try Monad unchanged.

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception of the Try Monad

        Returns
        -------
        try: Try[T]
            Returns the Try Monad unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.inspect_failure(lambda e: print(f"Error: {e}"))
        Error: boom
        Failure(ValueError('boom'))
        """
        raise NotImplementedError

    def match(
        self, *, success: Callable[[T], U], failure: Callable[[Exception], U]
    ) -> U:
        """Matches the Try Monad to either the `success` or the `failure` callback with the same
        return type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        success: Callable[[T], U]
            Callback function for Try monads of type Success
        failure: Callable[[Exception], U]
            Callback function for Try monads of type Failure

        Returns
        -------
        result: U
            Returns the result of the callback that was called.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.match(success=lambda x: f"Success: {x}", failure=lambda e: "Error")
        'Success: 10'
        """
        raise NotImplementedError

    def to_either(self) -> Either[Exception, T]:
        """Converts the Try Monad into an Either Monad (Success -> Right, Failure -> Left).

        Returns
        -------
        either: Either[Exception, T]
            Returns Right of the Success value, or Left of the exception.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_either()
        Right(10)
        """
        raise NotImplementedError

    def to_result(self) -> Result[T, Exception]:
        """Converts the Try Monad into a Result Monad (Success -> Ok, Failure -> Err).

        Returns
        -------
        result: Result[T, Exception]
            Returns Ok of the Success value, or Err of the exception.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_result()
        Ok(10)
        """
        raise NotImplementedError

    def is_success(self) -> bool:
        """Returns True if the Try Monad is Success, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_success()
        True
        """
        raise NotImplementedError

    def is_failure(self) -> bool:
        """Returns True if the Try Monad is Failure, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_failure()
        False
        """
        raise NotImplementedError

    @staticmethod
    @overload
    def safe(function: Callable[[], U]) -> Try[U]: ...

    @staticmethod
    @overload
    def safe(
        function: Callable[[], U], *, exceptions: tuple[type[Exception], ...]
    ) -> Try[U]: ...

    @staticmethod
    def safe(
        function: Callable[[], U],
        *,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> Try[U]:
        """Calls function and wraps its return value in Success, or a raised exception in Failure.

        Parameters
        ----------
        function: Callable[[], U]
            Callable function which may raise an exception
        exceptions: tuple[type[Exception], ...]
            The exception types to catch (default: `Exception`). Any other exception propagates.

        Returns
        -------
        try: Try[U]
            Returns a Try Monad which contains either the function result or the raised Exception.

        Examples
        --------
        >>> def risky_call():
        ...     raise ValueError("Boom")
        >>> Success.safe(risky_call)
        Failure(ValueError('Boom'))
        >>> def safe_call():
        ...     return 42
        >>> Success.safe(safe_call)
        Success(42)
        >>> Success.safe(lambda: int("x"), exceptions=(ValueError,)).is_failure()
        True
        """
        try:
            return Success(function())
        except exceptions as e:
            return Failure(e)


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Success(_TryImpl[SuccessT]):
    """The Success variant of the Try Monad, wrapping the value of a computation that completed
    without raising.

    Equality and hashing compare the wrapped value (an unhashable value makes the Success
    unhashable).

    Examples
    --------
    >>> Success(42)
    Success(42)
    >>> str(Success("text"))
    'Success(text)'
    """

    value: SuccessT

    def __init__(self, value: SuccessT) -> None:
        _set_success_value(self, value)

    def map(self, function: Callable[[SuccessT], U]) -> Try[U]:
        try:
            return Success(function(self.value))
        except Exception as e:
            return Failure(e)

    def map_failure(self, function: Callable[[Exception], Exception]) -> Self:
        return self

    def bind(self, function: Callable[[SuccessT], Try[U]]) -> Try[U]:
        try:
            return function(self.value)
        except Exception as e:
            return Failure(e)

    and_then = bind

    def bind_failure(self, function: Callable[[Exception], Try[SuccessT]]) -> Self:
        return self

    or_else = bind_failure

    def filter(self, predicate: Callable[[SuccessT], bool]) -> Try[SuccessT]:
        try:
            if predicate(self.value):
                return self
        except Exception as e:
            return Failure(e)
        return Failure(ValueError(f"predicate does not hold for {self.value!r}"))

    def apply(self, function: Try[Callable[[SuccessT], U]]) -> Try[U]:
        if isinstance(function, Success):
            try:
                return Success(function.value(self.value))
            except Exception as e:
                return Failure(e)
        failed: Failure[Any] = function
        return failed

    def apply2(self: Success[Callable[[U], V]], value: Try[U]) -> Try[V]:
        if isinstance(value, Success):
            try:
                return Success(self.value(value.value))
            except Exception as e:
                return Failure(e)
        failed: Failure[Any] = value
        return failed

    def is_success_and(self, function: Callable[[SuccessT], bool]) -> bool:
        return function(self.value)

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[SuccessT], U]) -> U:
        return function(self.value)

    def map_or_else(
        self,
        default_function: Callable[[Exception], U],
        function: Callable[[SuccessT], U],
    ) -> U:
        return function(self.value)

    def and_(self, other: Try[U]) -> Try[U]:
        return other

    def or_(self, other: Try[SuccessT]) -> Self:
        return self

    def zip(self, other: Try[U]) -> Try[tuple[SuccessT, U]]:
        if isinstance(other, Success):
            return Success((self.value, other.value))
        failed: Failure[Any] = other
        return failed

    def flatten(self: Success[Try[U]]) -> Try[U]:
        return self.value

    def unwrap(self) -> SuccessT:
        return self.value

    def unwrap_or(self, default_value: SuccessT) -> SuccessT:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, function: Callable[[Exception], SuccessT]) -> SuccessT:
        return self.value

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        return default_value

    def inspect(self, function: Callable[[SuccessT], None]) -> Self:
        function(self.value)
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
        return self

    def match(
        self, *, success: Callable[[SuccessT], U], failure: Callable[[Exception], U]
    ) -> U:
        return success(self.value)

    def to_either(self) -> Either[Exception, SuccessT]:
        return Right(self.value)

    def to_result(self) -> Result[SuccessT, Exception]:
        return Ok(self.value)

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"Success({self.value})"

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False, init=False)
class Failure(_TryImpl[SuccessT]):
    """The Failure variant of the Try Monad, wrapping the exception a computation raised.

    Two Failures are equal (and hash equally) if their exceptions have the same type and `args`.
    Hashing a Failure whose exception has unhashable `args` raises `TypeError`. `unwrap` re-raises
    the stored exception object itself, so each call adds frames to its `__traceback__`.

    Examples
    --------
    >>> Failure(ValueError("boom"))
    Failure(ValueError('boom'))
    >>> str(Failure(ValueError("boom")))
    'Failure(boom)'
    >>> Failure(ValueError("boom")) == Failure(ValueError("boom"))
    True
    >>> Failure(ValueError("boom")) == Failure(TypeError("boom"))
    False
    """

    exception: Exception

    def __init__(self, exception: Exception) -> None:
        if not isinstance(exception, Exception):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Failure value must be of type Exception")
        _set_failure_exception(self, exception)

    def map(self: Failure[Any], function: Callable[[SuccessT], U]) -> Failure[U]:
        return self

    def map_failure(
        self, function: Callable[[Exception], Exception]
    ) -> Failure[SuccessT]:
        try:
            return Failure(function(self.exception))
        except Exception as e:
            return Failure(e)

    def bind(self: Failure[Any], function: Callable[[SuccessT], Try[U]]) -> Failure[U]:
        return self

    and_then = bind

    def bind_failure(
        self, function: Callable[[Exception], Try[SuccessT]]
    ) -> Try[SuccessT]:
        try:
            return function(self.exception)
        except Exception as e:
            return Failure(e)

    or_else = bind_failure

    def filter(self, predicate: Callable[[SuccessT], bool]) -> Self:
        return self

    def apply(self: Failure[Any], function: Try[Callable[[SuccessT], U]]) -> Failure[U]:
        if isinstance(function, Failure):
            failed: Failure[Any] = function
            return failed
        return self

    def apply2(self: Failure[Callable[[U], V]], value: Try[U]) -> Failure[V]:
        failed: Failure[Any] = self
        return failed

    def is_success_and(self, function: Callable[[SuccessT], bool]) -> bool:
        return False

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        return function(self.exception)

    def map_or(self, default_value: U, function: Callable[[SuccessT], U]) -> U:
        return default_value

    def map_or_else(
        self,
        default_function: Callable[[Exception], U],
        function: Callable[[SuccessT], U],
    ) -> U:
        return default_function(self.exception)

    def and_(self: Failure[Any], other: Try[U]) -> Failure[U]:
        return self

    def or_(self, other: Try[U]) -> Try[U]:
        return other

    def zip(self: Failure[Any], other: Try[U]) -> Failure[tuple[SuccessT, U]]:
        return self

    def flatten(self: Failure[Try[U]]) -> Failure[U]:
        failed: Failure[Any] = self
        return failed

    def unwrap(self) -> Never:
        raise self.exception

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, function: Callable[[Exception], U]) -> U:
        return function(self.exception)

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        return self.exception

    def inspect(self, function: Callable[[SuccessT], None]) -> Self:
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
        function(self.exception)
        return self

    def match(
        self, *, success: Callable[[SuccessT], U], failure: Callable[[Exception], U]
    ) -> U:
        return failure(self.exception)

    def to_either(self) -> Either[Exception, SuccessT]:
        return Left(self.exception)

    def to_result(self) -> Result[SuccessT, Exception]:
        return Err(self.exception)

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Failure):
            return NotImplemented
        exception: Exception = other.exception
        return type(self.exception) is type(exception) and (
            self.exception.args == exception.args
        )

    def __hash__(self) -> int:
        return hash((type(self.exception), self.exception.args))

    def __str__(self) -> str:
        return f"Failure({self.exception})"

    def __repr__(self) -> str:
        return f"Failure({self.exception!r})"


type Try[T] = Success[T] | Failure[T]

# A frozen dataclass's generated __init__ assigns fields through object.__setattr__, which is slow.
# The variants' __init__ set their slot through the slot's member descriptor instead (about a third
# faster); the instances stay frozen, so assignment still raises FrozenInstanceError.
_set_success_value: Callable[[Success[Any], Any], None] = Success.__dict__[
    "value"
].__set__
_set_failure_exception: Callable[[Failure[Any], Exception], None] = Failure.__dict__[
    "exception"
].__set__
