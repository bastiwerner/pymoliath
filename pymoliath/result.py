"""
# Result Monad

The Result Monad is a container used to represent computations that can result in one of two
values: a success value or an error value. It encapsulates values that could be `Ok` (the
successful result) or `Err` (typically representing an error), allowing for a functional approach
to error handling by chaining operations without constant explicit error checks.

* Rust: [Result](https://doc.rust-lang.org/std/result/)
* Haskell: [Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)

This implementation is heavily inspired by the Rust `Result` type - see `pymoliath.either` for the
sibling implementation using Haskell's `Left`/`Right` naming.

The `Result` type is a sealed sum type that can be either `Ok` or `Err`. Like in Rust, both
variants carry both type parameters:

```python
type Result[T, E] = Ok[T, E] | Err[T, E]
```

## Typing like in Rust

Both type parameters are covariant and the side a variant does not use defaults to `Never`: a bare
`Ok(10)` is an `Ok[int, Never]` and a bare `Err("e")` is an `Err[Never, str]`. Both are assignable
to a `Result[int, str]`, and a conditional lambda such as
`lambda x: Ok(x) if x > 0 else Err("negative")` is inferred as a `Result[int, str]`.

Like in Rust, the error type is fixed by the receiver: `bind` on a `Result[int, str]` needs a
function returning a `Result[U, str]`, and a bare `Ok(10)` (error type `Never`) needs an
annotation before it can be bound to fallible code:

```python
value: Result[int, str] = Ok(10)

def parse(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err("not a number")
```

Methods called directly on a variant keep the precise variant type, e.g. `Ok(1).map(str)` is an
`Ok[str, Never]` and `Err("e").map(str)` an `Err[str, str]`.

## Practical Examples and Benefits:

The Result Monad is particularly useful in scenarios where a function might fail and return an
error (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database
that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without
   checking `if error` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Err`, the subsequent operations
   are skipped automatically, and the final result will be `Err`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly,
   making the code more robust against unhandled exceptions and making the flow of data more
   transparent.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if"/try-except logic into a linear pipeline of transformations.

```python
# Without Result (Imperative)
try:
    user = get_user(user_id)
    profile = get_profile(user)
    permission = get_permission(profile)
    print(permission)
except UserNotFoundError:
    print("User not found")
except ProfileNotFoundError:
    print("Profile not found")
except PermissionDeniedError:
    print("Access denied")

# With Result (Functional)
(get_user(user_id)
    .bind(get_profile)
    .bind(get_permission)
    .unwrap_or_else(lambda error: f"Error: {error}"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Result`
monad. `Ok` and `Err` are the only variants (both are `@final`), so a `match` over both is
exhaustive and type checkers narrow the value in each branch.

```python
match result_value:
    case Ok(x):
        # This block executes if the operation was successful
        print(f"Success value: {x}")
    case Err(y):
        # This block executes if the operation failed
        print(f"Error encountered: {y}")
```

`Result` is a type alias, so use `is_result` (or `isinstance(x, RESULT_TYPES)`) for runtime checks
and `is_ok`/`is_err` to narrow a `Result` to one of its variants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, Self, final, overload

from typing_extensions import Generic, TypeIs, TypeVar

from pymoliath.errors import UnwrapError

# Both parameters are covariant, so `Ok[int, Never]` is a `Result[int, str]`. Like in Rust the
# receiver fixes the types a method accepts (`unwrap_or(default: T)`, `bind` returning
# `Result[U, E]`), which puts a covariant parameter in an input position. That is sound here: the
# containers are immutable and those arguments are only ever returned, typed by the receiver's
# (wider) view - the same reasoning as typeshed's `Sequence.index`. Those methods carry a
# `# type: ignore[misc]`.
T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)

# Variant parameters: the side a variant does not use defaults to Never (PEP 696).
T_Never = TypeVar("T_Never", covariant=True, default=Never)
E_Never = TypeVar("E_Never", covariant=True, default=Never)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Y = TypeVar("Y")
F = TypeVar("F")
X = TypeVar("X", bound=BaseException)


class _ResultImpl(Generic[T, E]):
    """Public interface of the Result Monad. The behaviour lives in `Ok` and `Err`, its only
    subclasses."""

    __slots__ = ()

    def map(self, function: Callable[[T], U]) -> Result[U, E]:
        """Calls function on a wrapped Ok value, otherwise leaving the Err value untouched.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        result: Result[U, E]
            Returns an Ok with the function result or otherwise the Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map(lambda x: x + 1)
        Ok(6)
        >>> err: Result[int, str] = Err("error")
        >>> err.map(lambda x: x + 1)
        Err('error')
        """
        raise NotImplementedError

    def map_err(self, function: Callable[[E], F]) -> Result[T, F]:
        """Calls function on a wrapped Err value, otherwise leaving the Ok value untouched.

        Parameters
        ----------
        function: Callable[[E], F]
            Function which takes a value of E and returns a value of F.

        Returns
        -------
        result: Result[T, F]
            Returns an Err with the function result or otherwise the Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map_err(lambda e: e.upper())
        Ok(5)
        >>> err: Result[int, str] = Err("error")
        >>> err.map_err(lambda e: e.upper())
        Err('ERROR')
        """
        raise NotImplementedError

    def bind(self, function: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Calls function if the Result Monad is Ok, otherwise returns the Err (Rust: `and_then`).

        Parameters
        ----------
        function: Callable[[T], Result[U, E]]
            Function which takes a value of T and returns a new Result Monad.

        Returns
        -------
        result: Result[U, E]
            Returns the function result if Ok, otherwise the Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> def get_next(x: int) -> Result[int, str]:
        ...     return Ok(x + 1) if x < 10 else Err("too big")
        >>> val.bind(get_next)
        Ok(6)
        >>> big: Result[int, str] = Ok(10)
        >>> big.bind(get_next)
        Err('too big')
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias of `bind`, named like in Rust.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.and_then(lambda x: Ok(x * 2))
        Ok(10)
        """
        raise NotImplementedError

    def bind_err(self, function: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Calls function if the Result Monad is Err, otherwise returns the Ok (Rust: `or_else`).

        Parameters
        ----------
        function: Callable[[E], Result[T, F]]
            Function which takes a value of E and returns a new Result Monad.

        Returns
        -------
        result: Result[T, F]
            Returns the function result if Err, otherwise the Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.bind_err(lambda e: Ok(0))
        Ok(5)
        >>> err: Result[int, str] = Err("error")
        >>> err.bind_err(lambda e: Ok(len(e)))
        Ok(5)
        """
        raise NotImplementedError

    def or_else(self, function: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Alias of `bind_err`, named like in Rust.

        Examples
        --------
        >>> err: Result[int, str] = Err("error")
        >>> err.or_else(lambda e: Ok(len(e)))
        Ok(5)
        """
        raise NotImplementedError

    def apply(self, function: Result[Callable[[T], U], E]) -> Result[U, E]:
        """Applies the function wrapped in `function` to the Ok value if both are Ok.

        If both are Err, the Err of `function` takes precedence. For functions of several
        arguments, use `map2`/`map3`.

        Parameters
        ----------
        function: Result[Callable[[T], U], E]
            Result Monad which contains a function.

        Returns
        -------
        result: Result[U, E]
            Returns Ok of the function result if both are Ok, otherwise the (first) Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> func: Result[Callable[[int], int], str] = Ok(lambda x: x * 2)
        >>> val.apply(func)
        Ok(20)
        >>> Err("value").apply(Err("function"))
        Err('function')
        """
        raise NotImplementedError

    def apply2(
        self: _ResultImpl[Callable[[U], V], E], value: Result[U, E]
    ) -> Result[V, E]:
        """Applies the function wrapped in this Result Monad to the value wrapped in `value`.

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both are
        empty/errors, this (the function side) takes precedence. Functions of several
        arguments can be applied one argument at a time when they are curried, e.g.
        `Ok(lambda a: lambda b: a + b).apply2(x).apply2(y)`; `map2`/`map3` take them uncurried.

        Parameters
        ----------
        value: Result[U, E]
            Result Monad which contains the argument.

        Returns
        -------
        result: Result[V, E]

        Examples
        --------
        >>> func: Result[Callable[[int], int], str] = Ok(lambda y: 10 + y)
        >>> val: Result[int, str] = Ok(5)
        >>> func.apply2(val)
        Ok(15)
        >>> add: Result[Callable[[int], Callable[[int], int]], str] = Ok(lambda a: lambda b: a + b)
        >>> add.apply2(Ok(1)).apply2(Ok(2))
        Ok(3)
        """
        raise NotImplementedError

    def is_ok_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Result Monad is Ok and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Ok value.

        Returns
        -------
        result: bool
            Returns the predicate result if Ok, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_ok_and(lambda x: x > 5)
        True
        """
        raise NotImplementedError

    def is_err_and(self, function: Callable[[E], bool]) -> bool:
        """Returns True if the Result Monad is Err and the predicate returns True for the error.

        Parameters
        ----------
        function: Callable[[E], bool]
            Predicate function applied to the Err value.

        Returns
        -------
        result: bool
            Returns the predicate result if Err, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_err_and(lambda e: True)
        False
        """
        raise NotImplementedError

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Ok value, or returns the default value if Err.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Result Monad is Err.
        function: Callable[[T], U]
            Function applied to the Ok value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[E], U], function: Callable[[T], U]
    ) -> U:
        """Applies the function to the Ok value, or the default function to the Err value.

        Parameters
        ----------
        default_function: Callable[[E], U]
            Function applied to the Err value.
        function: Callable[[T], U]
            Function applied to the Ok value.

        Returns
        -------
        result: U
            Returns the result of whichever function was applied.

        Examples
        --------
        >>> err: Result[int, str] = Err("error")
        >>> err.map_or_else(lambda e: len(e), lambda x: x * 2)
        5
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool], error: E) -> Result[T, E]:  # type: ignore[misc]
        """Keeps the Ok value if the predicate holds for it, otherwise returns Err(error).

        Parameters
        ----------
        predicate: Callable[[T], bool]
            Predicate function applied to the Ok value.
        error: E
            Error used if the predicate does not hold.

        Returns
        -------
        result: Result[T, E]
            Returns the Ok, Err(error) if the predicate fails, or the original Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(-1)
        >>> val.filter(lambda x: x > 0, "negative")
        Err('negative')
        """
        raise NotImplementedError

    def and_(self, other: Result[U, E]) -> Result[U, E]:
        """Returns `other` if the Result Monad is Ok, otherwise the Err.

        Parameters
        ----------
        other: Result[U, E]
            Result Monad to be returned if this Result Monad is Ok.

        Returns
        -------
        result: Result[U, E]
            Returns `other` or the Err.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.and_(val2)
        Ok(2)
        """
        raise NotImplementedError

    def or_(self, other: Result[T, F]) -> Result[T, F]:
        """Returns this Result Monad if it is Ok, otherwise `other`.

        Parameters
        ----------
        other: Result[T, F]
            Result Monad to be returned if this Result Monad is Err.

        Returns
        -------
        result: Result[T, F]
            Returns the Ok or `other`.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.or_(val2)
        Ok(1)
        """
        raise NotImplementedError

    def zip(self, other: Result[U, E]) -> Result[tuple[T, U], E]:
        """Combines this Result Monad with another into a Result Monad of a tuple, or the first Err.

        Parameters
        ----------
        other: Result[U, E]
            Result Monad to be zipped with this Result Monad.

        Returns
        -------
        result: Result[tuple[T, U], E]
            Returns Ok of a tuple of both values, or Err.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.zip(val2)
        Ok((1, 2))
        """
        raise NotImplementedError

    def swap(self) -> Result[E, T]:
        """Swaps the Ok and the Err value.

        Returns
        -------
        result: Result[E, T]
            Returns Err of the Ok value, or Ok of the Err value.

        Examples
        --------
        >>> Ok(1).swap()
        Err(1)
        >>> Err("e").swap()
        Ok('e')
        """
        raise NotImplementedError

    def flatten(self: _ResultImpl[Result[U, F], F]) -> Result[U, F]:
        """Flattens a nested Result Monad by one level.

        Returns
        -------
        result: Result[U, F]
            Returns the nested Result Monad if Ok, otherwise the Err.

        Examples
        --------
        >>> Ok(Ok(1)).flatten()
        Ok(1)
        """
        raise NotImplementedError

    def merge(self: _ResultImpl[U, U]) -> U:
        """Returns the Ok or the Err value, for a Result whose both sides have the same type.

        Returns
        -------
        value: U
            The wrapped value.

        Examples
        --------
        >>> Err(1).merge()
        1
        """
        raise NotImplementedError

    def transpose(self: _ResultImpl[Option[U], F]) -> Option[Result[U, F]]:
        """Transposes a Result of an Option into an Option of a Result.

        `Ok(Nil())` becomes `Nil()`, `Ok(Some(x))` becomes `Some(Ok(x))` and `Err(e)` becomes
        `Some(Err(e))`.

        Examples
        --------
        >>> from pymoliath.option import Some
        >>> Ok(Some(1)).transpose()
        Some(Ok(1))
        """
        raise NotImplementedError

    def ok(self) -> Option[T]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[T]
            Returns Some with the Ok value, or Nil if Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.ok()
        Some(1)
        """
        raise NotImplementedError

    def err(self) -> Option[E]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding any Ok value.

        Returns
        -------
        option: Option[E]
            Returns Some with the Err value, or Nil if Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.err()
        Nil()
        """
        raise NotImplementedError

    def unwrap(self) -> T:
        """Returns the Ok value, or otherwise raises an `UnwrapError`.

        If the Err value is an exception, it is chained as the `__cause__` of the `UnwrapError`.

        Returns
        -------
        result: T
            Returns the Ok value.

        Raises
        ------
        UnwrapError
            If the Result Monad is Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap()
        1
        """
        raise NotImplementedError

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        """Returns the Ok value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: T
            Default value of T

        Returns
        -------
        result: T
            Returns the Ok value or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or(0)
        1
        """
        raise NotImplementedError

    def unwrap_or_else(self, err_function: Callable[[E], T]) -> T:
        """Returns the Ok value, or otherwise calls the err_function with the Err value.

        Parameters
        ----------
        err_function: Callable[[E], T]
            Error function which will be called if the result is of type Err.

        Returns
        -------
        result: T
            Returns the Ok value or the err_function result.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or_else(lambda e: 0)
        1
        """
        raise NotImplementedError

    def unwrap_err_or(self, default_value: E) -> E:  # type: ignore[misc]
        """Returns the Err value, or otherwise a provided default value of E.

        Parameters
        ----------
        default_value: E
            Default value of E

        Returns
        -------
        result: E
            Returns the Err value or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_err_or("default")
        'default'
        """
        raise NotImplementedError

    def inspect(self, function: Callable[[T], None]) -> Self:
        """Calls function with the Ok value (if any) and returns the Result Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Ok value of the Result monad

        Returns
        -------
        result: Result[T, E]

        Examples
        --------
        >>> val: Result[int, str] = Ok(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Ok(42)
        """
        raise NotImplementedError

    def inspect_err(self, function: Callable[[E], None]) -> Self:
        """Calls function with the Err value (if any) and returns the Result Monad unchanged.

        Parameters
        ----------
        function: Callable[[E], None]
            Inspection function which takes the Err value of the Result monad

        Returns
        -------
        result: Result[T, E]

        Examples
        --------
        >>> val: Result[int, str] = Err("boom")
        >>> val.inspect_err(lambda e: print(f"Error: {e}"))
        Error: boom
        Err('boom')
        """
        raise NotImplementedError

    def match(self, *, ok: Callable[[T], U], err: Callable[[E], U]) -> U:
        """Matches the Result Monad to either the `ok` or the `err` callback with the same return
        type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        ok: Callable[[T], U]
            Callback function for Result monads of type Ok
        err: Callable[[E], U]
            Callback function for Result monads of type Err

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.match(ok=lambda x: f"Ok: {x}", err=lambda e: "Error")
        'Ok: 10'
        """
        raise NotImplementedError

    def is_ok(self) -> bool:
        """Returns True if the Result Monad is Ok, otherwise False. Use the module-level `is_ok`
        to narrow the type.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_ok()
        True
        """
        raise NotImplementedError

    def is_err(self) -> bool:
        """Returns True if the Result Monad is Err, otherwise False. Use the module-level `is_err`
        to narrow the type.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_err()
        False
        """
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True, repr=False)
class Ok(_ResultImpl[T, E_Never]):
    """The Ok variant of the Result Monad, wrapping a success value.

    Equality and hashing compare the wrapped value (an unhashable value makes the Ok unhashable).

    Examples
    --------
    >>> Ok(42)
    Ok(42)
    >>> str(Ok("text"))
    'Ok(text)'
    """

    value: T

    def map(self, function: Callable[[T], U]) -> Ok[U, E_Never]:
        return Ok(function(self.value))

    def map_err(self, function: Callable[[E_Never], F]) -> Ok[T, F]:
        return self  # type: ignore[return-value]

    def bind(self, function: Callable[[T], Result[U, E_Never]]) -> Result[U, E_Never]:
        return function(self.value)

    and_then = bind

    def bind_err(self, function: Callable[[E_Never], Result[T, F]]) -> Ok[T, F]:
        return self  # type: ignore[return-value]

    or_else = bind_err

    def apply(self, function: Result[Callable[[T], U], E_Never]) -> Result[U, E_Never]:
        if isinstance(function, Ok):
            return Ok(function.value(self.value))
        return function  # type: ignore[return-value]

    def apply2(
        self: Ok[Callable[[U], V], E_Never], value: Result[U, E_Never]
    ) -> Result[V, E_Never]:
        if isinstance(value, Ok):
            return Ok(self.value(value.value))
        return value  # type: ignore[return-value]

    def is_ok_and(self, function: Callable[[T], bool]) -> bool:
        return function(self.value)

    def is_err_and(self, function: Callable[[E_Never], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[E_Never], U], function: Callable[[T], U]
    ) -> U:
        return function(self.value)

    def filter(
        self,
        predicate: Callable[[T], bool],
        error: E_Never,  # type: ignore[misc]
    ) -> Result[T, E_Never]:
        return self if predicate(self.value) else Err(error)

    def and_(self, other: Result[U, E_Never]) -> Result[U, E_Never]:
        return other

    def or_(self, other: Result[T, F]) -> Ok[T, F]:
        return self  # type: ignore[return-value]

    def zip(self, other: Result[U, E_Never]) -> Result[tuple[T, U], E_Never]:
        if isinstance(other, Ok):
            return Ok((self.value, other.value))
        return other  # type: ignore[return-value]

    def swap(self) -> Err[E_Never, T]:
        return Err(self.value)

    def flatten(self: Ok[Result[U, F], F]) -> Result[U, F]:
        return self.value

    def merge(self: Ok[U, U]) -> U:
        return self.value

    def transpose(self: Ok[Option[U], F]) -> Option[Result[U, F]]:
        option = self.value
        if isinstance(option, Some):
            return Some(Ok(option.value))
        return option

    def ok(self) -> Some[T]:
        return Some(self.value)

    def err(self) -> Nil:
        return Nil()

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, err_function: Callable[[E_Never], T]) -> T:
        return self.value

    def unwrap_err_or(self, default_value: E_Never) -> E_Never:  # type: ignore[misc]
        return default_value

    def inspect(self, function: Callable[[T], None]) -> Self:
        function(self.value)
        return self

    def inspect_err(self, function: Callable[[E_Never], None]) -> Self:
        return self

    def match(self, *, ok: Callable[[T], U], err: Callable[[E_Never], U]) -> U:
        return ok(self.value)

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"Ok({self.value})"

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False)
class Err(_ResultImpl[T_Never, E_Never]):
    """The Err variant of the Result Monad, wrapping an error value.

    Equality and hashing compare the wrapped error. Exceptions compare by identity, so two
    separately raised but otherwise identical exceptions are not equal.

    Examples
    --------
    >>> Err("error")
    Err('error')
    >>> str(Err("error"))
    'Err(error)'
    """

    error: E_Never

    def map(self, function: Callable[[T_Never], U]) -> Err[U, E_Never]:
        return self  # type: ignore[return-value]

    def map_err(self, function: Callable[[E_Never], F]) -> Err[T_Never, F]:
        return Err(function(self.error))

    def bind(
        self, function: Callable[[T_Never], Result[U, E_Never]]
    ) -> Err[U, E_Never]:
        return self  # type: ignore[return-value]

    and_then = bind

    def bind_err(
        self, function: Callable[[E_Never], Result[T_Never, F]]
    ) -> Result[T_Never, F]:
        return function(self.error)

    or_else = bind_err

    def apply(
        self, function: Result[Callable[[T_Never], U], E_Never]
    ) -> Err[U, E_Never]:
        if isinstance(function, Err):
            return function  # type: ignore[return-value]
        return self  # type: ignore[return-value]

    def apply2(
        self: Err[Callable[[U], V], E_Never], value: Result[U, E_Never]
    ) -> Err[V, E_Never]:
        return self  # type: ignore[return-value]

    def is_ok_and(self, function: Callable[[T_Never], bool]) -> bool:
        return False

    def is_err_and(self, function: Callable[[E_Never], bool]) -> bool:
        return function(self.error)

    def map_or(self, default_value: U, function: Callable[[T_Never], U]) -> U:
        return default_value

    def map_or_else(
        self, default_function: Callable[[E_Never], U], function: Callable[[T_Never], U]
    ) -> U:
        return default_function(self.error)

    def filter(self, predicate: Callable[[T_Never], bool], error: E_Never) -> Self:  # type: ignore[misc]
        return self

    def and_(self, other: Result[U, E_Never]) -> Err[U, E_Never]:
        return self  # type: ignore[return-value]

    def or_(self, other: Result[T_Never, F]) -> Result[T_Never, F]:
        return other

    def zip(self, other: Result[U, E_Never]) -> Err[tuple[T_Never, U], E_Never]:
        return self  # type: ignore[return-value]

    def swap(self) -> Ok[E_Never, T_Never]:
        return Ok(self.error)

    def flatten(self: Err[Result[U, F], F]) -> Err[U, F]:
        return self  # type: ignore[return-value]

    def merge(self: Err[U, U]) -> U:
        return self.error

    def transpose(self: Err[Option[U], F]) -> Some[Result[U, F]]:
        return Some(self)  # type: ignore[arg-type]

    def ok(self) -> Nil:
        return Nil()

    def err(self) -> Some[E_Never]:
        return Some(self.error)

    def unwrap(self) -> Never:
        error = self.error
        message = f"called unwrap on {self!r}"
        if isinstance(error, BaseException):
            raise UnwrapError(self, message) from error
        raise UnwrapError(self, message)

    def unwrap_or(self, default_value: T_Never) -> T_Never:  # type: ignore[misc]
        return default_value

    def unwrap_or_else(self, err_function: Callable[[E_Never], T_Never]) -> T_Never:
        return err_function(self.error)

    def unwrap_err_or(self, default_value: E_Never) -> E_Never:  # type: ignore[misc]
        return self.error

    def inspect(self, function: Callable[[T_Never], None]) -> Self:
        return self

    def inspect_err(self, function: Callable[[E_Never], None]) -> Self:
        function(self.error)
        return self

    def match(self, *, ok: Callable[[T_Never], U], err: Callable[[E_Never], U]) -> U:
        return err(self.error)

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"Err({self.error})"

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


type Result[OkT, ErrT] = Ok[OkT, ErrT] | Err[OkT, ErrT]

RESULT_TYPES: tuple[type[Ok[Any, Any]], type[Err[Any, Any]]] = (Ok, Err)
"""The runtime classes of `Result`, for `isinstance` checks (`Result` itself is a type alias)."""


def is_result(value: object) -> TypeIs[Result[Any, Any]]:
    """Returns True if `value` is an Ok or an Err.

    Examples
    --------
    >>> is_result(Ok(1)), is_result(1)
    (True, False)
    """
    return isinstance(value, RESULT_TYPES)


def is_ok(result: Result[U, F]) -> TypeIs[Ok[U, F]]:
    """Returns True if the Result Monad is Ok, narrowing it to `Ok` for type checkers.

    Examples
    --------
    >>> val: Result[int, str] = Ok(1)
    >>> if is_ok(val):
    ...     print(val.value)
    1
    """
    return isinstance(result, Ok)


def is_err(result: Result[U, F]) -> TypeIs[Err[U, F]]:
    """Returns True if the Result Monad is Err, narrowing it to `Err` for type checkers.

    Examples
    --------
    >>> val: Result[int, str] = Err("e")
    >>> if is_err(val):
    ...     print(val.error)
    e
    """
    return isinstance(result, Err)


def from_option(option: Option[U], error: F) -> Result[U, F]:
    """Converts an Option Monad into a Result Monad: Some(x) becomes Ok(x), Nil becomes Err(error).

    Examples
    --------
    >>> from pymoliath.option import Nil, Some
    >>> from_option(Some(1), "missing")
    Ok(1)
    >>> from_option(Nil(), "missing")
    Err('missing')
    """
    if isinstance(option, Some):
        return Ok(option.value)
    return Err(error)


def map2(
    first: Result[U, F], second: Result[V, F], function: Callable[[U, V], W]
) -> Result[W, F]:
    """Applies a two-argument function to the values of two Result Monads if both are Ok.

    If both are Err, the first Err takes precedence.

    Examples
    --------
    >>> map2(Ok(1), Ok(2), lambda a, b: a + b)
    Ok(3)
    >>> map2(Err("first"), Err("second"), lambda a, b: a + b)
    Err('first')
    """
    if isinstance(first, Err):
        return first  # type: ignore[return-value]
    if isinstance(second, Err):
        return second  # type: ignore[return-value]
    return Ok(function(first.value, second.value))


def map3(
    first: Result[U, F],
    second: Result[V, F],
    third: Result[W, F],
    function: Callable[[U, V, W], Y],
) -> Result[Y, F]:
    """Applies a three-argument function to the values of three Result Monads if all are Ok.

    If several are Err, the first Err takes precedence.

    Examples
    --------
    >>> map3(Ok(1), Ok(2), Ok(3), lambda a, b, c: a + b + c)
    Ok(6)
    """
    if isinstance(first, Err):
        return first  # type: ignore[return-value]
    if isinstance(second, Err):
        return second  # type: ignore[return-value]
    if isinstance(third, Err):
        return third  # type: ignore[return-value]
    return Ok(function(first.value, second.value, third.value))


@overload
def result_safe(function: Callable[[], U]) -> Result[U, Exception]: ...


@overload
def result_safe(
    function: Callable[[], U], *, exceptions: tuple[type[X], ...]
) -> Result[U, X]: ...


def result_safe(
    function: Callable[[], U],
    *,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Result[U, BaseException]:
    """Calls function and wraps its return value in Ok, or a raised exception in Err.

    Parameters
    ----------
    function: Callable[[], U]
        Zero-argument function which may raise an exception.
    exceptions: tuple[type[X], ...]
        The exception types to catch (default: `Exception`). Any other exception propagates.

    Returns
    -------
    result: Result[U, X]

    Examples
    --------
    >>> result_safe(lambda: 1)
    Ok(1)
    >>> result_safe(lambda: 1 / 0)
    Err(ZeroDivisionError('division by zero'))
    >>> result_safe(lambda: int("x"), exceptions=(ValueError,)).is_err()
    True
    """
    try:
        return Ok(function())
    except exceptions as e:
        return Err(e)


# Imported last: option.py imports Ok/Err from this module, so the cycle resolves once at import
# time instead of on every call.
from pymoliath.option import Nil, Option, Some  # noqa: E402
