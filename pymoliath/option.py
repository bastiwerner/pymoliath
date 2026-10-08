"""
# Option Monad

The Option Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Rust: [Option](https://doc.rust-lang.org/std/option/)
* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

This implementation is heavily inspired by the Rust `Option` type and the Haskell `Maybe` monad -
see `pymoliath.maybe` for the sibling implementation using `Just`/`Nothing` naming.

The `Option` type is a sealed sum type that can be either `Some` or `Nil`:

```python
type Option[T] = Some[T] | Nil
```

## Typing like in Rust

The type parameter is covariant and `Nil` is a singleton of type `Option[Never]`, so a bare `Nil()`
is assignable to every `Option[T]` and a conditional lambda such as
`lambda x: Some(x) if x > 0 else Nil()` is inferred as an `Option[int]`:

```python
empty: Option[int] = Nil()

def find(value: int) -> Option[int]:
    return Some(value) if value > 0 else Nil()
```

Methods called directly on a variant keep the precise variant type, e.g. `Some(1).map(str)` is a
`Some[str]` and `Nil().map(str)` is `Nil`.

## Practical Examples and Benefits:

The Option Monad is particularly useful in scenarios where a function might not return a value (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without checking `if result is None` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Nil`, the subsequent operations are skipped automatically, and the final result will be `Nil`.
3. Type Safety: It forces the developer to acknowledge the possibility of "nothingness" explicitly, making the code more robust against `AttributeError: 'NoneType' object has no attribute...`.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if" logic into a linear pipeline of transformations.

```python
# Without Option (Imperative)
user = get_user(user_id)
if user:
    profile = get_profile(user)
    if profile:
        permission = get_permission(profile)
        if permission:
            print(permission)

# With Option (Functional)
(get_user(user_id)
    .bind(get_profile)
    .bind(get_permission)
    .unwrap_or("Default Permission"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of an
`Option` monad. `Some` and `Nil` are the only variants (both are `@final`), so a `match` over
both is exhaustive and type checkers narrow the value in each branch.

```python
match option_value:
    case Some(x):
        # This block executes if the monad contains a value
        print(f"Some value: {x}")
    case Nil():
        # This block executes if the monad is Nil
        print("No value present")
```

`Option` is a type alias, so check it at runtime with `isinstance(x, (Some, Nil))`. Both a `match`
over `Some`/`Nil` and `isinstance(x, Some)` narrow the type to the variant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, Self, final, overload

from typing_extensions import Generic, TypeVar

# option.py and result.py convert into each other. Importing the module (not its names) lets the
# cycle resolve at import time; its attributes are looked up when the conversions run.
import pymoliath.result as _result
from pymoliath.errors import UnwrapError

if TYPE_CHECKING:
    from pymoliath.result import Err, Ok, Result

# Covariant, so `Some[bool]` is an `Option[int]` and `Nil` (an `Option[Never]`) is every `Option[T]`.
# Like in Rust the receiver fixes the types some methods accept (`unwrap_or(default: T)`,
# `or_(other: Option[T])`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `type: ignore` (misc).
T = TypeVar("T", covariant=True)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")


class _OptionImpl(Generic[T]):
    """Public interface of the Option Monad. The behaviour lives in `Some` and `Nil`, its only
    subclasses."""

    __slots__ = ()

    def map(self, function: Callable[[T], U]) -> Option[U]:
        """Calls function on a wrapped Some value, otherwise returns Nil.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        option: Option[U]
            Returns a Some with the function result or otherwise Nil.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.map(lambda x: x + 1)
        Some(6)
        >>> empty: Option[int] = Nil()
        >>> empty.map(lambda x: x + 1)
        Nil()
        """
        raise NotImplementedError

    def bind(self, function: Callable[[T], Option[U]]) -> Option[U]:
        """Calls function if the Option Monad is Some, otherwise returns Nil (Rust: `and_then`).

        Parameters
        ----------
        function: Callable[[T], Option[U]]
            Function which takes a value of T and returns a new Option Monad.

        Returns
        -------
        option: Option[U]
            Returns the function result if Some, otherwise Nil.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.bind(lambda x: Some(x * 2) if x > 0 else Nil())
        Some(10)
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[T], Option[U]]) -> Option[U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[T], Option[U]]
            Function which takes a value of T and returns a new Option Monad.

        Returns
        -------
        option: Option[U]
            Returns the function result if Some, otherwise Nil.

        Examples
        --------
        >>> Some(5).and_then(lambda x: Some(x * 2))
        Some(10)
        """
        raise NotImplementedError

    def or_else(self, function: Callable[[], Option[T]]) -> Option[T]:
        """Returns this Option Monad if it is Some, otherwise the result of function.

        Parameters
        ----------
        function: Callable[[], Option[T]]
            Function computing the fallback Option Monad.

        Returns
        -------
        option: Option[T]
            Returns the Some, or the function result if Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.or_else(lambda: Some(1))
        Some(1)
        """
        raise NotImplementedError

    def apply(self, function: Option[Callable[[T], U]]) -> Option[U]:
        """Applies the function wrapped in `function` to the Some value if both are Some,
        otherwise returns Nil. For functions of several
        arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: Option[Callable[[T], U]]
            Option Monad which contains a function.

        Returns
        -------
        option: Option[U]
            Returns Some of the function result if both are Some, otherwise Nil.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> func: Option[Callable[[int], int]] = Some(lambda x: x * 2)
        >>> val.apply(func)
        Some(20)
        """
        raise NotImplementedError

    def apply2(self: _OptionImpl[Callable[[U], V]], value: Option[U]) -> Option[V]:
        """Applies the function wrapped in this Option Monad to the value wrapped in `value`.

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both are
        empty/errors, this (the function side) takes precedence. Functions of several
        arguments can be applied one argument at a time when they are curried, e.g.
        `Some(lambda a: lambda b: a + b).apply2(x).apply2(y)`.

        Parameters
        ----------
        value: Option[U]
            Option Monad which contains the argument.

        Returns
        -------
        option: Option[V]
            Returns Some of the function result if both are Some, otherwise Nil.

        Examples
        --------
        >>> func: Option[Callable[[int], int]] = Some(lambda y: 10 + y)
        >>> func.apply2(Some(5))
        Some(15)
        >>> func.apply2(Nil())
        Nil()
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Returns the Option Monad if it is Some and the predicate returns True, otherwise Nil.

        Parameters
        ----------
        predicate: Callable[[T], bool]
            Predicate function applied to the Some value.

        Returns
        -------
        option: Option[T]
            Returns the Some if the predicate holds, otherwise Nil.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.filter(lambda x: x > 5)
        Some(10)
        >>> val.filter(lambda x: x > 10)
        Nil()
        """
        raise NotImplementedError

    def is_some_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Option Monad is Some and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Some value.

        Returns
        -------
        result: bool
            Returns the predicate result if Some, otherwise False.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.is_some_and(lambda x: x > 5)
        True
        """
        raise NotImplementedError

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Some value, or returns the default value if Nil.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Option Monad is Nil.
        function: Callable[[T], U]
            Function applied to the Some value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[T], U]
    ) -> U:
        """Applies the function to the Some value, or returns the default function's result if Nil.

        Parameters
        ----------
        default_function: Callable[[], U]
            Function computing the result if the Option Monad is Nil.
        function: Callable[[T], U]
            Function applied to the Some value.

        Returns
        -------
        result: U
            Returns the function result or the default function's result.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.map_or_else(lambda: 0, lambda x: x * 2)
        0
        """
        raise NotImplementedError

    def and_(self, other: Option[U]) -> Option[U]:
        """Returns `other` if the Option Monad is Some, otherwise Nil.

        Parameters
        ----------
        other: Option[U]
            Option Monad to be returned if this Option Monad is Some.

        Returns
        -------
        option: Option[U]
            Returns `other` or Nil.

        Examples
        --------
        >>> val1: Option[int] = Some(1)
        >>> val2: Option[int] = Some(2)
        >>> val1.and_(val2)
        Some(2)
        """
        raise NotImplementedError

    def or_(self, other: Option[T]) -> Option[T]:
        """Returns this Option Monad if it is Some, otherwise `other`.

        Parameters
        ----------
        other: Option[T]
            Option Monad to be returned if this Option Monad is Nil.

        Returns
        -------
        option: Option[T]
            Returns the Some or `other`.

        Examples
        --------
        >>> val1: Option[int] = Nil()
        >>> val2: Option[int] = Some(2)
        >>> val1.or_(val2)
        Some(2)
        """
        raise NotImplementedError

    def xor(self, other: Option[T]) -> Option[T]:
        """Returns the Some if exactly one of this Option Monad and `other` is Some, otherwise Nil.

        Parameters
        ----------
        other: Option[T]
            Option Monad to be compared with this Option Monad.

        Returns
        -------
        option: Option[T]
            Returns the only Some, or Nil if both or neither are Some.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.xor(Nil())
        Some(1)
        >>> val.xor(Some(2))
        Nil()
        """
        raise NotImplementedError

    def zip(self, other: Option[U]) -> Option[tuple[T, U]]:
        """Combines this Option Monad with another into an Option Monad of a tuple, or Nil if
        either is Nil.

        Parameters
        ----------
        other: Option[U]
            Option Monad to be zipped with this Option Monad.

        Returns
        -------
        option: Option[tuple[T, U]]
            Returns Some of a tuple of both values, or Nil.

        Examples
        --------
        >>> val1: Option[int] = Some(1)
        >>> val2: Option[str] = Some("a")
        >>> val1.zip(val2)
        Some((1, 'a'))
        """
        raise NotImplementedError

    def flatten(self: _OptionImpl[Option[U]]) -> Option[U]:
        """Flattens a nested Option Monad by one level.

        Returns
        -------
        option: Option[U]
            Returns the nested Option Monad if Some, otherwise Nil.

        Examples
        --------
        >>> Some(Some(1)).flatten()
        Some(1)
        """
        raise NotImplementedError

    def transpose(
        self: _OptionImpl[Result[U, E]],
    ) -> Result[Option[U], E]:
        """Transposes an Option of a Result into a Result of an Option.

        `Nil()` becomes `Ok(Nil())`, `Some(Ok(x))` becomes `Ok(Some(x))` and `Some(Err(e))` becomes
        `Err(e)`.

        Returns
        -------
        result: Result[Option[U], E]
            Returns the transposed Result Monad.

        Examples
        --------
        >>> from pymoliath.result import Ok
        >>> Some(Ok(1)).transpose()
        Ok(Some(1))
        """
        raise NotImplementedError

    def ok_or(self, error: E) -> Result[T, E]:
        """Converts the Option Monad into a Result Monad, mapping Some(v) to Ok(v) and Nil to
        Err(error).

        Parameters
        ----------
        error: E
            Error value used if the Option Monad is Nil.

        Returns
        -------
        result: Result[T, E]
            Returns Ok with the Some value, or Err with the error value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.ok_or("missing")
        Ok(1)
        """
        raise NotImplementedError

    def ok_or_else(self, function: Callable[[], E]) -> Result[T, E]:
        """Converts the Option Monad into a Result Monad, mapping Some(v) to Ok(v) and Nil to
        Err(function()).

        Parameters
        ----------
        function: Callable[[], E]
            Function computing the error value if the Option Monad is Nil.

        Returns
        -------
        result: Result[T, E]
            Returns Ok with the Some value, or Err with the function result.

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.ok_or_else(lambda: "missing")
        Err('missing')
        """
        raise NotImplementedError

    def unwrap(self) -> T:
        """Returns the Some value, or otherwise raises an `UnwrapError`.

        Returns
        -------
        result: T
            Returns the Some value.

        Raises
        ------
        UnwrapError
            If the Option Monad is Nil.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.unwrap()
        1
        """
        raise NotImplementedError

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        """Returns the Some value, or otherwise the provided default value.

        On an `Option[T]` the default must be a T. A bare `Nil()` accepts a default of any type.

        Parameters
        ----------
        default_value: T
            Default value used if the Option Monad is Nil.

        Returns
        -------
        result: T
            Returns the Some value or the default value.

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.unwrap_or(0)
        0
        """
        raise NotImplementedError

    def unwrap_or_else(self, function: Callable[[], T]) -> T:
        """Returns the Some value, or otherwise the result of `function`.

        Parameters
        ----------
        function: Callable[[], T]
            Function which will be called if the Option Monad is Nil.

        Returns
        -------
        result: T
            Returns the Some value or the function result.

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.unwrap_or_else(lambda: 0)
        0
        """
        raise NotImplementedError

    def inspect(self, function: Callable[[T], None]) -> Self:
        """Calls function with the Some value (if any) and returns the Option Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Some value of the Option monad

        Returns
        -------
        option: Option[T]
            Returns the Option Monad unchanged.

        Examples
        --------
        >>> val: Option[int] = Some(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Some(42)
        """
        raise NotImplementedError

    def match(self, *, some: Callable[[T], U], nil: Callable[[], U]) -> U:
        """Matches the Option Monad to either the `some` or the `nil` callback with the same return
        type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        some: Callable[[T], U]
            Callback function for Option monads of type Some
        nil: Callable[[], U]
            Callback function for Option monads of type Nil

        Returns
        -------
        result: U
            Returns the result of the callback that was called.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.match(some=lambda x: f"Some: {x}", nil=lambda: "Nil")
        'Some: 10'
        """
        raise NotImplementedError

    def is_nil(self) -> bool:
        """Returns True if the Option Monad is Nil, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.is_nil()
        True
        """
        raise NotImplementedError

    def is_some(self) -> bool:
        """Returns True if the Option Monad is Some, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.is_some()
        True
        """
        raise NotImplementedError

    def to_optional(self) -> T | None:
        """Converts the Option Monad into an optional value: the Some value or None.

        Returns
        -------
        value: T | None
            Returns the Some value, or None if Nil.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.to_optional()
        5
        """
        raise NotImplementedError

    @staticmethod
    @overload
    def from_optional(value: None) -> Nil: ...

    @staticmethod
    @overload
    def from_optional(value: U | None) -> Option[U]: ...

    @staticmethod
    def from_optional(value: U | None) -> Option[U]:
        """Creates an Option Monad from an optional value: Nil for None, otherwise Some.

        Parameters
        ----------
        value: U | None
            Optional value.

        Returns
        -------
        option: Option[U]
            Returns Nil for None, otherwise Some of the value.

        Examples
        --------
        >>> Some.from_optional(None)
        Nil()
        >>> Some.from_optional(1)
        Some(1)
        """
        if value is None:
            return _NIL
        return Some(value)

    @staticmethod
    @overload
    def safe(function: Callable[[], U]) -> Option[U]: ...

    @staticmethod
    @overload
    def safe(
        function: Callable[[], U], *, exceptions: tuple[type[BaseException], ...]
    ) -> Option[U]: ...

    @staticmethod
    def safe(
        function: Callable[[], U],
        *,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> Option[U]:
        """Calls function and wraps its return value in Some, or returns Nil if it raises.

        Parameters
        ----------
        function: Callable[[], U]
            Zero-argument function which may raise an exception.
        exceptions: tuple[type[BaseException], ...]
            The exception types to turn into Nil (default: `Exception`). Any other exception
            propagates.

        Returns
        -------
        option: Option[U]
            Returns Some of the function result, or Nil if it raised one of `exceptions`.

        Examples
        --------
        >>> Some.safe(lambda: 1)
        Some(1)
        >>> Some.safe(lambda: 1 / 0)
        Nil()
        >>> Some.safe(lambda: int("x"), exceptions=(ValueError,))
        Nil()
        """
        try:
            return Some(function())
        except exceptions:
            return _NIL


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Some(_OptionImpl[T]):
    """The Some variant of the Option Monad, wrapping a value.

    Equality and hashing compare the wrapped value (an unhashable value makes the Some unhashable).

    Examples
    --------
    >>> Some(42)
    Some(42)
    >>> str(Some("text"))
    'Some(text)'
    """

    value: T

    def __init__(self, value: T) -> None:
        _set_some_value(self, value)

    def map(self, function: Callable[[T], U]) -> Some[U]:
        return Some(function(self.value))

    def bind(self, function: Callable[[T], Option[U]]) -> Option[U]:
        return function(self.value)

    and_then = bind

    def or_else(self, function: Callable[[], Option[T]]) -> Self:
        return self

    def apply(self, function: Option[Callable[[T], U]]) -> Option[U]:
        if isinstance(function, Some):
            return Some(function.value(self.value))
        return function

    def apply2(self: Some[Callable[[U], V]], value: Option[U]) -> Option[V]:
        if isinstance(value, Some):
            return Some(self.value(value.value))
        return value

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self if predicate(self.value) else _NIL

    def is_some_and(self, function: Callable[[T], bool]) -> bool:
        return function(self.value)

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[T], U]
    ) -> U:
        return function(self.value)

    def and_(self, other: Option[U]) -> Option[U]:
        return other

    def or_(self, other: Option[T]) -> Self:
        return self

    def xor(self, other: Option[T]) -> Option[T]:
        return self if isinstance(other, Nil) else _NIL

    def zip(self, other: Option[U]) -> Option[tuple[T, U]]:
        if isinstance(other, Some):
            return Some((self.value, other.value))
        return other

    def flatten(self: Some[Option[U]]) -> Option[U]:
        return self.value

    def transpose(self: Some[Result[U, E]]) -> Result[Option[U], E]:
        result = self.value
        if isinstance(result, _result.Ok):
            return _result.Ok(Some(result.value))
        failed: Err[Any, E] = result
        return failed

    def ok_or(self, error: E) -> Ok[T, E]:
        return _result.Ok(self.value)

    def ok_or_else(self, function: Callable[[], E]) -> Ok[T, E]:
        return _result.Ok(self.value)

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, function: Callable[[], T]) -> T:
        return self.value

    def inspect(self, function: Callable[[T], None]) -> Self:
        function(self.value)
        return self

    def match(self, *, some: Callable[[T], U], nil: Callable[[], U]) -> U:
        return some(self.value)

    def is_nil(self) -> bool:
        return False

    def is_some(self) -> bool:
        return True

    def to_optional(self) -> T:
        return self.value

    def __str__(self) -> str:
        return f"Some({self.value})"

    def __repr__(self) -> str:
        return f"Some({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Nil(_OptionImpl[Never]):
    """The Nil variant of the Option Monad, representing the absence of a value.

    `Nil` is a singleton of type `Option[Never]`: every `Nil()` is the same object and is
    assignable to any `Option[T]`.

    Examples
    --------
    >>> Nil()
    Nil()
    >>> Nil() is Nil()
    True
    """

    def __new__(cls) -> Nil:
        return _NIL

    def map(self, function: Callable[[Never], U]) -> Nil:
        return self

    def bind(self, function: Callable[[Never], Option[U]]) -> Nil:
        return self

    and_then = bind

    def or_else(self, function: Callable[[], Option[U]]) -> Option[U]:
        return function()

    def apply(self, function: Option[Callable[[Never], U]]) -> Nil:
        return self

    def apply2(self, value: Option[U]) -> Nil:
        return self

    def filter(self, predicate: Callable[[Never], bool]) -> Nil:
        return self

    def is_some_and(self, function: Callable[[Never], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[Never], U]) -> U:
        return default_value

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[Never], U]
    ) -> U:
        return default_function()

    def and_(self, other: Option[U]) -> Nil:
        return self

    def or_(self, other: Option[U]) -> Option[U]:
        return other

    def xor(self, other: Option[U]) -> Option[U]:
        return other

    def zip(self, other: Option[U]) -> Nil:
        return self

    def flatten(self) -> Nil:
        return self

    def transpose(self) -> Ok[Nil, Never]:
        return _result.Ok(self)

    def ok_or(self, error: E) -> Err[Never, E]:
        return _result.Err(error)

    def ok_or_else(self, function: Callable[[], E]) -> Err[Never, E]:
        return _result.Err(function())

    def unwrap(self) -> Never:
        raise UnwrapError(self, "called unwrap on Nil()")

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, function: Callable[[], U]) -> U:
        return function()

    def inspect(self, function: Callable[[Never], None]) -> Self:
        return self

    def match(self, *, some: Callable[[Never], U], nil: Callable[[], U]) -> U:
        return nil()

    def is_nil(self) -> bool:
        return True

    def is_some(self) -> bool:
        return False

    def to_optional(self) -> None:
        return None

    def __str__(self) -> str:
        return "Nil()"

    def __repr__(self) -> str:
        return "Nil()"


type Option[T] = Some[T] | Nil

# A frozen dataclass's generated __init__ assigns fields through object.__setattr__, which is slow.
# Some's __init__ sets its slot through the slot's member descriptor instead (about a third
# faster); the instance stays frozen, so assignment still raises FrozenInstanceError.
_set_some_value: Callable[[Some[Any], Any], None] = Some.__dict__["value"].__set__

# The one Nil instance, which `Nil()` returns.
_NIL: Nil = object.__new__(Nil)
