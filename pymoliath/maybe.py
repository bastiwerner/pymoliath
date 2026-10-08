"""
# Maybe Monad

The Maybe Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data.Maybe.html)
* Rust: [Option](https://doc.rust-lang.org/std/option/)

This implementation is heavily inspired by the Haskell `Maybe` monad and the Rust `Option` type -
see `pymoliath.option` for the sibling implementation using `Some`/`Nil` naming.

The `Maybe` type is a sealed sum type that can be either `Just` or `Nothing`:

```python
type Maybe[T] = Just[T] | Nothing
```

## Typing like in Rust

The type parameter is covariant and `Nothing` is a singleton of type `Maybe[Never]`, so a bare
`Nothing()` is assignable to every `Maybe[T]` and a conditional lambda such as
`lambda x: Just(x) if x > 0 else Nothing()` is inferred as a `Maybe[int]`:

```python
empty: Maybe[int] = Nothing()

def lookup(value: int) -> Maybe[int]:
    return Just(value) if value > 0 else Nothing()
```

Methods called directly on a variant keep the precise variant type, e.g. `Just(1).map(str)` is a
`Just[str]` and `Nothing().map(str)` is `Nothing`.

## Practical Examples and Benefits:

The Maybe Monad is particularly useful in scenarios where a function might not return a value (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without checking `if result is None` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Nothing`, the subsequent operations are skipped automatically, and the final result will be `Nothing`.
3. Type Safety: It forces the developer to acknowledge the possibility of "nothingness" explicitly, making the code more robust against `AttributeError: 'NoneType' object has no attribute...`.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if" logic into a linear pipeline of transformations.

```python
# Without Maybe (Imperative)
user = get_user(user_id)
if user:
    profile = get_profile(user)
    if profile:
        permission = get_permission(profile)
        if permission:
            print(permission)

# With Maybe (Functional)
(get_user(user_id)
    .bind(get_profile)
    .bind(get_permission)
    .unwrap_or("Default Permission"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Maybe`
monad. `Just` and `Nothing` are the only variants (both are `@final`), so a `match` over both is
exhaustive and type checkers narrow the value in each branch.

```python
match maybe_value:
    case Just(x):
        # This block executes if the monad contains a value
        print(f"Just value: {x}")
    case Nothing():
        # This block executes if the monad is Nothing
        print("No value present")
```

`Maybe` is a type alias, so check it at runtime with `isinstance(x, (Just, Nothing))`. Both a `match`
over `Just`/`Nothing` and `isinstance(x, Just)` narrow the type to the variant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Never, Self, final, overload

from typing_extensions import Generic, TypeVar

# maybe.py and either.py convert into each other. Importing the module (not its names) lets the
# cycle resolve at import time; its attributes are looked up when the conversions run.
import pymoliath.either as _either
from pymoliath.errors import UnwrapError

if TYPE_CHECKING:
    from pymoliath.either import Either, Left, Right

# Covariant, so `Just[bool]` is a `Maybe[int]` and `Nothing` (a `Maybe[Never]`) is every `Maybe[T]`.
# Like in Rust the receiver fixes the types some methods accept (`unwrap_or(default: T)`,
# `or_(other: Maybe[T])`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `type: ignore` (misc).
T = TypeVar("T", covariant=True)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
L = TypeVar("L")


class _MaybeImpl(Generic[T]):
    """Public interface of the Maybe Monad. The behaviour lives in `Just` and `Nothing`, its only
    subclasses."""

    __slots__ = ()

    def map(self, function: Callable[[T], U]) -> Maybe[U]:
        """Calls function on a wrapped Just value, otherwise returns Nothing.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        maybe: Maybe[U]
            Returns a Just with the function result or otherwise Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.map(lambda x: x + 1)
        Just(6)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map(lambda x: x + 1)
        Nothing()
        """
        raise NotImplementedError

    def bind(self, function: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """Calls function if the Maybe Monad is Just, otherwise returns Nothing (Rust: `and_then`).

        Parameters
        ----------
        function: Callable[[T], Maybe[U]]
            Function which takes a value of T and returns a new Maybe Monad.

        Returns
        -------
        maybe: Maybe[U]
            Returns the function result if Just, otherwise Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.bind(lambda x: Just(x * 2) if x > 0 else Nothing())
        Just(10)
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[T], Maybe[U]]
            Function which takes a value of T and returns a new Maybe Monad.

        Returns
        -------
        maybe: Maybe[U]
            Returns the function result if Just, otherwise Nothing.

        Examples
        --------
        >>> Just(5).and_then(lambda x: Just(x * 2))
        Just(10)
        """
        raise NotImplementedError

    def or_else(self, function: Callable[[], Maybe[T]]) -> Maybe[T]:
        """Returns this Maybe Monad if it is Just, otherwise the result of function.

        Parameters
        ----------
        function: Callable[[], Maybe[T]]
            Function computing the fallback Maybe Monad.

        Returns
        -------
        maybe: Maybe[T]
            Returns the Just, or the function result if Nothing.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.or_else(lambda: Just(1))
        Just(1)
        """
        raise NotImplementedError

    def apply(self, function: Maybe[Callable[[T], U]]) -> Maybe[U]:
        """Applies the function wrapped in `function` to the Just value if both are Just,
        otherwise returns Nothing. For functions of several
        arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: Maybe[Callable[[T], U]]
            Maybe Monad which contains a function.

        Returns
        -------
        maybe: Maybe[U]
            Returns Just of the function result if both are Just, otherwise Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> func: Maybe[Callable[[int], int]] = Just(lambda x: x * 2)
        >>> val.apply(func)
        Just(20)
        """
        raise NotImplementedError

    def apply2(self: _MaybeImpl[Callable[[U], V]], value: Maybe[U]) -> Maybe[V]:
        """Applies the function wrapped in this Maybe Monad to the value wrapped in `value`.

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both are
        empty/errors, this (the function side) takes precedence. Functions of several
        arguments can be applied one argument at a time when they are curried, e.g.
        `Just(lambda a: lambda b: a + b).apply2(x).apply2(y)`.

        Parameters
        ----------
        value: Maybe[U]
            Maybe Monad which contains the argument.

        Returns
        -------
        maybe: Maybe[V]
            Returns Just of the function result if both are Just, otherwise Nothing.

        Examples
        --------
        >>> func: Maybe[Callable[[int], int]] = Just(lambda y: 10 + y)
        >>> func.apply2(Just(5))
        Just(15)
        >>> func.apply2(Nothing())
        Nothing()
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        """Returns the Maybe Monad if it is Just and the predicate returns True, otherwise Nothing.

        Parameters
        ----------
        predicate: Callable[[T], bool]
            Predicate function applied to the Just value.

        Returns
        -------
        maybe: Maybe[T]
            Returns the Just if the predicate holds, otherwise Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.filter(lambda x: x > 5)
        Just(10)
        >>> val.filter(lambda x: x > 10)
        Nothing()
        """
        raise NotImplementedError

    def is_just_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Maybe Monad is Just and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Just value.

        Returns
        -------
        result: bool
            Returns the predicate result if Just, otherwise False.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.is_just_and(lambda x: x > 5)
        True
        """
        raise NotImplementedError

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Just value, or returns the default value if Nothing.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Maybe Monad is Nothing.
        function: Callable[[T], U]
            Function applied to the Just value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[T], U]
    ) -> U:
        """Applies the function to the Just value, or returns the default function's result if Nothing.

        Parameters
        ----------
        default_function: Callable[[], U]
            Function computing the result if the Maybe Monad is Nothing.
        function: Callable[[T], U]
            Function applied to the Just value.

        Returns
        -------
        result: U
            Returns the function result or the default function's result.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map_or_else(lambda: 0, lambda x: x * 2)
        0
        """
        raise NotImplementedError

    def and_(self, other: Maybe[U]) -> Maybe[U]:
        """Returns `other` if the Maybe Monad is Just, otherwise Nothing.

        Parameters
        ----------
        other: Maybe[U]
            Maybe Monad to be returned if this Maybe Monad is Just.

        Returns
        -------
        maybe: Maybe[U]
            Returns `other` or Nothing.

        Examples
        --------
        >>> val1: Maybe[int] = Just(1)
        >>> val2: Maybe[int] = Just(2)
        >>> val1.and_(val2)
        Just(2)
        """
        raise NotImplementedError

    def or_(self, other: Maybe[T]) -> Maybe[T]:
        """Returns this Maybe Monad if it is Just, otherwise `other`.

        Parameters
        ----------
        other: Maybe[T]
            Maybe Monad to be returned if this Maybe Monad is Nothing.

        Returns
        -------
        maybe: Maybe[T]
            Returns the Just or `other`.

        Examples
        --------
        >>> val1: Maybe[int] = Nothing()
        >>> val2: Maybe[int] = Just(2)
        >>> val1.or_(val2)
        Just(2)
        """
        raise NotImplementedError

    def xor(self, other: Maybe[T]) -> Maybe[T]:
        """Returns the Just if exactly one of this Maybe Monad and `other` is Just, otherwise Nothing.

        Parameters
        ----------
        other: Maybe[T]
            Maybe Monad to be compared with this Maybe Monad.

        Returns
        -------
        maybe: Maybe[T]
            Returns the only Just, or Nothing if both or neither are Just.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.xor(Nothing())
        Just(1)
        >>> val.xor(Just(2))
        Nothing()
        """
        raise NotImplementedError

    def zip(self, other: Maybe[U]) -> Maybe[tuple[T, U]]:
        """Combines this Maybe Monad with another into a Maybe Monad of a tuple, or Nothing if
        either is Nothing.

        Parameters
        ----------
        other: Maybe[U]
            Maybe Monad to be zipped with this Maybe Monad.

        Returns
        -------
        maybe: Maybe[tuple[T, U]]
            Returns Just of a tuple of both values, or Nothing.

        Examples
        --------
        >>> val1: Maybe[int] = Just(1)
        >>> val2: Maybe[str] = Just("a")
        >>> val1.zip(val2)
        Just((1, 'a'))
        """
        raise NotImplementedError

    def flatten(self: _MaybeImpl[Maybe[U]]) -> Maybe[U]:
        """Flattens a nested Maybe Monad by one level.

        Returns
        -------
        maybe: Maybe[U]
            Returns the nested Maybe Monad if Just, otherwise Nothing.

        Examples
        --------
        >>> Just(Just(1)).flatten()
        Just(1)
        """
        raise NotImplementedError

    def transpose(self: _MaybeImpl[Either[L, U]]) -> Either[L, Maybe[U]]:
        """Transposes a Maybe of an Either into an Either of a Maybe.

        `Nothing()` becomes `Right(Nothing())`, `Just(Right(x))` becomes `Right(Just(x))` and
        `Just(Left(e))` becomes `Left(e)`.

        Returns
        -------
        either: Either[L, Maybe[U]]
            Returns the transposed Either Monad.

        Examples
        --------
        >>> from pymoliath.either import Right
        >>> Just(Right(1)).transpose()
        Right(Just(1))
        """
        raise NotImplementedError

    def right_or(self, left_value: L) -> Either[L, T]:
        """Converts the Maybe Monad into an Either Monad, mapping Just(v) to Right(v) and Nothing to
        Left(left_value).

        Parameters
        ----------
        left_value: L
            Left value used if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[L, T]
            Returns Right with the Just value, or Left with left_value.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.right_or("missing")
        Right(1)
        """
        raise NotImplementedError

    def right_or_else(self, function: Callable[[], L]) -> Either[L, T]:
        """Converts the Maybe Monad into an Either Monad, mapping Just(v) to Right(v) and Nothing to
        Left(function()).

        Parameters
        ----------
        function: Callable[[], L]
            Function computing the Left value if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[L, T]
            Returns Right with the Just value, or Left with the function result.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.right_or_else(lambda: "missing")
        Left('missing')
        """
        raise NotImplementedError

    def unwrap(self) -> T:
        """Returns the Just value, or otherwise raises an `UnwrapError`.

        Returns
        -------
        result: T
            Returns the Just value.

        Raises
        ------
        UnwrapError
            If the Maybe Monad is Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.unwrap()
        1
        """
        raise NotImplementedError

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        """Returns the Just value, or otherwise the provided default value.

        On a `Maybe[T]` the default must be a T. A bare `Nothing()` accepts a default of any type.

        Parameters
        ----------
        default_value: T
            Default value used if the Maybe Monad is Nothing.

        Returns
        -------
        result: T
            Returns the Just value or the default value.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.unwrap_or(0)
        0
        """
        raise NotImplementedError

    def unwrap_or_else(self, function: Callable[[], T]) -> T:
        """Returns the Just value, or otherwise the result of `function`.

        Parameters
        ----------
        function: Callable[[], T]
            Function which will be called if the Maybe Monad is Nothing.

        Returns
        -------
        result: T
            Returns the Just value or the function result.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.unwrap_or_else(lambda: 0)
        0
        """
        raise NotImplementedError

    def inspect(self, function: Callable[[T], None]) -> Self:
        """Calls function with the Just value (if any) and returns the Maybe Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Just value of the Maybe monad

        Returns
        -------
        maybe: Maybe[T]
            Returns the Maybe Monad unchanged.

        Examples
        --------
        >>> val: Maybe[int] = Just(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Just(42)
        """
        raise NotImplementedError

    def match(self, *, just: Callable[[T], U], nothing: Callable[[], U]) -> U:
        """Matches the Maybe Monad to either the `just` or the `nothing` callback with the same return
        type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        just: Callable[[T], U]
            Callback function for Maybe monads of type Just
        nothing: Callable[[], U]
            Callback function for Maybe monads of type Nothing

        Returns
        -------
        result: U
            Returns the result of the callback that was called.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.match(just=lambda x: f"Just: {x}", nothing=lambda: "Nothing")
        'Just: 10'
        """
        raise NotImplementedError

    def is_nothing(self) -> bool:
        """Returns True if the Maybe Monad is Nothing, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.is_nothing()
        True
        """
        raise NotImplementedError

    def is_just(self) -> bool:
        """Returns True if the Maybe Monad is Just, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.is_just()
        True
        """
        raise NotImplementedError

    def to_optional(self) -> T | None:
        """Converts the Maybe Monad into an optional value: the Just value or None.

        Returns
        -------
        value: T | None
            Returns the Just value, or None if Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.to_optional()
        5
        """
        raise NotImplementedError

    @staticmethod
    @overload
    def from_optional(value: None) -> Nothing: ...

    @staticmethod
    @overload
    def from_optional(value: U | None) -> Maybe[U]: ...

    @staticmethod
    def from_optional(value: U | None) -> Maybe[U]:
        """Creates a Maybe Monad from an optional value: Nothing for None, otherwise Just.

        Parameters
        ----------
        value: U | None
            Optional value.

        Returns
        -------
        maybe: Maybe[U]
            Returns Nothing for None, otherwise Just of the value.

        Examples
        --------
        >>> Just.from_optional(None)
        Nothing()
        >>> Just.from_optional(1)
        Just(1)
        """
        if value is None:
            return _NOTHING
        return Just(value)

    @staticmethod
    @overload
    def safe(function: Callable[[], U]) -> Maybe[U]: ...

    @staticmethod
    @overload
    def safe(
        function: Callable[[], U], *, exceptions: tuple[type[BaseException], ...]
    ) -> Maybe[U]: ...

    @staticmethod
    def safe(
        function: Callable[[], U],
        *,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> Maybe[U]:
        """Calls function and wraps its return value in Just, or returns Nothing if it raises.

        Parameters
        ----------
        function: Callable[[], U]
            Zero-argument function which may raise an exception.
        exceptions: tuple[type[BaseException], ...]
            The exception types to turn into Nothing (default: `Exception`). Any other exception
            propagates.

        Returns
        -------
        maybe: Maybe[U]
            Returns Just of the function result, or Nothing if it raised one of `exceptions`.

        Examples
        --------
        >>> Just.safe(lambda: 1)
        Just(1)
        >>> Just.safe(lambda: 1 / 0)
        Nothing()
        >>> Just.safe(lambda: int("x"), exceptions=(ValueError,))
        Nothing()
        """
        try:
            return Just(function())
        except exceptions:
            return _NOTHING


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Just(_MaybeImpl[T]):
    """The Just variant of the Maybe Monad, wrapping a value.

    Equality and hashing compare the wrapped value (an unhashable value makes the Just unhashable).

    Examples
    --------
    >>> Just(42)
    Just(42)
    >>> str(Just("text"))
    'Just(text)'
    """

    value: T

    def __init__(self, value: T) -> None:
        _set_just_value(self, value)

    def map(self, function: Callable[[T], U]) -> Just[U]:
        return Just(function(self.value))

    def bind(self, function: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return function(self.value)

    and_then = bind

    def or_else(self, function: Callable[[], Maybe[T]]) -> Self:
        return self

    def apply(self, function: Maybe[Callable[[T], U]]) -> Maybe[U]:
        if isinstance(function, Just):
            return Just(function.value(self.value))
        return function

    def apply2(self: Just[Callable[[U], V]], value: Maybe[U]) -> Maybe[V]:
        if isinstance(value, Just):
            return Just(self.value(value.value))
        return value

    def filter(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        return self if predicate(self.value) else _NOTHING

    def is_just_and(self, function: Callable[[T], bool]) -> bool:
        return function(self.value)

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[T], U]
    ) -> U:
        return function(self.value)

    def and_(self, other: Maybe[U]) -> Maybe[U]:
        return other

    def or_(self, other: Maybe[T]) -> Self:
        return self

    def xor(self, other: Maybe[T]) -> Maybe[T]:
        return self if isinstance(other, Nothing) else _NOTHING

    def zip(self, other: Maybe[U]) -> Maybe[tuple[T, U]]:
        if isinstance(other, Just):
            return Just((self.value, other.value))
        return other

    def flatten(self: Just[Maybe[U]]) -> Maybe[U]:
        return self.value

    def transpose(self: Just[Either[L, U]]) -> Either[L, Maybe[U]]:
        either = self.value
        if isinstance(either, _either.Right):
            return _either.Right(Just(either.value))
        failed: Left[L, Any] = either
        return failed

    def right_or(self, left_value: L) -> Right[L, T]:
        return _either.Right(self.value)

    def right_or_else(self, function: Callable[[], L]) -> Right[L, T]:
        return _either.Right(self.value)

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, function: Callable[[], T]) -> T:
        return self.value

    def inspect(self, function: Callable[[T], None]) -> Self:
        function(self.value)
        return self

    def match(self, *, just: Callable[[T], U], nothing: Callable[[], U]) -> U:
        return just(self.value)

    def is_nothing(self) -> bool:
        return False

    def is_just(self) -> bool:
        return True

    def to_optional(self) -> T:
        return self.value

    def __str__(self) -> str:
        return f"Just({self.value})"

    def __repr__(self) -> str:
        return f"Just({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Nothing(_MaybeImpl[Never]):
    """The Nothing variant of the Maybe Monad, representing the absence of a value.

    `Nothing` is a singleton of type `Maybe[Never]`: every `Nothing()` is the same object and is
    assignable to any `Maybe[T]`.

    Examples
    --------
    >>> Nothing()
    Nothing()
    >>> Nothing() is Nothing()
    True
    """

    _instance: ClassVar[Nothing | None] = None

    def __new__(cls) -> Nothing:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def map(self, function: Callable[[Never], U]) -> Nothing:
        return self

    def bind(self, function: Callable[[Never], Maybe[U]]) -> Nothing:
        return self

    and_then = bind

    def or_else(self, function: Callable[[], Maybe[U]]) -> Maybe[U]:
        return function()

    def apply(self, function: Maybe[Callable[[Never], U]]) -> Nothing:
        return self

    def apply2(self, value: Maybe[U]) -> Nothing:
        return self

    def filter(self, predicate: Callable[[Never], bool]) -> Nothing:
        return self

    def is_just_and(self, function: Callable[[Never], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[Never], U]) -> U:
        return default_value

    def map_or_else(
        self, default_function: Callable[[], U], function: Callable[[Never], U]
    ) -> U:
        return default_function()

    def and_(self, other: Maybe[U]) -> Nothing:
        return self

    def or_(self, other: Maybe[U]) -> Maybe[U]:
        return other

    def xor(self, other: Maybe[U]) -> Maybe[U]:
        return other

    def zip(self, other: Maybe[U]) -> Nothing:
        return self

    def flatten(self) -> Nothing:
        return self

    def transpose(self) -> Right[Never, Nothing]:
        return _either.Right(self)

    def right_or(self, left_value: L) -> Left[L, Never]:
        return _either.Left(left_value)

    def right_or_else(self, function: Callable[[], L]) -> Left[L, Never]:
        return _either.Left(function())

    def unwrap(self) -> Never:
        raise UnwrapError(self, "called unwrap on Nothing()")

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, function: Callable[[], U]) -> U:
        return function()

    def inspect(self, function: Callable[[Never], None]) -> Self:
        return self

    def match(self, *, just: Callable[[Never], U], nothing: Callable[[], U]) -> U:
        return nothing()

    def is_nothing(self) -> bool:
        return True

    def is_just(self) -> bool:
        return False

    def to_optional(self) -> None:
        return None

    def __str__(self) -> str:
        return "Nothing()"

    def __repr__(self) -> str:
        return "Nothing()"


type Maybe[T] = Just[T] | Nothing

# A frozen dataclass's generated __init__ assigns fields through object.__setattr__, which is slow.
# Just's __init__ sets its slot through the slot's member descriptor instead (about a third
# faster); the instance stays frozen, so assignment still raises FrozenInstanceError.
_set_just_value: Callable[[Just[Any], Any], None] = Just.__dict__["value"].__set__

# The Nothing singleton, returned directly on hot paths instead of going through Nothing.__new__.
_NOTHING = Nothing()
