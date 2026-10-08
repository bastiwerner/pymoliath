"""
# Maybe Monad

The Maybe Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data.Maybe.html)
* Rust: [Option](https://doc.rust-lang.org/std/option/)

This implementation is heavily inspired by the Haskell `Maybe` monad and the Rust `Option` type -
see `pymoliath.option` for the sibling implementation using `Some`/`Nil` naming.

The `Maybe` type is a sealed sum type that can be either `Just` or `Nothing`. Like in Rust, both
variants carry the type parameter:

```python
type Maybe[T] = Just[T] | Nothing[T]
```

## Typing like in Rust

Because both variants know the full `Maybe[T]`, lambdas passed to `map`/`bind`/... are inferred
cleanly and a `match` over `Just`/`Nothing` is exhaustive. A bare `Nothing()` leaves its type open
(`Nothing[Unknown]`) so it can be solved from context, e.g. in
`lambda x: Just(x) if x > 0 else Nothing()`. Like in Rust, annotate an empty value that has no
context:

```python
empty: Maybe[int] = Nothing()

def lookup(value: int) -> Maybe[int]:
    return Just(value) if value > 0 else Nothing()
```

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
monad. `Just` and `Nothing` are the only variants (the Maybe is sealed), so a `match` over both is
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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Never,
    assert_never,
    cast,
    final,
    overload,
)

from typing_extensions import TypeVar

from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.either import Either

# Old-style TypeVars are invariant by default, which is what Maybe needs anyway
# (T also appears in parameter positions, e.g. unwrap_or and or_).
T = TypeVar("T")
U = TypeVar("U")
L = TypeVar("L")


class _MaybeImpl(Generic[T]):
    """Shared implementation of the Maybe Monad - `Just` and `Nothing` are its only subclasses."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Runtime "sealed": only Just and Nothing (defined in this module) may subclass.
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("Maybe cannot be subclassed; use Just or Nothing")

    def _as_maybe(self) -> Maybe[T]:
        # Safe: Just and Nothing are the only subclasses (see __init_subclass__).
        return cast("Maybe[T]", self)

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
        match m := self._as_maybe():
            case Just(value):
                return Just(function(value))
            case Nothing():
                return Nothing()
            case _:
                assert_never(m)

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
        match m := self._as_maybe():
            case Just(value):
                return function(value)
            case Nothing():
                return Nothing()
            case _:
                assert_never(m)

    def apply(self, applicative: Maybe[Callable[..., U]]) -> Maybe[U]:
        """Applies the passed applicative wrapping a function if the Maybe Monad is Just, otherwise
        returns Nothing. Functions of several arguments are curried.

        Parameters
        ----------
        applicative: Maybe[Callable[[T], U]]
            Applicative Maybe Monad which contains a function.

        Returns
        -------
        maybe: Maybe[U]
            Returns a Maybe Monad from the applied function if both are Just, otherwise Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> func: Maybe[Callable[[int], int]] = Just(lambda x: x * 2)
        >>> val.apply(func)
        Just(20)
        """
        return applicative.bind(lambda function: self.map(curry(function)))

    def apply2(
        self: _MaybeImpl[Callable[..., U]], applicative_value: Maybe[Any]
    ) -> Maybe[U]:
        """Applies the function wrapped in this Maybe Monad to the passed Maybe Monad wrapping a
        value if both are Just, otherwise returns Nothing. Functions of several arguments are curried.

        Parameters
        ----------
        applicative_value: Maybe[Any]
            Maybe monad which contains a value.

        Returns
        -------
        maybe: Maybe[U]
            Returns a Maybe Monad from the applied function if both are Just, otherwise Nothing.

        Examples
        --------
        >>> func: Maybe[Callable[[int], int]] = Just(lambda y: 10 + y)
        >>> val: Maybe[int] = Just(5)
        >>> func.apply2(val)
        Just(15)
        >>> empty: Maybe[int] = Nothing()
        >>> func.apply2(empty)
        Nothing()
        """
        return self.bind(lambda function: applicative_value.map(curry(function)))

    def filter(self, filter_function: Callable[[T], bool]) -> Maybe[T]:
        """Returns the Maybe Monad if it is Just and the predicate returns True, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[T], bool]
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
        match m := self._as_maybe():
            case Just(value) if filter_function(value):
                return m
            case Just() | Nothing():
                return Nothing()
            case _:
                assert_never(m)

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
        match m := self._as_maybe():
            case Just(value):
                return function(value)
            case Nothing():
                return False
            case _:
                assert_never(m)

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
        match m := self._as_maybe():
            case Just(value):
                return function(value)
            case Nothing():
                return default_value
            case _:
                assert_never(m)

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
        match m := self._as_maybe():
            case Just():
                return other
            case Nothing():
                return Nothing()
            case _:
                assert_never(m)

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
        match m := self._as_maybe():
            case Just():
                return m
            case Nothing():
                return other
            case _:
                assert_never(m)

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
        return self.bind(
            lambda value: other.map(lambda other_value: (value, other_value))
        )

    @overload
    def flatten(self: _MaybeImpl[Just[U]]) -> Maybe[U]: ...

    @overload
    def flatten(self: _MaybeImpl[Nothing[U]]) -> Maybe[U]: ...

    @overload
    def flatten(self: _MaybeImpl[Maybe[U]]) -> Maybe[U]: ...

    @overload
    def flatten(self: Nothing[Any]) -> Maybe[Never]: ...

    def flatten(self: _MaybeImpl[Any]) -> Maybe[Any]:
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
        match m := self._as_maybe():
            case Just(value):
                return cast("Maybe[Any]", value)
            case Nothing():
                return Nothing()
            case _:
                assert_never(m)

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
        from pymoliath.either import Left, Right

        match m := self._as_maybe():
            case Just(value):
                return Right(value)
            case Nothing():
                return Left(left_value)
            case _:
                assert_never(m)

    def right_or_else(self, left_function: Callable[[], L]) -> Either[L, T]:
        """Converts the Maybe Monad into an Either Monad, mapping Just(v) to Right(v) and Nothing to
        Left(left_function()).

        Parameters
        ----------
        left_function: Callable[[], L]
            Function computing the Left value if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[L, T]
            Returns Right with the Just value, or Left with the left_function result.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.right_or_else(lambda: "missing")
        Left(missing)
        """
        from pymoliath.either import Left, Right

        match m := self._as_maybe():
            case Just(value):
                return Right(value)
            case Nothing():
                return Left(left_function())
            case _:
                assert_never(m)

    def unwrap(self) -> T:
        """Returns the Just value, or otherwise raises an Exception.

        Returns
        -------
        result: T
            Returns the Just value.

        Raises
        ------
        Exception
            If the Maybe Monad is Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.unwrap()
        1
        """
        match m := self._as_maybe():
            case Just(value):
                return value
            case Nothing():
                raise Exception("Unwrap error on Maybe monad")
            case _:
                assert_never(m)

    def unwrap_or(self, default_value: T) -> T:
        """Returns the Just value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: T
            Default value of T

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
        match m := self._as_maybe():
            case Just(value):
                return value
            case Nothing():
                return default_value
            case _:
                assert_never(m)

    def unwrap_or_else(self, nothing_function: Callable[[], T]) -> T:
        """Returns the Just value, or otherwise calls the nothing_function.

        Parameters
        ----------
        nothing_function: Callable[[], T]
            Function which will be called if the Maybe Monad is Nothing.

        Returns
        -------
        result: T
            Returns the Just value or the nothing_function result.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.unwrap_or_else(lambda: 0)
        0
        """
        match m := self._as_maybe():
            case Just(value):
                return value
            case Nothing():
                return nothing_function()
            case _:
                assert_never(m)

    def inspect(self, function: Callable[[T], None]) -> Maybe[T]:
        """Calls function with the Just value (if any) and returns the Maybe Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Just value of the Maybe monad

        Returns
        -------
        maybe: Maybe[T]

        Examples
        --------
        >>> val: Maybe[int] = Just(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Just(42)
        """
        match m := self._as_maybe():
            case Just(value):
                function(value)
            case Nothing():
                pass
            case _:
                assert_never(m)
        return m

    def match(
        self, just_function: Callable[[T], U], nothing_function: Callable[[], U]
    ) -> U:
        """Matches the Maybe Monad to either a Just function or a Nothing function with the same
        return type.

        Parameters
        ----------
        just_function: Callable[[T], U]
            Callback function for Maybe monads of type Just
        nothing_function: Callable[[], U]
            Callback function for Maybe monads of type Nothing

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.match(lambda x: f"Just: {x}", lambda: "Nothing")
        'Just: 10'
        """
        match m := self._as_maybe():
            case Just(value):
                return just_function(value)
            case Nothing():
                return nothing_function()
            case _:
                assert_never(m)

    def is_nothing(self) -> bool:
        """Returns True if the Maybe Monad is Nothing, otherwise False.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.is_nothing()
        True
        """
        return isinstance(self, Nothing)

    def is_just(self) -> bool:
        """Returns True if the Maybe Monad is Just, otherwise False.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.is_just()
        True
        """
        return isinstance(self, Just)

    def to_optional(self) -> T | None:
        """Converts the Maybe Monad into an optional value: the Just value or None.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.to_optional()
        5
        """
        match m := self._as_maybe():
            case Just(value):
                return value
            case Nothing():
                return None
            case _:
                assert_never(m)

    @staticmethod
    def from_optional(value: U | None) -> Maybe[U]:
        """Creates a Maybe Monad from an optional value: Nothing for None, otherwise Just.

        Examples
        --------
        >>> Just.from_optional(None)
        Nothing()
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Maybe Monad.

        Examples
        --------
        >>> str(Just(42))
        'Just(42)'
        >>> str(Nothing())
        'Nothing()'
        """
        match m := self._as_maybe():
            case Just(value):
                return f"Just({value})"
            case Nothing():
                return "Nothing()"
            case _:
                assert_never(m)

    def __repr__(self) -> str:
        """Returns the string representation of the Maybe Monad (same as __str__)."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is the same variant wrapping a value of the same type and string.

        Examples
        --------
        >>> Just(1) == Just(1)
        True
        >>> Just(1) == Nothing()
        False
        """
        if type(self) is not type(other):
            return False
        match m := self._as_maybe():
            case Just(value):
                other_value = cast("Just[Any]", other).value
                return type(value) is type(other_value) and str(value) == str(
                    other_value
                )
            case Nothing():
                return True
            case _:
                assert_never(m)

    __hash__ = None  # type: ignore[assignment]


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Just(_MaybeImpl[T]):
    """The Just variant of the Maybe Monad, wrapping a value.

    Examples
    --------
    >>> Just(42)
    Just(42)
    """

    value: T


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Nothing(_MaybeImpl[T]):
    """The Nothing variant of the Maybe Monad, representing the absence of a value (a singleton).

    Examples
    --------
    >>> Nothing()
    Nothing()
    """

    _instance: ClassVar[Nothing[Any] | None] = None

    def __new__(cls) -> Nothing[T]:
        # Nothing carries no data, so all instances (of any T) are the same object.
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cast("Nothing[T]", cls._instance)


# Nothing's type parameter has deliberately no default: if it defaulted to Never, pyright would pin
# a lambda's type to Never as soon as one branch returns Nothing
# (`lambda x: Just(x) if x else Nothing()`).
type Maybe[T] = Just[T] | Nothing[T]


def from_optional(value: T | None) -> Maybe[T]:
    """Creates a Maybe Monad from an optional value: Nothing for None, otherwise Just.

    Parameters
    ----------
    value: T | None
        Maybeal value.

    Returns
    -------
    maybe: Maybe[T]

    Examples
    --------
    >>> from_optional(None)
    Nothing()
    >>> from_optional(1)
    Just(1)
    """
    if value is None:
        return Nothing()
    return Just(value)


def safe(function: Callable[[], T]) -> Maybe[T]:
    """Calls function and wraps its return value in Just, or returns Nothing if it raises an Exception.

    Parameters
    ----------
    function: Callable[[], T]
        Zero-argument function which may raise an Exception.

    Returns
    -------
    maybe: Maybe[T]

    Examples
    --------
    >>> safe(lambda: 1)
    Just(1)
    >>> safe(lambda: 1 / 0)
    Nothing()
    """
    try:
        return Just(function())
    except Exception:
        return Nothing()
