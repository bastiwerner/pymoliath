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

`Maybe` is a type alias, so use `is_maybe` (or `isinstance(x, MAYBE_TYPES)`) for runtime checks
and `is_just`/`is_nothing` to narrow a `Maybe` to one of its variants. Every common method also
exists as a curried module-level function for use with `pymoliath.util.flow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Never, Self, final, overload

from typing_extensions import Generic, TypeIs, TypeVar

from pymoliath.errors import UnwrapError

# Covariant, so `Just[bool]` is a `Maybe[int]` and `Nothing` (a `Maybe[Never]`) is every `Maybe[T]`.
# Like in Rust the receiver fixes the types some methods accept (`unwrap_or(default: T)`,
# `or_(other: Maybe[T])`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `# type: ignore[misc]`.
T = TypeVar("T", covariant=True)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Y = TypeVar("Y")
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

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.or_else(lambda: Just(1))
        Just(1)
        """
        raise NotImplementedError

    def apply(self, function: Maybe[Callable[[T], U]]) -> Maybe[U]:
        """Applies the function wrapped in `function` to the Just value if both are Just,
        otherwise returns Nothing. For functions of several arguments, use `map2`/`map3`.

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
        raise NotImplementedError

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

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.match(just=lambda x: f"Just: {x}", nothing=lambda: "Nothing")
        'Just: 10'
        """
        raise NotImplementedError

    def is_nothing(self) -> bool:
        """Returns True if the Maybe Monad is Nothing, otherwise False. Use the module-level `is_nothing`
        to narrow the type.

        Examples
        --------
        >>> val: Maybe[int] = Nothing()
        >>> val.is_nothing()
        True
        """
        raise NotImplementedError

    def is_just(self) -> bool:
        """Returns True if the Maybe Monad is Just, otherwise False. Use the module-level
        `is_just` to narrow the type.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.is_just()
        True
        """
        raise NotImplementedError

    def to_optional(self) -> T | None:
        """Converts the Maybe Monad into an optional value: the Just value or None.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.to_optional()
        5
        """
        raise NotImplementedError

    @staticmethod
    def from_optional(value: U | None) -> Maybe[U]:
        """Creates a Maybe Monad from an optional value: Nothing for None, otherwise Just.

        Examples
        --------
        >>> Just.from_optional(None)
        Nothing()
        """
        return from_optional(value)


@final
@dataclass(frozen=True, slots=True, repr=False)
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

    def filter(self, filter_function: Callable[[T], bool]) -> Maybe[T]:
        return self if filter_function(self.value) else Nothing()

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
        return self if isinstance(other, Nothing) else Nothing()

    def zip(self, other: Maybe[U]) -> Maybe[tuple[T, U]]:
        if isinstance(other, Just):
            return Just((self.value, other.value))
        return other

    def flatten(self: Just[Maybe[U]]) -> Maybe[U]:
        return self.value

    def transpose(self: Just[Either[L, U]]) -> Either[L, Maybe[U]]:
        either = self.value
        if isinstance(either, Right):
            return Right(Just(either.value))
        return either  # type: ignore[return-value]

    def right_or(self, left_value: L) -> Right[T, L]:
        return Right(self.value)

    def right_or_else(self, left_function: Callable[[], L]) -> Right[T, L]:
        return Right(self.value)

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, nothing_function: Callable[[], T]) -> T:
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
@dataclass(frozen=True, slots=True, repr=False)
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

    def filter(self, filter_function: Callable[[Never], bool]) -> Nothing:
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

    def transpose(self) -> Right[Nothing, Never]:
        return Right(self)

    def right_or(self, left_value: L) -> Left[L, Never]:
        return Left(left_value)

    def right_or_else(self, left_function: Callable[[], L]) -> Left[L, Never]:
        return Left(left_function())

    def unwrap(self) -> Never:
        raise UnwrapError(self, "called unwrap on Nothing()")

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], U]) -> U:
        return nothing_function()

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

MAYBE_TYPES: tuple[type[Just[Any]], type[Nothing]] = (Just, Nothing)
"""The runtime classes of `Maybe`, for `isinstance` checks (`Maybe` itself is a type alias)."""


def is_maybe(value: object) -> TypeIs[Maybe[Any]]:
    """Returns True if `value` is a Just or Nothing.

    Examples
    --------
    >>> is_maybe(Nothing()), is_maybe(None)
    (True, False)
    """
    return isinstance(value, MAYBE_TYPES)


def is_just(maybe: Maybe[U]) -> TypeIs[Just[U]]:
    """Returns True if the Maybe Monad is Just, narrowing it to `Just` for type checkers.

    Examples
    --------
    >>> val: Maybe[int] = Just(1)
    >>> if is_just(val):
    ...     print(val.value)
    1
    """
    return isinstance(maybe, Just)


def is_nothing(maybe: Maybe[U]) -> TypeIs[Nothing]:
    """Returns True if the Maybe Monad is Nothing, narrowing it to `Nothing` for type checkers.

    Examples
    --------
    >>> is_nothing(Nothing())
    True
    """
    return isinstance(maybe, Nothing)


def from_optional(value: U | None) -> Maybe[U]:
    """Creates a Maybe Monad from an optional value: Nothing for None, otherwise Just.

    Parameters
    ----------
    value: U | None
        Optional value.

    Returns
    -------
    maybe: Maybe[U]

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


def map2(first: Maybe[U], second: Maybe[V], function: Callable[[U, V], W]) -> Maybe[W]:
    """Applies a two-argument function to the values of two Maybe Monads if both are Just.

    Examples
    --------
    >>> map2(Just(1), Just(2), lambda a, b: a + b)
    Just(3)
    >>> map2(Just(1), Nothing(), lambda a, b: a + b)
    Nothing()
    """
    if isinstance(first, Just) and isinstance(second, Just):
        return Just(function(first.value, second.value))
    return Nothing()


def map3(
    first: Maybe[U],
    second: Maybe[V],
    third: Maybe[W],
    function: Callable[[U, V, W], Y],
) -> Maybe[Y]:
    """Applies a three-argument function to the values of three Maybe Monads if all are Just.

    Examples
    --------
    >>> map3(Just(1), Just(2), Just(3), lambda a, b, c: a + b + c)
    Just(6)
    """
    if isinstance(first, Just) and isinstance(second, Just) and isinstance(third, Just):
        return Just(function(first.value, second.value, third.value))
    return Nothing()


@overload
def safe(function: Callable[[], U]) -> Maybe[U]: ...


@overload
def safe(
    function: Callable[[], U], *, exceptions: tuple[type[BaseException], ...]
) -> Maybe[U]: ...


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

    Examples
    --------
    >>> safe(lambda: 1)
    Just(1)
    >>> safe(lambda: 1 / 0)
    Nothing()
    >>> safe(lambda: int("x"), exceptions=(ValueError,))
    Nothing()
    """
    try:
        return Just(function())
    except exceptions:
        return Nothing()


# Curried module-level functions, for point-free pipelines (see `pymoliath.util.flow`).


def map(function: Callable[[U], V]) -> Callable[[Maybe[U]], Maybe[V]]:
    """Curried `Maybe.map`.

    Examples
    --------
    >>> map(lambda x: x + 1)(Just(1))
    Just(2)
    """
    return lambda maybe: maybe.map(function)


def bind(function: Callable[[U], Maybe[V]]) -> Callable[[Maybe[U]], Maybe[V]]:
    """Curried `Maybe.bind`.

    Examples
    --------
    >>> bind(lambda x: Just(x + 1))(Just(1))
    Just(2)
    """
    return lambda maybe: maybe.bind(function)


def filter(function: Callable[[U], bool]) -> Callable[[Maybe[U]], Maybe[U]]:
    """Curried `Maybe.filter`.

    Examples
    --------
    >>> filter(lambda x: x > 1)(Just(1))
    Nothing()
    """
    return lambda maybe: maybe.filter(function)


def unwrap_or(default_value: U) -> Callable[[Maybe[U]], U]:
    """Curried `Maybe.unwrap_or`.

    Examples
    --------
    >>> unwrap_or(0)(Nothing())
    0
    """
    return lambda maybe: maybe.unwrap_or(default_value)


def unwrap_or_else(function: Callable[[], U]) -> Callable[[Maybe[U]], U]:
    """Curried `Maybe.unwrap_or_else`.

    Examples
    --------
    >>> unwrap_or_else(lambda: 0)(Nothing())
    0
    """
    return lambda maybe: maybe.unwrap_or_else(function)


def inspect(function: Callable[[U], None]) -> Callable[[Maybe[U]], Maybe[U]]:
    """Curried `Maybe.inspect`.

    Examples
    --------
    >>> inspect(print)(Just(1))
    1
    Just(1)
    """
    return lambda maybe: maybe.inspect(function)


# Imported last: either.py imports Just/Nothing from this module, so the cycle resolves once at
# import time instead of on every call.
from pymoliath.either import Either, Left, Right  # noqa: E402
