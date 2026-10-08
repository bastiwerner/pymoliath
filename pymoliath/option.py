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

`Option` is a type alias, so use `is_option` (or `isinstance(x, OPTION_TYPES)`) for runtime checks
and `is_some`/`is_nil` to narrow an `Option` to one of its variants. Every common method also
exists as a curried module-level function for use with `pymoliath.util.flow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Never, Self, final, overload

from typing_extensions import Generic, TypeIs, TypeVar

from pymoliath.errors import UnwrapError

# Covariant, so `Some[bool]` is an `Option[int]` and `Nil` (an `Option[Never]`) is every `Option[T]`.
# Like in Rust the receiver fixes the types some methods accept (`unwrap_or(default: T)`,
# `or_(other: Option[T])`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `# type: ignore[misc]`.
T = TypeVar("T", covariant=True)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Y = TypeVar("Y")
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

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.or_else(lambda: Some(1))
        Some(1)
        """
        raise NotImplementedError

    def apply(self, function: Option[Callable[[T], U]]) -> Option[U]:
        """Applies the function wrapped in `function` to the Some value if both are Some,
        otherwise returns Nil. For functions of several arguments, use `map2`/`map3`.

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

    def filter(self, filter_function: Callable[[T], bool]) -> Option[T]:
        """Returns the Option Monad if it is Some and the predicate returns True, otherwise Nil.

        Parameters
        ----------
        filter_function: Callable[[T], bool]
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

    def transpose(self: _OptionImpl[Result[U, E]]) -> Result[Option[U], E]:
        """Transposes an Option of a Result into a Result of an Option.

        `Nil()` becomes `Ok(Nil())`, `Some(Ok(x))` becomes `Ok(Some(x))` and `Some(Err(e))` becomes
        `Err(e)`.

        Examples
        --------
        >>> from pymoliath.result import Ok
        >>> Some(Ok(1)).transpose()
        Ok(Some(1))
        """
        raise NotImplementedError

    def ok_or(self, err_value: E) -> Result[T, E]:
        """Converts the Option Monad into a Result Monad, mapping Some(v) to Ok(v) and Nil to
        Err(err_value).

        Parameters
        ----------
        err_value: E
            Error value used if the Option Monad is Nil.

        Returns
        -------
        result: Result[T, E]
            Returns Ok with the Some value, or Err with err_value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.ok_or("missing")
        Ok(1)
        """
        raise NotImplementedError

    def ok_or_else(self, err_function: Callable[[], E]) -> Result[T, E]:
        """Converts the Option Monad into a Result Monad, mapping Some(v) to Ok(v) and Nil to
        Err(err_function()).

        Parameters
        ----------
        err_function: Callable[[], E]
            Function computing the error value if the Option Monad is Nil.

        Returns
        -------
        result: Result[T, E]
            Returns Ok with the Some value, or Err with the err_function result.

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
        """Returns the Some value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: T
            Default value of T

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

    def unwrap_or_else(self, nothing_function: Callable[[], T]) -> T:
        """Returns the Some value, or otherwise calls the nothing_function.

        Parameters
        ----------
        nothing_function: Callable[[], T]
            Function which will be called if the Option Monad is Nil.

        Returns
        -------
        result: T
            Returns the Some value or the nothing_function result.

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

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.match(some=lambda x: f"Some: {x}", nil=lambda: "Nil")
        'Some: 10'
        """
        raise NotImplementedError

    def is_nothing(self) -> bool:
        """Returns True if the Option Monad is Nil, otherwise False. Use the module-level `is_nil`
        to narrow the type.

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.is_nothing()
        True
        """
        raise NotImplementedError

    def is_some(self) -> bool:
        """Returns True if the Option Monad is Some, otherwise False. Use the module-level
        `is_some` to narrow the type.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.is_some()
        True
        """
        raise NotImplementedError

    def to_optional(self) -> T | None:
        """Converts the Option Monad into an optional value: the Some value or None.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.to_optional()
        5
        """
        raise NotImplementedError

    @staticmethod
    def from_optional(value: U | None) -> Option[U]:
        """Creates an Option Monad from an optional value: Nil for None, otherwise Some.

        Examples
        --------
        >>> Some.from_optional(None)
        Nil()
        """
        return from_optional(value)


@final
@dataclass(frozen=True, slots=True, repr=False)
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

    def filter(self, filter_function: Callable[[T], bool]) -> Option[T]:
        return self if filter_function(self.value) else Nil()

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
        return self if isinstance(other, Nil) else Nil()

    def zip(self, other: Option[U]) -> Option[tuple[T, U]]:
        if isinstance(other, Some):
            return Some((self.value, other.value))
        return other

    def flatten(self: Some[Option[U]]) -> Option[U]:
        return self.value

    def transpose(self: Some[Result[U, E]]) -> Result[Option[U], E]:
        result = self.value
        if isinstance(result, Ok):
            return Ok(Some(result.value))
        return result  # type: ignore[return-value]

    def ok_or(self, err_value: E) -> Ok[T, E]:
        return Ok(self.value)

    def ok_or_else(self, err_function: Callable[[], E]) -> Ok[T, E]:
        return Ok(self.value)

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, nothing_function: Callable[[], T]) -> T:
        return self.value

    def inspect(self, function: Callable[[T], None]) -> Self:
        function(self.value)
        return self

    def match(self, *, some: Callable[[T], U], nil: Callable[[], U]) -> U:
        return some(self.value)

    def is_nothing(self) -> bool:
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
@dataclass(frozen=True, slots=True, repr=False)
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

    _instance: ClassVar[Nil | None] = None

    def __new__(cls) -> Nil:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def map(self, function: Callable[[Never], U]) -> Nil:
        return self

    def bind(self, function: Callable[[Never], Option[U]]) -> Nil:
        return self

    and_then = bind

    def or_else(self, function: Callable[[], Option[U]]) -> Option[U]:
        return function()

    def apply(self, function: Option[Callable[[Never], U]]) -> Nil:
        return self

    def filter(self, filter_function: Callable[[Never], bool]) -> Nil:
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
        return Ok(self)

    def ok_or(self, err_value: E) -> Err[Never, E]:
        return Err(err_value)

    def ok_or_else(self, err_function: Callable[[], E]) -> Err[Never, E]:
        return Err(err_function())

    def unwrap(self) -> Never:
        raise UnwrapError(self, "called unwrap on Nil()")

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], U]) -> U:
        return nothing_function()

    def inspect(self, function: Callable[[Never], None]) -> Self:
        return self

    def match(self, *, some: Callable[[Never], U], nil: Callable[[], U]) -> U:
        return nil()

    def is_nothing(self) -> bool:
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

OPTION_TYPES: tuple[type[Some[Any]], type[Nil]] = (Some, Nil)
"""The runtime classes of `Option`, for `isinstance` checks (`Option` itself is a type alias)."""


def is_option(value: object) -> TypeIs[Option[Any]]:
    """Returns True if `value` is a Some or Nil.

    Examples
    --------
    >>> is_option(Nil()), is_option(None)
    (True, False)
    """
    return isinstance(value, OPTION_TYPES)


def is_some(option: Option[U]) -> TypeIs[Some[U]]:
    """Returns True if the Option Monad is Some, narrowing it to `Some` for type checkers.

    Examples
    --------
    >>> val: Option[int] = Some(1)
    >>> if is_some(val):
    ...     print(val.value)
    1
    """
    return isinstance(option, Some)


def is_nil(option: Option[U]) -> TypeIs[Nil]:
    """Returns True if the Option Monad is Nil, narrowing it to `Nil` for type checkers.

    Examples
    --------
    >>> is_nil(Nil())
    True
    """
    return isinstance(option, Nil)


def from_optional(value: U | None) -> Option[U]:
    """Creates an Option Monad from an optional value: Nil for None, otherwise Some.

    Parameters
    ----------
    value: U | None
        Optional value.

    Returns
    -------
    option: Option[U]

    Examples
    --------
    >>> from_optional(None)
    Nil()
    >>> from_optional(1)
    Some(1)
    """
    if value is None:
        return Nil()
    return Some(value)


def map2(
    first: Option[U], second: Option[V], function: Callable[[U, V], W]
) -> Option[W]:
    """Applies a two-argument function to the values of two Option Monads if both are Some.

    Examples
    --------
    >>> map2(Some(1), Some(2), lambda a, b: a + b)
    Some(3)
    >>> map2(Some(1), Nil(), lambda a, b: a + b)
    Nil()
    """
    if isinstance(first, Some) and isinstance(second, Some):
        return Some(function(first.value, second.value))
    return Nil()


def map3(
    first: Option[U],
    second: Option[V],
    third: Option[W],
    function: Callable[[U, V, W], Y],
) -> Option[Y]:
    """Applies a three-argument function to the values of three Option Monads if all are Some.

    Examples
    --------
    >>> map3(Some(1), Some(2), Some(3), lambda a, b, c: a + b + c)
    Some(6)
    """
    if isinstance(first, Some) and isinstance(second, Some) and isinstance(third, Some):
        return Some(function(first.value, second.value, third.value))
    return Nil()


@overload
def safe(function: Callable[[], U]) -> Option[U]: ...


@overload
def safe(
    function: Callable[[], U], *, exceptions: tuple[type[BaseException], ...]
) -> Option[U]: ...


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

    Examples
    --------
    >>> safe(lambda: 1)
    Some(1)
    >>> safe(lambda: 1 / 0)
    Nil()
    >>> safe(lambda: int("x"), exceptions=(ValueError,))
    Nil()
    """
    try:
        return Some(function())
    except exceptions:
        return Nil()


# Curried module-level functions, for point-free pipelines (see `pymoliath.util.flow`).


def map(function: Callable[[U], V]) -> Callable[[Option[U]], Option[V]]:
    """Curried `Option.map`.

    Examples
    --------
    >>> map(lambda x: x + 1)(Some(1))
    Some(2)
    """
    return lambda option: option.map(function)


def bind(function: Callable[[U], Option[V]]) -> Callable[[Option[U]], Option[V]]:
    """Curried `Option.bind`.

    Examples
    --------
    >>> bind(lambda x: Some(x + 1))(Some(1))
    Some(2)
    """
    return lambda option: option.bind(function)


def filter(function: Callable[[U], bool]) -> Callable[[Option[U]], Option[U]]:
    """Curried `Option.filter`.

    Examples
    --------
    >>> filter(lambda x: x > 1)(Some(1))
    Nil()
    """
    return lambda option: option.filter(function)


def unwrap_or(default_value: U) -> Callable[[Option[U]], U]:
    """Curried `Option.unwrap_or`.

    Examples
    --------
    >>> unwrap_or(0)(Nil())
    0
    """
    return lambda option: option.unwrap_or(default_value)


def unwrap_or_else(function: Callable[[], U]) -> Callable[[Option[U]], U]:
    """Curried `Option.unwrap_or_else`.

    Examples
    --------
    >>> unwrap_or_else(lambda: 0)(Nil())
    0
    """
    return lambda option: option.unwrap_or_else(function)


def inspect(function: Callable[[U], None]) -> Callable[[Option[U]], Option[U]]:
    """Curried `Option.inspect`.

    Examples
    --------
    >>> inspect(print)(Some(1))
    1
    Some(1)
    """
    return lambda option: option.inspect(function)


# Imported last: result.py imports Some/Nil from this module, so the cycle resolves once at import
# time instead of on every call.
from pymoliath.result import Err, Ok, Result  # noqa: E402
