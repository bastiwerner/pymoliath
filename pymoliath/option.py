"""
# Option Monad

The Option Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Rust: [Option](https://doc.rust-lang.org/std/option/)
* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

This implementation is heavily inspired by the Rust `Option` type and the Haskell `Maybe` monad -
see `pymoliath.maybe` for the sibling implementation using `Just`/`Nothing` naming.

The `Option` type is a sealed sum type that can be either `Some` or `Nil`. Like in Rust, both
variants carry the type parameter:

```python
type Option[T] = Some[T] | Nil[T]
```

## Typing like in Rust

Because both variants know the full `Option[T]`, lambdas passed to `map`/`bind`/... are inferred
cleanly and a `match` over `Some`/`Nil` is exhaustive. A bare `Nil()` leaves its type open
(`Nil[Unknown]`) so it can be solved from context, e.g. in
`lambda x: Some(x) if x > 0 else Nil()`. Like in Rust, annotate an empty value that has no context:

```python
empty: Option[int] = Nil()

def find(value: int) -> Option[int]:
    return Some(value) if value > 0 else Nil()
```

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
`Option` monad. `Some` and `Nil` are the only variants (the Option is sealed), so a `match` over
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
    from pymoliath.result import Result

# Old-style TypeVars are invariant by default, which is what Option needs anyway
# (T also appears in parameter positions, e.g. unwrap_or and or_).
T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


class _OptionImpl(Generic[T]):
    """Shared implementation of the Option Monad - `Some` and `Nil` are its only subclasses."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Runtime "sealed": only Some and Nil (defined in this module) may subclass.
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("Option cannot be subclassed; use Some or Nil")

    def _as_option(self) -> Option[T]:
        # Safe: Some and Nil are the only subclasses (see __init_subclass__).
        return cast("Option[T]", self)

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
        match o := self._as_option():
            case Some(value):
                return Some(function(value))
            case Nil():
                return Nil()
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some(value):
                return function(value)
            case Nil():
                return Nil()
            case _:
                assert_never(o)

    def apply(self, applicative: Option[Callable[..., U]]) -> Option[U]:
        """Applies the passed applicative wrapping a function if the Option Monad is Some, otherwise
        returns Nil. Functions of several arguments are curried.

        Parameters
        ----------
        applicative: Option[Callable[[T], U]]
            Applicative Option Monad which contains a function.

        Returns
        -------
        option: Option[U]
            Returns an Option Monad from the applied function if both are Some, otherwise Nil.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> func: Option[Callable[[int], int]] = Some(lambda x: x * 2)
        >>> val.apply(func)
        Some(20)
        """
        return applicative.bind(lambda function: self.map(curry(function)))

    def apply2(
        self: _OptionImpl[Callable[..., U]], applicative_value: Option[Any]
    ) -> Option[U]:
        """Applies the function wrapped in this Option Monad to the passed Option Monad wrapping a
        value if both are Some, otherwise returns Nil. Functions of several arguments are curried.

        Parameters
        ----------
        applicative_value: Option[Any]
            Option monad which contains a value.

        Returns
        -------
        option: Option[U]
            Returns an Option Monad from the applied function if both are Some, otherwise Nil.

        Examples
        --------
        >>> func: Option[Callable[[int], int]] = Some(lambda y: 10 + y)
        >>> val: Option[int] = Some(5)
        >>> func.apply2(val)
        Some(15)
        >>> empty: Option[int] = Nil()
        >>> func.apply2(empty)
        Nil()
        """
        return self.bind(lambda function: applicative_value.map(curry(function)))

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
        match o := self._as_option():
            case Some(value) if filter_function(value):
                return o
            case Some() | Nil():
                return Nil()
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some(value):
                return function(value)
            case Nil():
                return False
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some(value):
                return function(value)
            case Nil():
                return default_value
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some():
                return other
            case Nil():
                return Nil()
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some():
                return o
            case Nil():
                return other
            case _:
                assert_never(o)

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
        return self.bind(
            lambda value: other.map(lambda other_value: (value, other_value))
        )

    @overload
    def flatten(self: _OptionImpl[Some[U]]) -> Option[U]: ...

    @overload
    def flatten(self: _OptionImpl[Nil[U]]) -> Option[U]: ...

    @overload
    def flatten(self: _OptionImpl[Option[U]]) -> Option[U]: ...

    @overload
    def flatten(self: Nil[Any]) -> Option[Never]: ...

    def flatten(self: _OptionImpl[Any]) -> Option[Any]:
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
        match o := self._as_option():
            case Some(value):
                return cast("Option[Any]", value)
            case Nil():
                return Nil()
            case _:
                assert_never(o)

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
        from pymoliath.result import Err, Ok

        match o := self._as_option():
            case Some(value):
                return Ok(value)
            case Nil():
                return Err(err_value)
            case _:
                assert_never(o)

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
        Err(missing)
        """
        from pymoliath.result import Err, Ok

        match o := self._as_option():
            case Some(value):
                return Ok(value)
            case Nil():
                return Err(err_function())
            case _:
                assert_never(o)

    def unwrap(self) -> T:
        """Returns the Some value, or otherwise raises an Exception.

        Returns
        -------
        result: T
            Returns the Some value.

        Raises
        ------
        Exception
            If the Option Monad is Nil.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.unwrap()
        1
        """
        match o := self._as_option():
            case Some(value):
                return value
            case Nil():
                raise Exception("Unwrap error on Option monad")
            case _:
                assert_never(o)

    def unwrap_or(self, default_value: T) -> T:
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
        match o := self._as_option():
            case Some(value):
                return value
            case Nil():
                return default_value
            case _:
                assert_never(o)

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
        match o := self._as_option():
            case Some(value):
                return value
            case Nil():
                return nothing_function()
            case _:
                assert_never(o)

    def inspect(self, function: Callable[[T], None]) -> Option[T]:
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
        match o := self._as_option():
            case Some(value):
                function(value)
            case Nil():
                pass
            case _:
                assert_never(o)
        return o

    def match(
        self, some_function: Callable[[T], U], nothing_function: Callable[[], U]
    ) -> U:
        """Matches the Option Monad to either a Some function or a Nil function with the same
        return type.

        Parameters
        ----------
        some_function: Callable[[T], U]
            Callback function for Option monads of type Some
        nothing_function: Callable[[], U]
            Callback function for Option monads of type Nil

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.match(lambda x: f"Some: {x}", lambda: "Nil")
        'Some: 10'
        """
        match o := self._as_option():
            case Some(value):
                return some_function(value)
            case Nil():
                return nothing_function()
            case _:
                assert_never(o)

    def is_nothing(self) -> bool:
        """Returns True if the Option Monad is Nil, otherwise False.

        Examples
        --------
        >>> val: Option[int] = Nil()
        >>> val.is_nothing()
        True
        """
        return isinstance(self, Nil)

    def is_some(self) -> bool:
        """Returns True if the Option Monad is Some, otherwise False.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.is_some()
        True
        """
        return isinstance(self, Some)

    def to_optional(self) -> T | None:
        """Converts the Option Monad into an optional value: the Some value or None.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.to_optional()
        5
        """
        match o := self._as_option():
            case Some(value):
                return value
            case Nil():
                return None
            case _:
                assert_never(o)

    @staticmethod
    def from_optional(value: U | None) -> Option[U]:
        """Creates an Option Monad from an optional value: Nil for None, otherwise Some.

        Examples
        --------
        >>> Some.from_optional(None)
        Nil()
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Option Monad.

        Examples
        --------
        >>> str(Some(42))
        'Some(42)'
        >>> str(Nil())
        'Nil()'
        """
        match o := self._as_option():
            case Some(value):
                return f"Some({value})"
            case Nil():
                return "Nil()"
            case _:
                assert_never(o)

    def __repr__(self) -> str:
        """Returns the string representation of the Option Monad (same as __str__)."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is the same variant wrapping a value of the same type and string.

        Examples
        --------
        >>> Some(1) == Some(1)
        True
        >>> Some(1) == Nil()
        False
        """
        if type(self) is not type(other):
            return False
        match o := self._as_option():
            case Some(value):
                other_value = cast("Some[Any]", other).value
                return type(value) is type(other_value) and str(value) == str(
                    other_value
                )
            case Nil():
                return True
            case _:
                assert_never(o)

    __hash__ = None  # type: ignore[assignment]


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Some(_OptionImpl[T]):
    """The Some variant of the Option Monad, wrapping a value.

    Examples
    --------
    >>> Some(42)
    Some(42)
    """

    value: T


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Nil(_OptionImpl[T]):
    """The Nil variant of the Option Monad, representing the absence of a value (a singleton).

    Examples
    --------
    >>> Nil()
    Nil()
    """

    _instance: ClassVar[Nil[Any] | None] = None

    def __new__(cls) -> Nil[T]:
        # Nil carries no data, so all instances (of any T) are the same object.
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cast("Nil[T]", cls._instance)


# Nil's type parameter has deliberately no default: if it defaulted to Never, pyright would pin a
# lambda's type to Never as soon as one branch returns Nil (`lambda x: Some(x) if x else Nil()`).
type Option[T] = Some[T] | Nil[T]


def from_optional(value: T | None) -> Option[T]:
    """Creates an Option Monad from an optional value: Nil for None, otherwise Some.

    Parameters
    ----------
    value: T | None
        Optional value.

    Returns
    -------
    option: Option[T]

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


def safe(function: Callable[[], T]) -> Option[T]:
    """Calls function and wraps its return value in Some, or returns Nil if it raises an Exception.

    Parameters
    ----------
    function: Callable[[], T]
        Zero-argument function which may raise an Exception.

    Returns
    -------
    option: Option[T]

    Examples
    --------
    >>> safe(lambda: 1)
    Some(1)
    >>> safe(lambda: 1 / 0)
    Nil()
    """
    try:
        return Some(function())
    except Exception:
        return Nil()
