"""
# Either Monad

The Either Monad is a container used to represent computations that can result in one of two values: a success value or a failure value.
It encapsulates values that could be `Left` (typically representing an error) or `Right` (representing the successful result),
allowing for a functional approach to error handling by chaining operations without constant explicit try-except or error checks.

* Haskell: [Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data.Either.html)
* Rust: [Result](https://doc.rust-lang.org/std/result/)

This implementation is heavily inspired by the Haskell `Either` type and the Rust `Result` type.

The `Either` type is a sealed sum type that can be either `Left` or `Right`. Like Rust's
`Result`, both variants carry both type parameters:

```python
type Either[L, R] = Left[L, R] | Right[R, L]
```

A bare `Right(10)` is a `Right[int, Never]`. A bare `Left("e")` leaves its Right type open
(`Left[str, Unknown]`) so it can be solved from context, e.g. in
`lambda x: Right(x) if x > 0 else Left("negative")`.

## Typing like in Rust

Because both variants know the full `Either[L, R]`, lambdas passed to `map`/`bind`/... are
inferred cleanly and a `match` over `Left`/`Right` is exhaustive. The one trade-off is invariance:
a bare `Right(10)` without any context is a `Right[int, Never]` and not assignable to an
`Either[str, int]`. Construct values directly in a `return` or an annotated assignment, just as
Rust needs a type annotation there:

```python
value: Either[str, int] = Right(10)

def parse(text: str) -> Either[str, int]:
    return Right(int(text)) if text.isdigit() else Left("not a number")
```

## Practical Examples and Benefits:

The Either Monad is particularly useful in scenarios where a function might fail and return an error (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without checking `if error` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Left`, the subsequent operations are skipped automatically, and the final result will be `Left`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly, making the code more robust against unhandled exceptions and making the flow of data more transparent.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if" logic into a linear pipeline of transformations.

```python
# Without Either (Imperative)
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

# With Either (Functional)
(get_user(user_id)
    .bind(get_profile)
    .bind(get_permission)
    .unwrap_or_else(lambda error: f"Error: {error}"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of an
`Either` monad. `Left` and `Right` are the only variants (the Either is sealed), so a `match` over
both is exhaustive and type checkers narrow the value in each branch.

```python
match either_value:
    case Right(x):
        # This block executes if the operation was successful
        print(f"Success value: {x}")
    case Left(y):
        # This block executes if the operation failed
        print(f"Error encountered: {y}")
```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
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
    from pymoliath.maybe import Maybe

# Old-style TypeVars are invariant by default, which is what Either needs anyway
# (L and R also appear in parameter positions, e.g. unwrap_or and bind).
L = TypeVar("L")
R = TypeVar("R")
U = TypeVar("U")
F = TypeVar("F")

# L with a default of Never (PEP 696): a bare `Right(10)` is a `Right[int, Never]`, so - like in
# Rust - it needs an annotation before it can be combined with fallible code.
L_Never = TypeVar("L_Never", default=Never)


class _EitherImpl(Generic[L, R]):
    """Shared implementation of the Either Monad - `Left` and `Right` are its only subclasses."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Runtime "sealed": only Left and Right (defined in this module) may subclass.
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("Either cannot be subclassed; use Left or Right")

    def _as_either(self) -> Either[L, R]:
        # Safe: Left and Right are the only subclasses (see __init_subclass__).
        return cast("Either[L, R]", self)

    def map(self, function: Callable[[R], U]) -> Either[L, U]:
        """Calls function on a wrapped Right value, otherwise leaving the Left value untouched.

        Parameters
        ----------
        function: Callable[[R], U]
            Function which takes a value of R and returns a value of type U.

        Returns
        -------
        either: Either[L, U]
            Returns a Right with the function result or otherwise the Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.map(lambda x: x + 5)
        Right(15)
        >>> err: Either[str, int] = Left("Error")
        >>> err.map(lambda x: x + 5)
        Left(Error)
        """
        match e := self._as_either():
            case Left(value):
                return Left(value)
            case Right(value):
                return Right(function(value))
            case _:
                assert_never(e)

    def map_left(self, function: Callable[[L], F]) -> Either[F, R]:
        """Calls function on a wrapped Left value, otherwise leaving the Right value untouched.

        Parameters
        ----------
        function: Callable[[L], F]
            Function which takes a value of L and returns a value of F.

        Returns
        -------
        either: Either[F, R]
            Returns a Left with the function result or otherwise the Right.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.map_left(lambda x: x.upper())
        Right(10)
        >>> err: Either[str, int] = Left("Error")
        >>> err.map_left(lambda x: x.upper())
        Left(ERROR)
        """
        match e := self._as_either():
            case Left(value):
                return Left(function(value))
            case Right(value):
                return Right(value)
            case _:
                assert_never(e)

    def bind(self, function: Callable[[R], Either[L, U]]) -> Either[L, U]:
        """Calls function if the Either Monad is Right, otherwise returns the Left.

        Parameters
        ----------
        function: Callable[[R], Either[L, U]]
            Function which takes a value of R and returns a new Either Monad.

        Returns
        -------
        either: Either[L, U]
            Returns the function result if Right, otherwise the Left.

        Examples
        --------
        >>> def get_next(x: int) -> Either[str, int]:
        ...     return Right(x + 1) if x < 10 else Left("too big")
        >>> val: Either[str, int] = Right(5)
        >>> val.bind(get_next)
        Right(6)
        >>> big: Either[str, int] = Right(10)
        >>> big.bind(get_next)
        Left(too big)
        """
        match e := self._as_either():
            case Left(value):
                return Left(value)
            case Right(value):
                return function(value)
            case _:
                assert_never(e)

    def bind_left(self, function: Callable[[L], Either[F, R]]) -> Either[F, R]:
        """Calls function if the Either Monad is Left, otherwise returns the Right.

        Parameters
        ----------
        function: Callable[[L], Either[F, R]]
            Function which takes a value of L and returns a new Either Monad.

        Returns
        -------
        either: Either[F, R]
            Returns the function result if Left, otherwise the Right.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.bind_left(lambda err: Left(f"{err} bind"))
        Right(10)
        >>> err: Either[str, int] = Left("Error")
        >>> err.bind_left(lambda err: Left(f"{err} bind"))
        Left(Error bind)
        """
        match e := self._as_either():
            case Left(value):
                return function(value)
            case Right(value):
                return Right(value)
            case _:
                assert_never(e)

    def apply(self, applicative: Either[F, Callable[..., U]]) -> Either[L | F, U]:
        """Applies the passed applicative wrapping a function if the Either Monad is Right,
        otherwise returns the (first) Left. Functions of several arguments are curried.

        Unlike `bind`, the Left types of both sides may differ (the result carries either), so
        bare applicatives such as `Right(lambda x: x)` combine without annotations.

        Parameters
        ----------
        applicative: Either[F, Callable[[R], U]]
            Applicative Either Monad which contains a function.

        Returns
        -------
        either: Either[L | F, U]
            Returns an Either Monad from the applied function if Right, otherwise a Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> func: Either[str, Callable[[int], int]] = Right(lambda x: x * 2)
        >>> val.apply(func)
        Right(20)
        """
        applicative_: Either[L | F, Callable[..., U]] = cast(Any, applicative)
        self_: Either[L | F, R] = cast(Any, self)
        return applicative_.bind(lambda function: self_.map(curry(function)))

    def apply2(
        self: _EitherImpl[L, Callable[..., U]], applicative_value: Either[F, Any]
    ) -> Either[L | F, U]:
        """Applies the function wrapped in this Either Monad to the passed Either Monad wrapping a
        value if both are Right, otherwise returns the (first) Left. Functions of several arguments
        are curried. As with `apply`, the Left types of both sides may differ.

        Parameters
        ----------
        applicative_value: Either[F, Any]
            Either Monad which contains a value.

        Returns
        -------
        either: Either[L | F, U]
            Returns an Either Monad from the applied function if Right, otherwise a Left.

        Examples
        --------
        >>> func: Either[str, Callable[[int], int]] = Right(lambda x: x + 1)
        >>> val: Either[str, int] = Right(10)
        >>> func.apply2(val)
        Right(11)
        """
        self_: Either[L | F, Callable[..., U]] = cast(Any, self)
        value_: Either[L | F, Any] = cast(Any, applicative_value)
        return self_.bind(lambda function: value_.map(curry(function)))

    def is_right_and(self, function: Callable[[R], bool]) -> bool:
        """Returns True if the Either Monad is Right and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[R], bool]
            Predicate function applied to the Right value.

        Returns
        -------
        result: bool
            Returns the predicate result if Right, otherwise False.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_right_and(lambda x: x > 5)
        True
        """
        match e := self._as_either():
            case Left():
                return False
            case Right(value):
                return function(value)
            case _:
                assert_never(e)

    def is_left_and(self, function: Callable[[L], bool]) -> bool:
        """Returns True if the Either Monad is Left and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[L], bool]
            Predicate function applied to the Left value.

        Returns
        -------
        result: bool
            Returns the predicate result if Left, otherwise False.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_left_and(lambda x: x == "Error")
        False
        """
        match e := self._as_either():
            case Left(value):
                return function(value)
            case Right():
                return False
            case _:
                assert_never(e)

    def map_or(self, default_value: U, function: Callable[[R], U]) -> U:
        """Applies the function to the Right value, or returns the default value if Left.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Either Monad is Left.
        function: Callable[[R], U]
            Function applied to the Right value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.map_or(0, lambda x: x + 5)
        15
        """
        match e := self._as_either():
            case Left():
                return default_value
            case Right(value):
                return function(value)
            case _:
                assert_never(e)

    def and_(self, other: Either[L, U]) -> Either[L, U]:
        """Returns `other` if the Either Monad is Right, otherwise the Left.

        Parameters
        ----------
        other: Either[L, U]
            Either Monad to be returned if this Either Monad is Right.

        Returns
        -------
        either: Either[L, U]
            Returns `other` or the Left.

        Examples
        --------
        >>> val1: Either[str, int] = Right(1)
        >>> val2: Either[str, int] = Right(2)
        >>> val1.and_(val2)
        Right(2)
        """
        match e := self._as_either():
            case Left(value):
                return Left(value)
            case Right():
                return other
            case _:
                assert_never(e)

    def or_(self, other: Either[F, R]) -> Either[F, R]:
        """Returns this Either Monad if it is Right, otherwise `other`.

        Parameters
        ----------
        other: Either[F, R]
            Either Monad to be returned if this Either Monad is Left.

        Returns
        -------
        either: Either[F, R]
            Returns the Right or `other`.

        Examples
        --------
        >>> val1: Either[str, int] = Right(1)
        >>> val2: Either[str, int] = Right(2)
        >>> val1.or_(val2)
        Right(1)
        """
        match e := self._as_either():
            case Left():
                return other
            case Right(value):
                return Right(value)
            case _:
                assert_never(e)

    def zip(self, other: Either[L, U]) -> Either[L, tuple[R, U]]:
        """Combines this Either Monad with another into an Either Monad of a tuple, or the first
        Left.

        Parameters
        ----------
        other: Either[L, U]
            Either Monad to be zipped with this Either Monad.

        Returns
        -------
        either: Either[L, tuple[R, U]]
            Returns Right of a tuple of both values, or Left.

        Examples
        --------
        >>> val1: Either[str, int] = Right(1)
        >>> val2: Either[str, int] = Right(2)
        >>> val1.zip(val2)
        Right((1, 2))
        """
        return self.bind(
            lambda value: other.map(lambda other_value: (value, other_value))
        )

    @overload
    def flatten(self: _EitherImpl[L, Right[U, F]]) -> Either[L | F, U]: ...

    @overload
    def flatten(self: _EitherImpl[L, Left[F, U]]) -> Either[L | F, U]: ...

    @overload
    def flatten(self: _EitherImpl[L, Either[F, U]]) -> Either[L | F, U]: ...

    @overload
    def flatten(self: Left[L, Any]) -> Either[L, Never]: ...

    def flatten(self: _EitherImpl[Any, Any]) -> Either[Any, Any]:
        """Flattens a nested Either Monad by one level.

        Returns
        -------
        either: Either[L, U]
            Returns the nested Either Monad if Right, otherwise the Left.

        Examples
        --------
        >>> Right(Right(10)).flatten()
        Right(10)
        """
        match e := self._as_either():
            case Left(value):
                return Left(value)
            case Right(value):
                return cast("Either[Any, Any]", value)
            case _:
                assert_never(e)

    def right(self) -> Maybe[R]:
        """Converts the Either Monad into a Maybe Monad, discarding any Left value.

        Returns
        -------
        maybe: Maybe[R]
            Returns Just with the Right value, or Nothing if Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.right()
        Just(10)
        """
        from pymoliath.maybe import Just, Nothing

        match e := self._as_either():
            case Left():
                return Nothing()
            case Right(value):
                return Just(value)
            case _:
                assert_never(e)

    def left(self) -> Maybe[L]:
        """Converts the Either Monad into a Maybe Monad of the Left value, discarding any Right
        value.

        Returns
        -------
        maybe: Maybe[L]
            Returns Just with the Left value, or Nothing if Right.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.left()
        Nothing()
        """
        from pymoliath.maybe import Just, Nothing

        match e := self._as_either():
            case Left(value):
                return Just(value)
            case Right():
                return Nothing()
            case _:
                assert_never(e)

    def unwrap(self) -> R:
        """Returns the Right value, or otherwise raises an Exception with the Left value.

        Returns
        -------
        result: R
            Returns the Right value.

        Raises
        ------
        Exception
            If the Either Monad is Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap()
        10
        """
        match e := self._as_either():
            case Left(value):
                raise Exception(value)
            case Right(value):
                return value
            case _:
                assert_never(e)

    def unwrap_or(self, default_value: R) -> R:
        """Returns the Right value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: R
            Default value of R

        Returns
        -------
        result: R
            Returns the Right value or the default value.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap_or(5)
        10
        """
        match e := self._as_either():
            case Left():
                return default_value
            case Right(value):
                return value
            case _:
                assert_never(e)

    def unwrap_or_else(self, left_function: Callable[[L], R]) -> R:
        """Returns the Right value, or otherwise calls the left_function with the Left value.

        Parameters
        ----------
        left_function: Callable[[L], R]
            Function which will be called if the Either Monad is Left.

        Returns
        -------
        result: R
            Returns the Right value or the left_function result.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap_or_else(lambda _: 0)
        10
        """
        match e := self._as_either():
            case Left(value):
                return left_function(value)
            case Right(value):
                return value
            case _:
                assert_never(e)

    def unwrap_left_or(self, default_value: L) -> L:
        """Returns the Left value, or otherwise a provided default value of L.

        Parameters
        ----------
        default_value: L
            Default value of L

        Returns
        -------
        result: L
            Returns the Left value or the default value.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap_left_or("Default")
        'Default'
        """
        match e := self._as_either():
            case Left(value):
                return value
            case Right():
                return default_value
            case _:
                assert_never(e)

    def inspect(self, function: Callable[[R], None]) -> Either[L, R]:
        """Calls function with the Right value (if any) and returns the Either Monad unchanged.

        Parameters
        ----------
        function: Callable[[R], None]
            Inspection function which takes the Right value of the Either Monad

        Returns
        -------
        either: Either[L, R]

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.inspect(lambda x: print(f"Value: {x}"))
        Value: 10
        Right(10)
        """
        match e := self._as_either():
            case Left():
                pass
            case Right(value):
                function(value)
            case _:
                assert_never(e)
        return e

    def inspect_left(self, function: Callable[[L], None]) -> Either[L, R]:
        """Calls function with the Left value (if any) and returns the Either Monad unchanged.

        Parameters
        ----------
        function: Callable[[L], None]
            Inspection function which takes the Left value of the Either Monad

        Returns
        -------
        either: Either[L, R]

        Examples
        --------
        >>> val: Either[str, int] = Left("boom")
        >>> val.inspect_left(lambda x: print(f"Left: {x}"))
        Left: boom
        Left(boom)
        """
        match e := self._as_either():
            case Left(value):
                function(value)
            case Right():
                pass
            case _:
                assert_never(e)
        return e

    def match(
        self, left_function: Callable[[L], U], right_function: Callable[[R], U]
    ) -> U:
        """Matches the Either Monad to either a Left function or a Right function with the same
        return type.

        Parameters
        ----------
        left_function: Callable[[L], U]
            Callback function for Either monads of type Left
        right_function: Callable[[R], U]
            Callback function for Either monads of type Right

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.match(lambda x: "Error", lambda x: f"Success: {x}")
        'Success: 10'
        """
        match e := self._as_either():
            case Left(value):
                return left_function(value)
            case Right(value):
                return right_function(value)
            case _:
                assert_never(e)

    def is_left(self) -> bool:
        """Returns True if the Either Monad is Left, otherwise False.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_left()
        False
        """
        return isinstance(self, Left)

    def is_right(self) -> bool:
        """Returns True if the Either Monad is Right, otherwise False.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_right()
        True
        """
        return isinstance(self, Right)

    def _payload(self) -> object:
        match e := self._as_either():
            case Left(value):
                return value
            case Right(value):
                return value
            case _:
                assert_never(e)

    def __str__(self) -> str:
        """Returns the string representation of the Either Monad.

        Examples
        --------
        >>> str(Right(10))
        'Right(10)'
        >>> str(Left("Error"))
        'Left(Error)'
        """
        return f"{type(self).__name__}({self._payload()})"

    def __repr__(self) -> str:
        """Returns the string representation of the Either Monad (same as __str__)."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is the same variant wrapping a value of the same type and string.

        Examples
        --------
        >>> Right(10) == Right(10)
        True
        >>> Right(10) == Left(10)
        False
        """
        if type(self) is not type(other):
            return False
        value, other_value = (
            self._payload(),
            cast("_EitherImpl[Any, Any]", other)._payload(),
        )
        return type(value) is type(other_value) and str(value) == str(other_value)

    __hash__ = None  # type: ignore[assignment]


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Left(_EitherImpl[L, R]):
    """The Left variant of the Either Monad, typically wrapping an error value.

    Examples
    --------
    >>> Left("Error Message")
    Left(Error Message)
    """

    value: L


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Right(
    _EitherImpl[L_Never, R], Generic[R, L_Never]
):  # R first: no default after a default
    """The Right variant of the Either Monad, wrapping the successful value.

    Examples
    --------
    >>> Right(10)
    Right(10)
    """

    value: R


# Left's Right type has deliberately no default: if it defaulted to Never, pyright would pin a
# lambda's Right type to Never as soon as one branch returns a Left
# (`lambda x: Right(x) if x else Left("zero")`).
type Either[LeftT, RightT] = Left[LeftT, RightT] | Right[RightT, LeftT]


def either_safe(function: Callable[[], R]) -> Either[Exception, R]:
    """Calls function and wraps its return value in Right, or a raised Exception in Left.

    Parameters
    ----------
    function: Callable[[], R]
        Zero-argument function which may raise an Exception.

    Returns
    -------
    either: Either[Exception, R]

    Examples
    --------
    >>> either_safe(lambda: 1)
    Right(1)
    >>> either_safe(lambda: 1 / 0)
    Left(division by zero)
    """
    try:
        return Right(function())
    except Exception as e:
        return Left(e)
