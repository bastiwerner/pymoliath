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
type Either[L, R] = Left[L, R] | Right[L, R]
```

## Typing like in Rust

Both type parameters are covariant and the side a variant does not use defaults to `Never`: a bare
`Right(10)` is a `Right[Never, int]` and a bare `Left("e")` is a `Left[str, Never]`. Both are
assignable to an `Either[str, int]`, and a conditional lambda such as
`lambda x: Right(x) if x > 0 else Left("negative")` is inferred as an `Either[str, int]`.

Like in Rust, the Left type is fixed by the receiver: `bind` on an `Either[str, int]` needs a
function returning an `Either[str, U]`, and a bare `Right(10)` (Left type `Never`) needs an
annotation before it can be bound to fallible code:

```python
value: Either[str, int] = Right(10)

def parse(text: str) -> Either[str, int]:
    return Right(int(text)) if text.isdigit() else Left("not a number")
```

Methods called directly on a variant keep the precise variant type, e.g. `Right(1).map(str)` is a
`Right[Never, str]` and `Left("e").map(str)` a `Left[str, str]`.

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
`Either` monad. `Left` and `Right` are the only variants (both are `@final`), so a `match` over
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

`Either` is a type alias, so check it at runtime with `isinstance(x, (Left, Right))`. Both a `match`
over `Left`/`Right` and `isinstance(x, Left)` narrow the type to the variant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, Self, final, overload

from typing_extensions import Generic, TypeVar

# either.py and maybe.py convert into each other. Importing the module (not its names) lets the
# cycle resolve at import time; its attributes are looked up when the conversions run.
import pymoliath.maybe as _maybe
from pymoliath.errors import UnwrapError

if TYPE_CHECKING:
    from pymoliath.maybe import Just, Maybe, Nothing

# `Nothing()` is a singleton, but constructing it still runs `__new__` and `__init__`. Hot paths
# (`Left.right()`, `Right.left()`) return this cached reference instead. It is filled on first use
# because maybe.py may still be initializing while this module is imported.
_nothing: Nothing | None = None


def _load_nothing() -> Nothing:
    global _nothing
    _nothing = _maybe.Nothing()
    return _nothing


# Both parameters are covariant, so `Right[Never, int]` is an `Either[str, int]`. Like in Rust the
# receiver fixes the types a method accepts (`unwrap_or(default: R)`, `bind` returning
# `Either[L, U]`), which puts a covariant parameter in an input position. That is sound here: the
# containers are immutable and those arguments are only ever returned, typed by the receiver's
# (wider) view - the same reasoning as typeshed's `Sequence.index`. Those methods carry a
# `type: ignore` (misc).
#
# Short-circuit paths return the instance itself instead of allocating a new one. Only the unused
# (phantom) type parameter changes there, so they are typed through `Any` in that one slot - a
# `self: Left[LeftT, Any]` annotation or a `failed: Left[LeftT, Any] = ...` local - which costs
# nothing at runtime and keeps the other parameter checked.
L = TypeVar("L", covariant=True)
R = TypeVar("R", covariant=True)

# The type parameters of `Left` and `Right` (both `[L, R]`, like `Either`). Both default to Never
# (PEP 696), so the side a variant does not use is Never: a bare `Left("e")` is a
# `Left[str, Never]` and a bare `Right(1)` a `Right[Never, int]`. (A parameter with a default
# can't precede one without, so both have one.)
LeftT = TypeVar("LeftT", covariant=True, default=Never)
RightT = TypeVar("RightT", covariant=True, default=Never)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
F = TypeVar("F")
X = TypeVar("X", bound=BaseException)


class _EitherImpl(Generic[L, R]):
    """Public interface of the Either Monad. The behaviour lives in `Left` and `Right`, its only
    subclasses."""

    __slots__ = ()

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
        Left('Error')
        """
        raise NotImplementedError

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
        Left('ERROR')
        """
        raise NotImplementedError

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
        Left('too big')
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[R], Either[L, U]]) -> Either[L, U]:
        """Alias of `bind`, named like in Rust.

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
        >>> val: Either[str, int] = Right(5)
        >>> val.and_then(lambda x: Right(x * 2))
        Right(10)
        """
        raise NotImplementedError

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
        Left('Error bind')
        """
        raise NotImplementedError

    def or_else(self, function: Callable[[L], Either[F, R]]) -> Either[F, R]:
        """Alias of `bind_left`, named like in Rust.

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
        >>> err: Either[str, int] = Left("Error")
        >>> err.or_else(lambda e: Right(len(e)))
        Right(5)
        """
        raise NotImplementedError

    def apply(self, function: Either[L, Callable[[R], U]]) -> Either[L, U]:
        """Applies the function wrapped in `function` to the Right value if both are Right.

        If both are Left, the Left of `function` takes precedence. For functions of several
        arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: Either[L, Callable[[R], U]]
            Either Monad which contains a function.

        Returns
        -------
        either: Either[L, U]
            Returns Right of the function result if both are Right, otherwise the (first) Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> func: Either[str, Callable[[int], int]] = Right(lambda x: x * 2)
        >>> val.apply(func)
        Right(20)
        >>> Left("value").apply(Left("function"))
        Left('function')
        """
        raise NotImplementedError

    def apply2(
        self: _EitherImpl[L, Callable[[U], V]], value: Either[L, U]
    ) -> Either[L, V]:
        """Applies the function wrapped in this Either Monad to the value wrapped in `value`.

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both are
        empty/errors, this (the function side) takes precedence. Functions of several
        arguments can be applied one argument at a time when they are curried, e.g.
        `Right(lambda a: lambda b: a + b).apply2(x).apply2(y)`.

        Parameters
        ----------
        value: Either[L, U]
            Either Monad which contains the argument.

        Returns
        -------
        result: Either[L, V]

        Examples
        --------
        >>> func: Either[str, Callable[[int], int]] = Right(lambda y: 10 + y)
        >>> val: Either[str, int] = Right(5)
        >>> func.apply2(val)
        Right(15)
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[L], U], function: Callable[[R], U]
    ) -> U:
        """Applies the function to the Right value, or the default function to the Left value.

        Parameters
        ----------
        default_function: Callable[[L], U]
            Function applied to the Left value.
        function: Callable[[R], U]
            Function applied to the Right value.

        Returns
        -------
        result: U
            Returns the result of whichever function was applied.

        Examples
        --------
        >>> err: Either[str, int] = Left("Error")
        >>> err.map_or_else(lambda e: len(e), lambda x: x * 2)
        5
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[R], bool], left_value: L) -> Either[L, R]:  # type: ignore[misc]
        """Keeps the Right value if the predicate holds for it, otherwise returns Left(left_value).

        Parameters
        ----------
        predicate: Callable[[R], bool]
            Predicate function applied to the Right value.
        left_value: L
            Left value used if the predicate does not hold.

        Returns
        -------
        either: Either[L, R]
            Returns the Right, Left(left_value) if the predicate fails, or the original Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(-1)
        >>> val.filter(lambda x: x > 0, "negative")
        Left('negative')
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def swap(self) -> Either[R, L]:
        """Swaps the Left and the Right value.

        Returns
        -------
        either: Either[R, L]
            Returns Left of the Right value, or Right of the Left value.

        Examples
        --------
        >>> Right(1).swap()
        Left(1)
        >>> Left("e").swap()
        Right('e')
        """
        raise NotImplementedError

    def flatten(self: _EitherImpl[F, Either[F, U]]) -> Either[F, U]:
        """Flattens a nested Either Monad by one level.

        Returns
        -------
        either: Either[F, U]
            Returns the nested Either Monad if Right, otherwise the Left.

        Examples
        --------
        >>> Right(Right(10)).flatten()
        Right(10)
        """
        raise NotImplementedError

    def merge(self: _EitherImpl[U, U]) -> U:
        """Returns the Left or the Right value, for an Either whose both sides have the same type.

        Returns
        -------
        value: U
            The wrapped value.

        Examples
        --------
        >>> Left(1).merge()
        1
        """
        raise NotImplementedError

    def transpose(self: _EitherImpl[F, Maybe[U]]) -> Maybe[Either[F, U]]:
        """Transposes an Either of a Maybe into a Maybe of an Either.

        `Right(Nothing())` becomes `Nothing()`, `Right(Just(x))` becomes `Just(Right(x))` and
        `Left(e)` becomes `Just(Left(e))`.

        Returns
        -------
        maybe: Maybe[Either[F, U]]
            Returns the transposed Maybe Monad.

        Examples
        --------
        >>> from pymoliath.maybe import Just
        >>> Right(Just(1)).transpose()
        Just(Right(1))
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def unwrap(self) -> R:
        """Returns the Right value, or otherwise raises an `UnwrapError`.

        If the Left value is an exception, it is chained as the `__cause__` of the `UnwrapError`.

        Returns
        -------
        result: R
            Returns the Right value.

        Raises
        ------
        UnwrapError
            If the Either Monad is Left.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap()
        10
        """
        raise NotImplementedError

    def unwrap_or(self, default_value: R) -> R:  # type: ignore[misc]
        """Returns the Right value, or otherwise the provided default value.

        On an `Either[L, R]` the default must be an R. On a bare Left (whose Right type is Never)
        any default is accepted, e.g. `Left("e").unwrap_or(10)`.

        Parameters
        ----------
        default_value: R
            Default value returned if the Either Monad is Left.

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
        raise NotImplementedError

    def unwrap_or_else(self, function: Callable[[L], R]) -> R:
        """Returns the Right value, or otherwise calls function with the Left value.

        Like `unwrap_or`, a bare Left accepts a function returning any type.

        Parameters
        ----------
        function: Callable[[L], R]
            Function which will be called with the Left value if the Either Monad is Left.

        Returns
        -------
        result: R
            Returns the Right value or the function result.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.unwrap_or_else(lambda _: 0)
        10
        """
        raise NotImplementedError

    def unwrap_left_or(self, default_value: L) -> L:  # type: ignore[misc]
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
        raise NotImplementedError

    def inspect(self, function: Callable[[R], None]) -> Self:
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
        raise NotImplementedError

    def inspect_left(self, function: Callable[[L], None]) -> Self:
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
        Left('boom')
        """
        raise NotImplementedError

    def match(self, *, left: Callable[[L], U], right: Callable[[R], U]) -> U:
        """Matches the Either Monad to either the `left` or the `right` callback with the same
        return type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        left: Callable[[L], U]
            Callback function for Either monads of type Left
        right: Callable[[R], U]
            Callback function for Either monads of type Right

        Returns
        -------
        result: U
            Returns the result of the callback that was called.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.match(left=lambda x: "Error", right=lambda x: f"Success: {x}")
        'Success: 10'
        """
        raise NotImplementedError

    def is_left(self) -> bool:
        """Returns True if the Either Monad is Left, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_left()
        False
        """
        raise NotImplementedError

    def is_right(self) -> bool:
        """Returns True if the Either Monad is Right, otherwise False.

        Returns
        -------
        result: bool

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_right()
        True
        """
        raise NotImplementedError

    @staticmethod
    def from_maybe(maybe: Maybe[U], left_value: F) -> Either[F, U]:
        """Converts a Maybe Monad into an Either Monad: Just(x) becomes Right(x), Nothing becomes
        Left(left_value).

        Available on both variants (`Right.from_maybe` and `Left.from_maybe` are the same function).

        Parameters
        ----------
        maybe: Maybe[U]
            Maybe Monad to be converted.
        left_value: F
            Left value used if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[F, U]
            Returns Right of the Just value, or Left(left_value).

        Examples
        --------
        >>> from pymoliath.maybe import Just, Nothing
        >>> Right.from_maybe(Just(1), "missing")
        Right(1)
        >>> Right.from_maybe(Nothing(), "missing")
        Left('missing')
        """
        if isinstance(maybe, _maybe.Just):
            return Right(maybe.value)
        return Left(left_value)

    @staticmethod
    @overload
    def safe(function: Callable[[], U]) -> Either[Exception, U]: ...

    @staticmethod
    @overload
    def safe(
        function: Callable[[], U], *, exceptions: tuple[type[X], ...]
    ) -> Either[X, U]: ...

    @staticmethod
    def safe(
        function: Callable[[], U],
        *,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> Either[BaseException, U]:
        """Calls function and wraps its return value in Right, or a raised exception in Left.

        Parameters
        ----------
        function: Callable[[], U]
            Zero-argument function which may raise an exception.
        exceptions: tuple[type[X], ...]
            The exception types to catch (default: `Exception`). Any other exception propagates.

        Returns
        -------
        either: Either[X, U]

        Examples
        --------
        >>> Right.safe(lambda: 1)
        Right(1)
        >>> Right.safe(lambda: 1 / 0)
        Left(ZeroDivisionError('division by zero'))
        >>> Right.safe(lambda: int("x"), exceptions=(ValueError,)).is_left()
        True
        """
        try:
            return Right(function())
        except exceptions as e:
            return Left(e)


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Left(_EitherImpl[LeftT, RightT]):
    """The Left variant of the Either Monad, typically wrapping an error value.

    Equality and hashing compare the wrapped value. Exceptions compare by identity, so two
    separately raised but otherwise identical exceptions are not equal.

    Examples
    --------
    >>> Left("Error Message")
    Left('Error Message')
    >>> str(Left("Error"))
    'Left(Error)'
    """

    value: LeftT

    def __init__(self, value: LeftT) -> None:
        _set_left_value(self, value)

    def map(self: Left[LeftT, Any], function: Callable[[RightT], U]) -> Left[LeftT, U]:
        return self

    def map_left(self, function: Callable[[LeftT], F]) -> Left[F, RightT]:
        return Left(function(self.value))

    def bind(
        self: Left[LeftT, Any], function: Callable[[RightT], Either[LeftT, U]]
    ) -> Left[LeftT, U]:
        return self

    and_then = bind

    def bind_left(
        self, function: Callable[[LeftT], Either[F, RightT]]
    ) -> Either[F, RightT]:
        return function(self.value)

    or_else = bind_left

    def apply(
        self: Left[LeftT, Any], function: Either[LeftT, Callable[[RightT], U]]
    ) -> Left[LeftT, U]:
        if isinstance(function, Left):
            failed: Left[LeftT, Any] = function
            return failed
        return self

    def apply2(
        self: Left[LeftT, Callable[[U], V]], value: Either[LeftT, U]
    ) -> Left[LeftT, V]:
        failed: Left[LeftT, Any] = self
        return failed

    def is_right_and(self, function: Callable[[RightT], bool]) -> bool:
        return False

    def is_left_and(self, function: Callable[[LeftT], bool]) -> bool:
        return function(self.value)

    def map_or(self, default_value: U, function: Callable[[RightT], U]) -> U:
        return default_value

    def map_or_else(
        self, default_function: Callable[[LeftT], U], function: Callable[[RightT], U]
    ) -> U:
        return default_function(self.value)

    def filter(self, predicate: Callable[[RightT], bool], left_value: LeftT) -> Self:  # type: ignore[misc]
        return self

    def and_(self: Left[LeftT, Any], other: Either[LeftT, U]) -> Left[LeftT, U]:
        return self

    def or_(self, other: Either[F, U]) -> Either[F, U]:
        return other

    def zip(
        self: Left[LeftT, Any], other: Either[LeftT, U]
    ) -> Left[LeftT, tuple[RightT, U]]:
        return self

    def swap(self) -> Right[RightT, LeftT]:
        return Right(self.value)

    def flatten(self: Left[F, Either[F, U]]) -> Left[F, U]:
        failed: Left[F, Any] = self
        return failed

    def merge(self: Left[U, U]) -> U:
        return self.value

    def transpose(self: Left[F, Maybe[U]]) -> Just[Either[F, U]]:
        failed: Left[F, Any] = self
        return _maybe.Just(failed)

    def right(self) -> Nothing:
        return _nothing or _load_nothing()

    def left(self) -> Just[LeftT]:
        return _maybe.Just(self.value)

    def unwrap(self) -> Never:
        value = self.value
        message = f"called unwrap on {self!r}"
        if isinstance(value, BaseException):
            raise UnwrapError(self, message) from value
        raise UnwrapError(self, message)

    def unwrap_or(self, default_value: U) -> U:
        return default_value

    def unwrap_or_else(self, function: Callable[[LeftT], U]) -> U:
        return function(self.value)

    def unwrap_left_or(self, default_value: LeftT) -> LeftT:  # type: ignore[misc]
        return self.value

    def inspect(self, function: Callable[[RightT], None]) -> Self:
        return self

    def inspect_left(self, function: Callable[[LeftT], None]) -> Self:
        function(self.value)
        return self

    def match(self, *, left: Callable[[LeftT], U], right: Callable[[RightT], U]) -> U:
        return left(self.value)

    def is_left(self) -> bool:
        return True

    def is_right(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"Left({self.value})"

    def __repr__(self) -> str:
        return f"Left({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False, init=False)
class Right(_EitherImpl[LeftT, RightT]):
    """The Right variant of the Either Monad, wrapping the successful value.

    Its type parameters are `Right[L, R]`, in the same order as `Either` and `Left`, so a bare
    `Right(10)` is a `Right[Never, int]`.

    Equality and hashing compare the wrapped value (an unhashable value makes the Right
    unhashable).

    Examples
    --------
    >>> Right(10)
    Right(10)
    >>> str(Right("text"))
    'Right(text)'
    """

    value: RightT

    def __init__(self, value: RightT) -> None:
        _set_right_value(self, value)

    def map(self, function: Callable[[RightT], U]) -> Right[LeftT, U]:
        return Right(function(self.value))

    def map_left(
        self: Right[Any, RightT], function: Callable[[LeftT], F]
    ) -> Right[F, RightT]:
        return self

    def bind(self, function: Callable[[RightT], Either[LeftT, U]]) -> Either[LeftT, U]:
        return function(self.value)

    and_then = bind

    def bind_left(
        self: Right[Any, RightT], function: Callable[[LeftT], Either[F, RightT]]
    ) -> Right[F, RightT]:
        return self

    or_else = bind_left

    def apply(self, function: Either[LeftT, Callable[[RightT], U]]) -> Either[LeftT, U]:
        if isinstance(function, Right):
            return Right(function.value(self.value))
        failed: Left[LeftT, Any] = function
        return failed

    def apply2(
        self: Right[LeftT, Callable[[U], V]], value: Either[LeftT, U]
    ) -> Either[LeftT, V]:
        if isinstance(value, Right):
            return Right(self.value(value.value))
        failed: Left[LeftT, Any] = value
        return failed

    def is_right_and(self, function: Callable[[RightT], bool]) -> bool:
        return function(self.value)

    def is_left_and(self, function: Callable[[LeftT], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[RightT], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[LeftT], U], function: Callable[[RightT], U]
    ) -> U:
        return function(self.value)

    def filter(
        self,
        predicate: Callable[[RightT], bool],
        left_value: LeftT,  # type: ignore[misc]
    ) -> Either[LeftT, RightT]:
        return self if predicate(self.value) else Left(left_value)

    def and_(self, other: Either[LeftT, U]) -> Either[LeftT, U]:
        return other

    def or_(self: Right[Any, RightT], other: Either[F, RightT]) -> Right[F, RightT]:
        return self

    def zip(self, other: Either[LeftT, U]) -> Either[LeftT, tuple[RightT, U]]:
        if isinstance(other, Right):
            return Right((self.value, other.value))
        failed: Left[LeftT, Any] = other
        return failed

    def swap(self) -> Left[RightT, LeftT]:
        return Left(self.value)

    def flatten(self: Right[F, Either[F, U]]) -> Either[F, U]:
        return self.value

    def merge(self: Right[U, U]) -> U:
        return self.value

    def transpose(self: Right[F, Maybe[U]]) -> Maybe[Either[F, U]]:
        maybe = self.value
        if isinstance(maybe, _maybe.Just):
            return _maybe.Just(Right(maybe.value))
        return maybe

    def right(self) -> Just[RightT]:
        return _maybe.Just(self.value)

    def left(self) -> Nothing:
        return _nothing or _load_nothing()

    def unwrap(self) -> RightT:
        return self.value

    def unwrap_or(self, default_value: RightT) -> RightT:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, function: Callable[[LeftT], RightT]) -> RightT:
        return self.value

    def unwrap_left_or(self, default_value: F) -> F:
        return default_value

    def inspect(self, function: Callable[[RightT], None]) -> Self:
        function(self.value)
        return self

    def inspect_left(self, function: Callable[[LeftT], None]) -> Self:
        return self

    def match(self, *, left: Callable[[LeftT], U], right: Callable[[RightT], U]) -> U:
        return right(self.value)

    def is_left(self) -> bool:
        return False

    def is_right(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"Right({self.value})"

    def __repr__(self) -> str:
        return f"Right({self.value!r})"


# The dataclasses are frozen, so their generated `__init__` has to bypass the blocking `__setattr__`
# through `object.__setattr__`, which is slow. The hand-written `__init__`s above call the slot's
# descriptor directly instead (about a third faster); assignment after construction still raises
# `FrozenInstanceError`.
_set_left_value: Callable[[Left[Any, Any], object], None] = Left.__dict__[
    "value"
].__set__
_set_right_value: Callable[[Right[Any, Any], object], None] = Right.__dict__[
    "value"
].__set__

type Either[L, R] = Left[L, R] | Right[L, R]
