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

## Typing like in Rust

Both type parameters are covariant and the side a variant does not use defaults to `Never`: a bare
`Right(10)` is a `Right[int, Never]` and a bare `Left("e")` is a `Left[str, Never]`. Both are
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
`Right[str, Never]` and `Left("e").map(str)` a `Left[str, str]`.

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

`Either` is a type alias, so use `is_either` (or `isinstance(x, EITHER_TYPES)`) for runtime checks
and `is_left`/`is_right` to narrow an `Either` to one of its variants. Every common method also
exists as a curried module-level function for use with `pymoliath.util.flow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, Self, final, overload

from typing_extensions import Generic, TypeIs, TypeVar

from pymoliath.errors import UnwrapError

# Both parameters are covariant, so `Right[int, Never]` is an `Either[str, int]`. Like in Rust the
# receiver fixes the types a method accepts (`unwrap_or(default: R)`, `bind` returning
# `Either[L, U]`), which puts a covariant parameter in an input position. That is sound here: the
# containers are immutable and those arguments are only ever returned, typed by the receiver's
# (wider) view - the same reasoning as typeshed's `Sequence.index`. Those methods carry a
# `# type: ignore[misc]`.
L = TypeVar("L", covariant=True)
R = TypeVar("R", covariant=True)

# Variant parameters: the side a variant does not use defaults to Never (PEP 696).
L_Never = TypeVar("L_Never", covariant=True, default=Never)
R_Never = TypeVar("R_Never", covariant=True, default=Never)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Y = TypeVar("Y")
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
        arguments, use `map2`/`map3`.

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
        raise NotImplementedError

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

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.match(left=lambda x: "Error", right=lambda x: f"Success: {x}")
        'Success: 10'
        """
        raise NotImplementedError

    def is_left(self) -> bool:
        """Returns True if the Either Monad is Left, otherwise False. Use the module-level
        `is_left` to narrow the type.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_left()
        False
        """
        raise NotImplementedError

    def is_right(self) -> bool:
        """Returns True if the Either Monad is Right, otherwise False. Use the module-level
        `is_right` to narrow the type.

        Examples
        --------
        >>> val: Either[str, int] = Right(10)
        >>> val.is_right()
        True
        """
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True, repr=False)
class Left(_EitherImpl[L_Never, R_Never]):
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

    value: L_Never

    def map(self, function: Callable[[R_Never], U]) -> Left[L_Never, U]:
        return self  # type: ignore[return-value]

    def map_left(self, function: Callable[[L_Never], F]) -> Left[F, R_Never]:
        return Left(function(self.value))

    def bind(
        self, function: Callable[[R_Never], Either[L_Never, U]]
    ) -> Left[L_Never, U]:
        return self  # type: ignore[return-value]

    and_then = bind

    def bind_left(
        self, function: Callable[[L_Never], Either[F, R_Never]]
    ) -> Either[F, R_Never]:
        return function(self.value)

    or_else = bind_left

    def apply(
        self, function: Either[L_Never, Callable[[R_Never], U]]
    ) -> Left[L_Never, U]:
        if isinstance(function, Left):
            return function  # type: ignore[return-value]
        return self  # type: ignore[return-value]

    def is_right_and(self, function: Callable[[R_Never], bool]) -> bool:
        return False

    def is_left_and(self, function: Callable[[L_Never], bool]) -> bool:
        return function(self.value)

    def map_or(self, default_value: U, function: Callable[[R_Never], U]) -> U:
        return default_value

    def map_or_else(
        self, default_function: Callable[[L_Never], U], function: Callable[[R_Never], U]
    ) -> U:
        return default_function(self.value)

    def filter(self, predicate: Callable[[R_Never], bool], left_value: L_Never) -> Self:  # type: ignore[misc]
        return self

    def and_(self, other: Either[L_Never, U]) -> Left[L_Never, U]:
        return self  # type: ignore[return-value]

    def or_(self, other: Either[F, R_Never]) -> Either[F, R_Never]:
        return other

    def zip(self, other: Either[L_Never, U]) -> Left[L_Never, tuple[R_Never, U]]:
        return self  # type: ignore[return-value]

    def swap(self) -> Right[L_Never, R_Never]:
        return Right(self.value)

    def flatten(self: Left[F, Either[F, U]]) -> Left[F, U]:
        return self  # type: ignore[return-value]

    def merge(self: Left[U, U]) -> U:
        return self.value

    def right(self) -> Nothing:
        return Nothing()

    def left(self) -> Just[L_Never]:
        return Just(self.value)

    def unwrap(self) -> Never:
        value = self.value
        message = f"called unwrap on {self!r}"
        if isinstance(value, BaseException):
            raise UnwrapError(self, message) from value
        raise UnwrapError(self, message)

    def unwrap_or(self, default_value: R_Never) -> R_Never:  # type: ignore[misc]
        return default_value

    def unwrap_or_else(self, left_function: Callable[[L_Never], R_Never]) -> R_Never:
        return left_function(self.value)

    def unwrap_left_or(self, default_value: L_Never) -> L_Never:  # type: ignore[misc]
        return self.value

    def inspect(self, function: Callable[[R_Never], None]) -> Self:
        return self

    def inspect_left(self, function: Callable[[L_Never], None]) -> Self:
        function(self.value)
        return self

    def match(
        self, *, left: Callable[[L_Never], U], right: Callable[[R_Never], U]
    ) -> U:
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
@dataclass(frozen=True, slots=True, repr=False)
class Right(_EitherImpl[L_Never, R_Never], Generic[R_Never, L_Never]):
    """The Right variant of the Either Monad, wrapping the successful value.

    Its type parameters are `Right[R, L]` (the Right type first), so a bare `Right(10)` is a
    `Right[int, Never]`.

    Equality and hashing compare the wrapped value (an unhashable value makes the Right
    unhashable).

    Examples
    --------
    >>> Right(10)
    Right(10)
    >>> str(Right("text"))
    'Right(text)'
    """

    value: R_Never

    def map(self, function: Callable[[R_Never], U]) -> Right[U, L_Never]:
        return Right(function(self.value))

    def map_left(self, function: Callable[[L_Never], F]) -> Right[R_Never, F]:
        return self  # type: ignore[return-value]

    def bind(
        self, function: Callable[[R_Never], Either[L_Never, U]]
    ) -> Either[L_Never, U]:
        return function(self.value)

    and_then = bind

    def bind_left(
        self, function: Callable[[L_Never], Either[F, R_Never]]
    ) -> Right[R_Never, F]:
        return self  # type: ignore[return-value]

    or_else = bind_left

    def apply(
        self, function: Either[L_Never, Callable[[R_Never], U]]
    ) -> Either[L_Never, U]:
        if isinstance(function, Right):
            return Right(function.value(self.value))
        return function  # type: ignore[return-value]

    def is_right_and(self, function: Callable[[R_Never], bool]) -> bool:
        return function(self.value)

    def is_left_and(self, function: Callable[[L_Never], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[R_Never], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[L_Never], U], function: Callable[[R_Never], U]
    ) -> U:
        return function(self.value)

    def filter(
        self,
        predicate: Callable[[R_Never], bool],
        left_value: L_Never,  # type: ignore[misc]
    ) -> Either[L_Never, R_Never]:
        return self if predicate(self.value) else Left(left_value)

    def and_(self, other: Either[L_Never, U]) -> Either[L_Never, U]:
        return other

    def or_(self, other: Either[F, R_Never]) -> Right[R_Never, F]:
        return self  # type: ignore[return-value]

    def zip(self, other: Either[L_Never, U]) -> Either[L_Never, tuple[R_Never, U]]:
        if isinstance(other, Right):
            return Right((self.value, other.value))
        return other  # type: ignore[return-value]

    def swap(self) -> Left[R_Never, L_Never]:
        return Left(self.value)

    def flatten(self: Right[Either[F, U], F]) -> Either[F, U]:
        return self.value

    def merge(self: Right[U, U]) -> U:
        return self.value

    def right(self) -> Just[R_Never]:
        return Just(self.value)

    def left(self) -> Nothing:
        return Nothing()

    def unwrap(self) -> R_Never:
        return self.value

    def unwrap_or(self, default_value: R_Never) -> R_Never:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, left_function: Callable[[L_Never], R_Never]) -> R_Never:
        return self.value

    def unwrap_left_or(self, default_value: L_Never) -> L_Never:  # type: ignore[misc]
        return default_value

    def inspect(self, function: Callable[[R_Never], None]) -> Self:
        function(self.value)
        return self

    def inspect_left(self, function: Callable[[L_Never], None]) -> Self:
        return self

    def match(
        self, *, left: Callable[[L_Never], U], right: Callable[[R_Never], U]
    ) -> U:
        return right(self.value)

    def is_left(self) -> bool:
        return False

    def is_right(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"Right({self.value})"

    def __repr__(self) -> str:
        return f"Right({self.value!r})"


type Either[LeftT, RightT] = Left[LeftT, RightT] | Right[RightT, LeftT]

EITHER_TYPES: tuple[type[Left[Any, Any]], type[Right[Any, Any]]] = (Left, Right)
"""The runtime classes of `Either`, for `isinstance` checks (`Either` itself is a type alias)."""


def is_either(value: object) -> TypeIs[Either[Any, Any]]:
    """Returns True if `value` is a Left or a Right.

    Examples
    --------
    >>> is_either(Right(1)), is_either(1)
    (True, False)
    """
    return isinstance(value, EITHER_TYPES)


def is_left(either: Either[F, U]) -> TypeIs[Left[F, U]]:
    """Returns True if the Either Monad is Left, narrowing it to `Left` for type checkers.

    Examples
    --------
    >>> val: Either[str, int] = Left("e")
    >>> if is_left(val):
    ...     print(val.value)
    e
    """
    return isinstance(either, Left)


def is_right(either: Either[F, U]) -> TypeIs[Right[U, F]]:
    """Returns True if the Either Monad is Right, narrowing it to `Right` for type checkers.

    Examples
    --------
    >>> val: Either[str, int] = Right(1)
    >>> if is_right(val):
    ...     print(val.value)
    1
    """
    return isinstance(either, Right)


def map2(
    first: Either[F, U], second: Either[F, V], function: Callable[[U, V], W]
) -> Either[F, W]:
    """Applies a two-argument function to the values of two Either Monads if both are Right.

    If both are Left, the first Left takes precedence.

    Examples
    --------
    >>> map2(Right(1), Right(2), lambda a, b: a + b)
    Right(3)
    >>> map2(Left("first"), Left("second"), lambda a, b: a + b)
    Left('first')
    """
    if isinstance(first, Left):
        return first  # type: ignore[return-value]
    if isinstance(second, Left):
        return second  # type: ignore[return-value]
    return Right(function(first.value, second.value))


def map3(
    first: Either[F, U],
    second: Either[F, V],
    third: Either[F, W],
    function: Callable[[U, V, W], Y],
) -> Either[F, Y]:
    """Applies a three-argument function to the values of three Either Monads if all are Right.

    If several are Left, the first Left takes precedence.

    Examples
    --------
    >>> map3(Right(1), Right(2), Right(3), lambda a, b, c: a + b + c)
    Right(6)
    """
    if isinstance(first, Left):
        return first  # type: ignore[return-value]
    if isinstance(second, Left):
        return second  # type: ignore[return-value]
    if isinstance(third, Left):
        return third  # type: ignore[return-value]
    return Right(function(first.value, second.value, third.value))


@overload
def either_safe(function: Callable[[], U]) -> Either[Exception, U]: ...


@overload
def either_safe(
    function: Callable[[], U], *, exceptions: tuple[type[X], ...]
) -> Either[X, U]: ...


def either_safe(
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
    >>> either_safe(lambda: 1)
    Right(1)
    >>> either_safe(lambda: 1 / 0)
    Left(ZeroDivisionError('division by zero'))
    >>> either_safe(lambda: int("x"), exceptions=(ValueError,)).is_left()
    True
    """
    try:
        return Right(function())
    except exceptions as e:
        return Left(e)


# Curried module-level functions, for point-free pipelines (see `pymoliath.util.flow`).


def map(function: Callable[[U], V]) -> Callable[[Either[F, U]], Either[F, V]]:
    """Curried `Either.map`.

    Examples
    --------
    >>> map(lambda x: x + 1)(Right(1))
    Right(2)
    """
    return lambda either: either.map(function)


def map_left(function: Callable[[F], V]) -> Callable[[Either[F, U]], Either[V, U]]:
    """Curried `Either.map_left`.

    Examples
    --------
    >>> map_left(str.upper)(Left("e"))
    Left('E')
    """
    return lambda either: either.map_left(function)


def bind(
    function: Callable[[U], Either[F, V]],
) -> Callable[[Either[F, U]], Either[F, V]]:
    """Curried `Either.bind`.

    Examples
    --------
    >>> bind(lambda x: Right(x + 1))(Right(1))
    Right(2)
    """
    return lambda either: either.bind(function)


def bind_left(
    function: Callable[[F], Either[V, U]],
) -> Callable[[Either[F, U]], Either[V, U]]:
    """Curried `Either.bind_left`.

    Examples
    --------
    >>> bind_left(lambda e: Right(len(e)))(Left("abc"))
    Right(3)
    """
    return lambda either: either.bind_left(function)


def unwrap_or(default_value: U) -> Callable[[Either[Any, U]], U]:
    """Curried `Either.unwrap_or`.

    Examples
    --------
    >>> unwrap_or(0)(Left("e"))
    0
    """
    return lambda either: either.unwrap_or(default_value)


def unwrap_or_else(function: Callable[[F], U]) -> Callable[[Either[F, U]], U]:
    """Curried `Either.unwrap_or_else`.

    Examples
    --------
    >>> unwrap_or_else(len)(Left("abc"))
    3
    """
    return lambda either: either.unwrap_or_else(function)


def inspect(function: Callable[[U], None]) -> Callable[[Either[F, U]], Either[F, U]]:
    """Curried `Either.inspect`.

    Examples
    --------
    >>> inspect(print)(Right(1))
    1
    Right(1)
    """
    return lambda either: either.inspect(function)


def inspect_left(
    function: Callable[[F], None],
) -> Callable[[Either[F, U]], Either[F, U]]:
    """Curried `Either.inspect_left`.

    Examples
    --------
    >>> inspect_left(print)(Left("e"))
    e
    Left('e')
    """
    return lambda either: either.inspect_left(function)


# Imported last: maybe.py imports Left/Right from this module, so the cycle resolves once at
# import time instead of on every call.
from pymoliath.maybe import Just, Maybe, Nothing  # noqa: E402
