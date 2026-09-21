"""
# Either Monad

The Either Monad is a container used to represent computations that can result in one of two values: a success value or a failure value.
It encapsulates values that could be `Left` (typically representing an error) or `Right` (representing the successful result),
allowing for a functional approach to error handling by chaining operations without constant explicit try-except or error checks.

* Haskell: [Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data.Either.html)
* Rust: [Result](https://doc.rust-lang.org/std/result/)

This implementation is heavily inspired by the Haskell `Either` type and the Rust `Result` type.

The `Either` type is a sum type that can be either `Left` or `Right`.
In this implementation, it is represented as a Union type in Python.

```python
Either = Left[TypeLeft] | Right[TypeRight]
```

## Practical Examples and Benefits:

The Either Monad is particularly useful in scenarios where a function might fail and return an error (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without checking `if error` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Left`, the subsequent operations are skipped automatically, and the final result will be `Left`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly, making the code more robust against unhandled exceptions and making the flow of data more transparent.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if" logic into a linear pipeline of transformations.

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
    .map(get_profile)
    .map(get_permission)
    .unwrap_or_else(lambda error: f"Error: {error}"))

Structural pattern matching provides a clean, declarative way to handle the contents of an `Either` monad. Because the `Left` and `Right` classes are designed to be compatible with Python's `match` statement, you can easily branch your logic based on whether the operation succeeded or failed without manually checking for specific error codes or using complex `if-is_left()` logic.

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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.maybe import Maybe

TypeLeft = TypeVar("TypeLeft")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Right(Generic[TypeRight]):
    """
    The Right side of the Either Monad.

    The Either Monad represents values with two possibilities: either Left[A] or Right[B].
    The Right instance represents the successful path of the computation.
    """

    __slots__ = ("_right_value",)
    __match_args__ = ("_right_value",)

    def __init__(self, value: TypeRight) -> None:
        """
        Initialize the Right Monad.

        Parameters
        ----------
        value: TypeRight
            The value to be stored in the Right container.

        Examples
        -------
        >>> val = Right(10)
        >>> print(val)
        Right(10)
        """
        self._right_value = value

    def map(
        self, function: Callable[[TypeRight], TypeResult]
    ) -> Either[Any, TypeResult]:
        """
        Apply a function to the wrapped value.

        If the value is Right, the function is applied to the internal value.
        If the value is Left, the Left container is returned unchanged.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            A function that accepts a value of type TypeRight and returns a TypeResult.

        Returns
        -------
        Either[TypeLeft, TypeResult]
            A new Either Monad containing the result of the function application.

        Examples
        -------
        >>> val = Right(10)
        >>> val.map(lambda x: x + 5)
        Right(15)
        """
        return Right(function(self._right_value))

    def map_left(
        self, function: Callable[[Any], TypeResult]
    ) -> Either[TypeResult, TypeRight]:
        """
        Apply a function to the Left value.

        Since this instance is Right, the operation is ignored and the
        Right instance is returned unchanged.

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            A function that accepts a TypeLeft value and returns a TypeResult.

        Returns
        -------
        Either[TypeResult, TypeRight]
            The current Right instance.

        Examples
        -------
        >>> val = Right(10)
        >>> val.map_left(lambda x: x + 1)
        Right(10)
        """
        return self

    def bind(
        self, function: Callable[[TypeRight], Either[Any, TypeResult]]
    ) -> Either[Any, TypeResult]:
        """
        Chain a computation that produces a new Either Monad.

        If this is Right, the provided function is executed with the internal
        value, and the resulting Either Monad is returned.

        Parameters
        ----------
        function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]
            A function that takes a value of type TypeRight and returns
            an Either Monad.

        Returns
        -------
        Either[TypeLeft, TypeResult]
            The result of the bound function.

        Examples
        -------
        >>> def get_next(x):
        ...     return Right(x + 1)
        >>> val = Right(10)
        >>> val.bind(get_next)
        Right(11)
        """
        return function(self._right_value)

    def bind_left(
        self, function: Callable[[Any], Either[TypeResult, TypeRight]]
    ) -> Either[TypeResult, TypeRight]:
        """
        Chain a computation that produces a new Either Monad on the Left side.

        Since this is Right, the operation is ignored and the
        Right instance is returned unchanged.

        Parameters
        ----------
        function: Callable[[Any], Either[TypeResult, TypeRight]]
            A function that takes a value of type TypeLeft and returns
            an Either Monad.

        Returns
        -------
        Either[TypeResult, TypeRight]
            The current Right instance.

        Examples
        -------
        >>> val = Right(10)
        >>> val.bind_left(lambda x: Left("Error"))
        Right(10)
        """
        return self

    def apply(
        self, applicative: Either[Any, Callable[..., TypeResult]]
    ) -> Either[Any, TypeResult]:
        """
        Apply an applicative functor containing a function to the Right value.

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[..., TypeResult]]
            An Either Monad containing a function to be applied.

        Returns
        -------
        Either[Any, TypeResult]
            The result of applying the function contained in the applicative.

        Examples
        -------
        >>> val = Right(10)
        >>> func_monad = Right(lambda x: x * 2)
        >>> val.apply(func_monad)
        Right(20)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Either[Any, TypeResult]:
            """Maps the applicative's function, curried, over this Right value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Right[Callable[..., TypeResult]],
        applicative_value: Either[Any, Any],
    ) -> Either[Any, TypeResult]:
        """
        Apply a function stored in this Right to a value inside another Either Monad.

        Parameters
        ----------
        applicative_value: Either[TypeLeft, TypePure]
            The Either Monad containing the value to be processed.

        Returns
        -------
        Either[Any, TypeResult]
            The result of applying the function to the inner value.

        Examples
        -------
        >>> func = lambda x: x + 1
        >>> val = Right(func)
        >>> other_val = Right(10)
        >>> val.apply2(other_val)
        Right(11)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Either[Any, TypeResult]:
            """Maps the applicative's function, curried, over the applicative_value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def is_right_and(self, function: Callable[[TypeRight], bool]) -> bool:
        """
        Check if the monad is Right and satisfies a predicate.

        Parameters
        ----------
        function: Callable[[TypeRight], bool]
            A predicate function applied to the contained Right value.

        Returns
        -------
        bool
            True if the monad is Right AND the predicate returns True.

        Examples
        -------
        >>> val = Right(10)
        >>> val.is_right_and(lambda x: x > 5)
        True
        """
        return function(self._right_value)

    def is_left_and(self, function: Callable[[Any], bool]) -> bool:
        """
        Check if the monad is Left and satisfies a predicate.

        Since this is a Right instance, this always returns False.

        Parameters
        ----------
        function: Callable[[TypeLeft], bool]
            A predicate function for the Left value.

        Returns
        -------
        bool
            Always False.

        Examples
        -------
        >>> val = Right(10)
        >>> val.is_left_and(lambda x: x < 0)
        False
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeRight], TypeResult]
    ) -> TypeResult:
        """
        Apply a function to the Right value, or return a default if Left.

        Because this is a Right instance, the function is applied to the
        internal value.

        Parameters
        ----------
        default_value: TypeResult
            The value to return if the monad were Left.
        function: Callable[[TypeRight], TypeResult]
            The function to apply to the Right value.

        Returns
        -------
        TypeResult
            The result of applying the function.

        Examples
        -------
        >>> val = Right(10)
        >>> val.map_or(0, lambda x: x + 5)
        15
        """
        return function(self._right_value)

    def and_(self, other: Either[Any, TypeResult]) -> Either[Any, TypeResult]:
        """
        Chain with another Either Monad.

        Returns the other Monad if this is Right.

        Parameters
        ----------
        other: Either[TypeLeft, TypeResult]
            The other Either Monad to return.

        Returns
        -------
        Either[Any, TypeResult]
            The 'other' Monad.

        Examples
        -------
        >>> val1 = Right(1)
        >>> val2 = Right(2)
        >>> val1.and_(val2)
        Right(2)
        """
        return other

    def or_(self, other: Either[Any, TypeRight]) -> Either[Any, TypeRight]:
        """
        Fallback to another Either Monad.

        Returns this instance if it is Right.

        Parameters
        ----------
        other: Either[TypeLeft, TypeRight]
            The fallback Either Monad.

        Returns
        -------
        Either[Any, TypeRight]
            The current Right instance.

        Examples
        -------
        >>> val1 = Right(1)
        >>> val2 = Right(2)
        >>> val1.or_(val2)
        Right(1)
        """
        return self

    def zip(
        self, other: Either[Any, TypePure]
    ) -> Either[Any, Tuple[TypeRight, TypePure]]:
        """
        Combine this Right value with another Either Monad into a tuple.

        Parameters
        ----------
        other: Either[TypeLeft, TypePure]
            The other Either Monad to zip with.

        Returns
        -------
        Either[Any, Tuple[TypeRight, TypePure]]
            A Right containing a tuple of both values.

        Examples
        -------
        >>> val1 = Right(1)
        >>> val2 = Right(2)
        >>> val1.zip(val2)
        Right((1, 2))
        """
        return other.map(lambda o: (self._right_value, o))

    @overload
    def flatten(self: Right[Right[TypeResult]]) -> Either[Any, TypeResult]: ...

    @overload
    def flatten(self: Right[Left[TypeLeft]]) -> Either[TypeLeft, Any]: ...

    def flatten(self) -> Either[Any, Any]:
        """
        Flatten a nested Either Monad by removing one layer of wrapping.

        Returns
        -------
        Either[Any, Any]
            The inner value of the nested Monad.

        Examples
        -------
        >>> val = Right(Right(10))
        >>> val.flatten()
        Right(10)
        """
        return cast(Either[Any, Any], self._right_value)

    def right(self) -> Maybe[TypeRight]:
        """
        Convert the Right value to a Maybe Monad.

        Returns
        -------
        Maybe[TypeRight]
            A Just containing the Right value.

        Examples
        -------
        >>> val = Right(10)
        >>> val.right()
        Just(10)
        """
        from pymoliath.maybe import Just

        return Just(self._right_value)

    def left(self) -> Maybe[Any]:
        """
        Convert the Left value to a Maybe Monad.

        Since this is Right, it returns Nothing.

        Returns
        -------
        Maybe[Any]
            Nothing.

        Examples
        -------
        >>> val = Right(10)
        >>> val.left()
        Nothing()
        """
        from pymoliath.maybe import Nothing

        return Nothing()

    def unwrap(self) -> TypeRight:
        """
        Extract the value from the Right Monad.

        Returns
        -------
        TypeRight
            The wrapped value.

        Examples
        -------
        >>> val = Right(10)
        >>> val.unwrap()
        10
        """
        return self._right_value

    def unwrap_or(self, default_value: TypeRight) -> TypeRight:
        """
        Extract the value or return a default if Left.

        Returns
        -------
        TypeRight
            The internal Right value.

        Examples
        -------
        >>> val = Right(10)
        >>> val.unwrap_or(5)
        10
        """
        return self._right_value

    def unwrap_or_else(self, left_function: Callable[[Any], TypeRight]) -> TypeRight:
        """
        Extract the value or apply a function if Left.

        Returns
        -------
        TypeRight
            The internal Right value.

        Examples
        -------
        >>> val = Right(10)
        >>> val.unwrap_or_else(lambda _: 0)
        10
        """
        return self._right_value

    def unwrap_left_or(self, default_value: TypeLeft) -> TypeLeft:
        """
        Extract the Left value or return a default.

        Since this is Right, the default value is returned.

        Parameters
        ----------
        default_value: TypeLeft
            The default value of type TypeLeft.

        Returns
        -------
        TypeLeft
            The default value.

        Examples
        -------
        >>> val = Right(10)
        >>> val.unwrap_left_or("Default")
        'Default'
        """
        return default_value

    def inspect(self, function: Callable[[TypeRight], None]) -> Either[Any, TypeRight]:
        """
        Perform a side-effecting action on the Right value.

        Parameters
        ----------
        function: Callable[[TypeRight], None]
            The inspection function to run.

        Returns
        -------
        Either[Any, TypeRight]
            The current Right instance.

        Examples
        -------
        >>> val = Right(10)
        >>> val.inspect(lambda x: print(f"Value: {x}"))
        Value: 10
        Right(10)
        """
        function(self._right_value)
        return self

    def inspect_left(self, function: Callable[[Any], None]) -> Either[Any, TypeRight]:
        """
        Perform a side-effecting action on the Left value.

        Since this is Right, the function is not executed.

        Parameters
        ----------
        function: Callable[[Any], None]
            The inspection function for the Left value.

        Returns
        -------
        Either[Any, TypeRight]
            The current Right instance.

        Examples
        -------
        >>> val = Right(10)
        >>> val.inspect_left(lambda x: print(f"Left: {x}"))
        Right(10)
        """
        return self

    def match(
        self,
        left_function: Callable[[Any], TypeResult],
        right_function: Callable[[TypeRight], TypeResult],
    ) -> TypeResult:
        """
        Handle the value using Railroad Oriented Programming.

        Parameters
        ----------
        left_function: Callable[[Any], TypeResult]
            Callback for the Left case.
        right_function: Callable[[TypeRight], TypeResult]
            Callback for the Right case.

        Returns
        -------
        TypeResult
            The result of the right_function.

        Examples
        -------
        >>> val = Right(10)
        >>> val.match(lambda x: "Error", lambda x: f"Success: {x}")
        'Success: 10'
        """
        return right_function(self._right_value)

    def is_left(self) -> bool:
        """
        Check if the monad is a Left type.

        Returns
        -------
        bool
            False.

        Examples
        -------
        >>> val = Right(10)
        >>> val.is_left()
        False
        """
        return False

    def is_right(self) -> bool:
        """
        Check if the monad is a Right type.

        Returns
        -------
        bool
            True.

        Examples
        -------
        >>> val = Right(10)
        >>> val.is_right()
        True
        """
        return True

    def __str__(self) -> str:
        """
        Returns the string representation of the Right Monad.

        Returns
        -------
        str
            The string representation.

        Examples
        -------
        >>> val = Right(10)
        >>> str(val)
        'Right(10)'
        """
        return f"Right({self._right_value})"

    def __eq__(self, __o: object) -> bool:
        """
        Compare this Right Monad with another object.

        Returns
        -------
        bool
            True if the other object is a Right Monad and has the same string representation.

        Examples
        -------
        >>> val1 = Right(10)
        >>> val2 = Right(10)
        >>> val1 == val2
        True
        """
        return isinstance(__o, Right) and str(self) == str(__o)

    def __repr__(self) -> str:
        """
        Returns the official string representation of the Right Monad.

        Returns
        -------
        str
            The official string representation.

        Examples
        -------
        >>> val = Right(10)
        >>> repr(val)
        'Right(10)'
        """
        return str(self)


class Left(Generic[TypeLeft]):
    """
    The Left side of the Either Monad.

    The Either Monad represents values with two possibilities: either Left[A] or Right[B].
    The Left instance represents the error or alternative path of the computation.
    """

    __slots__ = ("_left_value",)
    __match_args__ = ("_left_value",)

    def __init__(self, value: TypeLeft):
        """
        Initialize the Left Monad.

        Parameters
        ----------
        value: TypeLeft
            The value to be stored in the Left container.

        Examples
        -------
        Create a Left instance with an error message.
        >>> val = Left("Error Message")
        >>> print(val)
        Left(Error Message)
        """
        self._left_value = value

    def map(
        self, function: Callable[[Any], TypeResult]
    ) -> Either[TypeLeft, TypeResult]:
        """
        Apply a function to the Right value.

        Since this is Left, the function is ignored and the
        Left instance is returned unchanged.

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            Function to be applied to the Right value.

        Returns
        -------
        Either[TypeLeft, TypeResult]
            The current Left instance.

        Examples
        -------
        Attempting to map over a Left instance returns the Left instance unchanged.
        >>> val = Left("Error")
        >>> val.map(lambda x: x + 1)
        Left(Error)
        """
        return self

    def map_left(
        self, function: Callable[[TypeLeft], TypeResult]
    ) -> Either[TypeResult, Any]:
        """
        Apply a function to the wrapped Left value.

        Parameters
        ----------
        function: Callable[[TypeLeft], TypeResult]
            Function which takes a value of TypeLeft and returns a TypeResult.

        Returns
        -------
        Either[TypeResult, Any]
            A new Left instance containing the result of the function.

        Examples
        -------
        Transform the inner value of a Left instance.
        >>> val = Left("Error")
        >>> val.map_left(lambda x: x.upper())
        Left(ERROR)
        """
        return Left(function(self._left_value))

    def bind(
        self, function: Callable[[Any], Either[TypeLeft, TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        """
        Chain a computation that produces a new Either Monad.

        Since this is Left, the operation is ignored and the
        Left instance is returned unchanged.

        Parameters
        ----------
        function: Callable[[Any], Either[TypeLeft, TypeResult]]
            Function which would be called with the Right value.

        Returns
        -------
        Either[TypeLeft, TypeResult]
            The current Left instance.

        Examples
        -------
        Binding a function to a Left instance returns the original Left instance.
        >>> val = Left("Error")
        >>> val.bind(lambda _: Left("New Error"))
        Left(Error)
        """
        return self

    def bind_left(
        self, function: Callable[[TypeLeft], Either[TypeResult, Any]]
    ) -> Either[TypeResult, Any]:
        """
        Chain a computation that produces a new Either Monad on the Left side.

        Parameters
        ----------
        function: Callable[[TypeLeft], Either[TypeResult, Any]]
            Function which takes a value of TypeLeft and returns a new Either Monad.

        Returns
        -------
        Either[TypeResult, Any]
            The result of the function call.

        Examples
        -------
        Chain a computation specifically on the Left side of the Monad.
        >>> val = Left("Error")
        >>> val.bind_left(lambda err: Left(f"{err} bind"))
        Left(Error bind)
        """
        return function(self._left_value)

    def apply(
        self, applicative: Either[TypeLeft, Callable[..., TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        """
        Apply an applicative functor for the Right case.

        Since this is Left, the operation is ignored and the
        Left instance is returned unchanged.

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[..., TypeResult]]
            Applicative Either Monad containing a function.

        Returns
        -------
        Either[TypeLeft, TypeResult]
            The current Left instance.

        Examples
        -------
        Applying an applicative to a Left instance returns the Left instance.
        >>> val = Left("Error")
        >>> val.apply(Right(lambda x: x))
        Left(Error)
        """
        return self

    def apply2(self, applicative_value: Either[TypeLeft, Any]) -> Either[TypeLeft, Any]:
        """
        Apply a value from another Either Monad when this is Right.

        Since this is Left, the operation is ignored and the
        Left instance is returned unchanged.

        Parameters
        ----------
        applicative_value: Either[TypeLeft, Any]
            Either Monad which contains a value.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance.

        Examples
        -------
        Applying a value from another Monad to a Left instance returns the Left instance.
        >>> val = Left("Error")
        >>> val.apply2(Right(10))
        Left(Error)
        """
        return self

    def is_right_and(self, function: Callable[[Any], bool]) -> bool:
        """
        Check if the monad is Right and satisfies a predicate.

        Since this is Left, this always returns False.

        Parameters
        ----------
        function: Callable[[Any], bool]
            Predicate function for the Right value.

        Returns
        -------
        bool
            False.

        Examples
        -------
        Check if a Left instance satisfies a Right-specific predicate.
        >>> val = Left("Error")
        >>> val.is_right_and(lambda x: x == "Error")
        False
        """
        return False

    def is_left_and(self, function: Callable[[TypeLeft], bool]) -> bool:
        """
        Check if the monad is Left and satisfies a predicate.

        Parameters
        ----------
        function: Callable[[TypeLeft], bool]
            Predicate function applied to the Left value.

        Returns
        -------
        bool
            The result of the predicate.

        Examples
        -------
        Check if the Left instance satisfies the provided predicate.
        >>> val = Left("Error")
        >>> val.is_left_and(lambda x: x == "Error")
        True
        """
        return function(self._left_value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """
        Apply a function to the Right value, or return a default if Left.

        Since this is Left, the default value is returned.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned.
        function: Callable[[Any], TypeResult]
            Function applied to the Right value.

        Returns
        -------
        TypeResult
            The default value.

        Examples
        -------
        Return a default value because the Monad is in a Left state.
        >>> val = Left("Error")
        >>> val.map_or("Fallback", lambda x: x)
        'Fallback'
        """
        return default_value

    def and_(self, other: Either[TypeLeft, Any]) -> Either[TypeLeft, Any]:
        """
        Chain with another Either Monad.

        Since this is Left, this returns the current Left instance.

        Parameters
        ----------
        other: Either[TypeLeft, Any]
            The other Either Monad.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance.

        Examples
        -------
        Chaining two Left instances returns the first Left instance.
        >>> val1 = Left("Error")
        >>> val2 = Left("Other Error")
        >>> val1.and_(val2)
        Left(Error)
        """
        return self

    def or_(self, other: Either[Any, TypeRight]) -> Either[Any, TypeRight]:
        """
        Fallback to another Either Monad.

        Since this is Left, the other Monad is returned.

        Parameters
        ----------
        other: Either[Any, TypeRight]
            The fallback Either Monad.

        Returns
        -------
        Either[Any, TypeRight]
            The other Monad.

        Examples
        -------
        Provide a fallback value when the current Monad is Left.
        >>> val1 = Left("Error")
        >>> val2 = Right(10)
        >>> val1.or_(val2)
        Right(10)
        """
        return other

    def zip(self, other: Either[Any, Any]) -> Either[TypeLeft, Any]:
        """
        Combine this Left with another Either Monad.

        Since this is Left, the current Left instance is returned.

        Parameters
        ----------
        other: Either[Any, Any]
            The other Either Monad to zip with.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance.

        Examples
        -------
        Zipping a Left instance with another monad returns the Left instance.
        >>> val1 = Left("Error")
        >>> val2 = Right(10)
        >>> val1.zip(val2)
        Left(Error)
        """
        return self

    def flatten(self) -> Either[TypeLeft, Any]:
        """
        Flatten a nested Either Monad.

        Since this is Left, the Left instance is returned unchanged - flatten only ever unwraps
        a nested Either carried on the Right side.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance.

        Examples
        -------
        Flattening a Left instance leaves it unchanged, whatever it wraps.
        >>> val = Left("Error")
        >>> val.flatten()
        Left(Error)
        """
        return self

    def right(self) -> Maybe[Any]:
        """
        Convert the Right value to a Maybe Monad.

        Since this is Left, it returns Nothing.

        Returns
        -------
        Maybe[Any]
            Nothing.

        Examples
        -------
        Convert a Left value to a Maybe type, resulting in Nothing.
        >>> val = Left("Error")
        >>> val.right()
        Nothing()
        """
        from pymoliath.maybe import Nothing

        return Nothing()

    def left(self) -> Maybe[TypeLeft]:
        """
        Convert the Left value to a Maybe Monad.

        Returns
        -------
        Maybe[TypeLeft]
            A Just containing the Left value.

        Examples
        -------
        Convert a Left value to a Maybe type, resulting in a Just value.
        >>> val = Left("Error")
        >>> val.left()
        Just(Error)
        """
        from pymoliath.maybe import Just

        return Just(self._left_value)

    def unwrap(self) -> Any:
        """
        Extract the value from the Left Monad.

        Raises
        -------
        Exception
            An Exception containing the Left value.

        Examples
        -------
        Extract the value from a Left instance, which raises an Exception.
        >>> val = Left("Error")
        >>> val.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Error
        """
        raise Exception(self._left_value)

    def unwrap_or(self, default_value: TypeRight) -> TypeRight:
        """
        Extract the value or return a default if Left.

        Since this is Left, the default value is returned.

        Parameters
        ----------
        default_value: TypeRight
            Default value of type TypeRight.

        Returns
        -------
        TypeRight
            The default value.

        Examples
        -------
        Extract the default value because the Monad is in a Left state.
        >>> val = Left("Error")
        >>> val.unwrap_or(10)
        10
        """
        return default_value

    def unwrap_or_else(
        self, left_function: Callable[[TypeLeft], TypeResult]
    ) -> TypeResult:
        """
        Extract the value or apply a function if Left.

        Since this is Left, the left_function is called with the Left value.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeResult]
            Function called with the Left value.

        Returns
        -------
        TypeResult
            The result of the function call.

        Examples
        -------
        Execute a fallback function because the Monad is in a Left state.
        >>> val = Left("Error")
        >>> val.unwrap_or_else(lambda x: "Handled Error")
        'Handled Error'
        """
        return left_function(self._left_value)

    def unwrap_left_or(self, default_value: TypeLeft) -> TypeLeft:
        """
        Extract the Left value or return a default.

        Since this is Left, the internal Left value is returned.

        Parameters
        ----------
        default_value: TypeLeft
            Default value of type TypeLeft.

        Returns
        -------
        TypeLeft
            The internal Left value.

        Examples
        -------
        Extract the internal value of the Left instance.
        >>> val = Left("Error")
        >>> val.unwrap_left_or("Default")
        'Error'
        """
        return self._left_value

    def inspect(self, function: Callable[[Any], None]) -> Either[TypeLeft, Any]:
        """
        Perform a side-effecting action on the Right value.

        Since this is Left, the function is ignored.

        Parameters
        ----------
        function: Callable[[Any], None]
            Inspection function for the Right value.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance.

        Examples
        -------
        Performing an inspection on a Left instance returns the Left instance unchanged.
        >>> val = Left("Error")
        >>> val.inspect(lambda x: print(f"Right value: {x}"))
        Left(Error)
        """
        return self

    def inspect_left(
        self, function: Callable[[TypeLeft], None]
    ) -> Either[TypeLeft, Any]:
        """
        Perform a side-effecting action on the Left value.

        Parameters
        ----------
        function: Callable[[TypeLeft], None]
            Inspection function for the Left value.

        Returns
        -------
        Either[TypeLeft, Any]
            The current Left instance after running the inspection.

        Examples
        -------
        Perform a side-effecting action on the Left value.
        >>> val = Left("Error")
        >>> val.inspect_left(lambda x: print(f"Left value: {x}"))
        Left value: Error
        Left(Error)
        """
        function(self._left_value)
        return self

    def match(
        self,
        left_function: Callable[[TypeLeft], TypeResult],
        right_function: Callable[[Any], TypeResult],
    ) -> TypeResult:
        """
        Handle the value using Railroad Oriented Programming for the Left case.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeResult]
            Callback for the Left case.
        right_function: Callable[[Any], TypeResult]
            Callback for the Right case.

        Returns
        -------
        TypeResult
            The result of the left_function.

        Examples
        -------
        Execute the left-specific handler when the Monad is Left.
        >>> val = Left("Error")
        >>> val.match(lambda x: f"Error: {x}", lambda x: "Success")
        'Error: Error'
        """
        return left_function(self._left_value)

    def is_left(self) -> bool:
        """
        Check if the monad is a Left type.

        Returns
        -------
        bool
            True.

        Examples
        -------
        Verify that the Monad is a Left instance.
        >>> val = Left("Error")
        >>> val.is_left()
        True
        """
        return True

    def is_right(self) -> bool:
        """
        Check if the monad is a Right type.

        Returns
        -------
        bool
            False.

        Examples
        -------
        Verify that the Monad is not a Right instance.
        >>> val = Left("Error")
        >>> val.is_right()
        False
        """
        return False

    def __str__(self) -> str:
        """
        Returns the string representation of the Left Monad.
        """
        return f"Left({self._left_value})"

    def __eq__(self, __o: object) -> bool:
        """
        Compare this Left Monad with another object.

        Returns
        -------
        bool
            True if the other object is a Left Monad and has the same string representation.

        Examples
        -------
        Compare two Left instances for equality.
        >>> val1 = Left("Error")
        >>> val2 = Left("Error")
        >>> val1 == val2
        True
        """
        return isinstance(__o, Left) and str(self) == str(__o)

    def __repr__(self) -> str:
        """
        Returns the official string representation of the Left Monad.
        """
        return str(self)


Either: TypeAlias = Left[TypeLeft] | Right[TypeRight]


def either_safe(function: Callable[[], TypeResult]) -> Either[Exception, TypeResult]:
    """
    Execute an unsafe function and wrap the result in an Either Monad.

    Captures any Exception raised by the function and returns it as a Left value,
    otherwise returns the result as a Right value.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        A callable function that may raise an exception.

    Returns
    -------
    Either[Exception, TypeResult]
        An Either Monad containing the result or the caught Exception.
    """
    try:
        return Right(function())
    except Exception as e:
        return Left(e)
