"""
# Try Monad

The Try Monad is a container used to represent computations that either succeed with a value or
fail by raising an exception. It encapsulates the outcome of code that might raise, allowing for a
functional approach to exception handling by chaining operations without wrapping every step in
its own `try`/`except` block. Unlike `Result`, whose `Ok`/`Err` values are constructed explicitly
by the caller, `Try`'s `map`/`bind` automatically catch any `Exception` raised by the function they
are given and turn it into a `Failure`.

* Scala: [Try](https://www.scala-lang.org/api/current/scala/util/Try.html)

This implementation is heavily inspired by Scala's `Try` type.

The `Try` type is a sum type that can be either `Success` or `Failure`.
In this implementation, it is represented as a Union type in Python.

```python
Try = Success[TypeSource] | Failure
```

## Practical Examples and Benefits:

The Try Monad is particularly useful in scenarios where a chain of operations might each raise an
exception (e.g. parsing a string, calling out to a library that isn't exception-free, or performing
several arithmetic operations that could divide by zero), and you want the first exception to short
-circuit the rest of the chain without a `try`/`except` around each step.

### Benefits:
1. Declarative Code: `map`/`bind` chain operations together while automatically catching any
   exception the passed function raises, rather than requiring a `try`/`except` per step.
2. Error Propagation: If any step in a chain of operations produces a `Failure`, the subsequent
   operations are skipped automatically, and the final result stays a `Failure`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly in the
   return type, rather than having exceptions propagate implicitly and invisibly through the call
   stack.

#### Example: Parsing and transforming a value that might raise.

```python
# Without Try (Imperative)
try:
    parsed = int(raw_value)
    doubled = parsed * 2
    print(doubled)
except ValueError as e:
    print(f"Error: {e}")

# With Try (Functional)
(safe(lambda: int(raw_value))
    .map(lambda parsed: parsed * 2)
    .match(lambda e: print(f"Error: {e}"), print))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Try`
monad. Because the `Success` and `Failure` classes are designed to be compatible with Python's
`match` statement, you can easily branch your logic based on whether the computation succeeded or
raised, without manually checking `is_success()`/`is_failure()`.

```python
match try_value:
    case Success(x):
        # This block executes if the computation succeeded
        print(f"Success value: {x}")
    case Failure(e):
        # This block executes if the computation raised
        print(f"Exception encountered: {e}")
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Tuple, TypeAlias, TypeVar, cast, overload

from pymoliath.either import Either, Left, Right
from pymoliath.result import Err, Ok, Result
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Success(Generic[TypeSource]):
    """The Success variant of the Try Monad.

    Represents a computation that completed without raising. It wraps a value of type
    `TypeSource` and provides a functional interface for chaining operations, catching any
    exception the chained function raises and turning it into a `Failure`.
    """

    __slots__ = ("_success_value",)
    __match_args__ = ("_success_value",)

    def __init__(self, value: TypeSource):
        """Success Monad constructor which takes a value of type TypeSource.

        Parameters
        ----------
        value: TypeSource
            Value to be stored in the Success Monad.

        Examples
        --------
        >>> success = Success(42)
        >>> print(success)
        Success(42)
        """
        self._success_value = value

    def map(self, function: Callable[[TypeSource], TypeResult]) -> Try[TypeResult]:
        """Calls function to the a wrapped Success value if not an Failure, otherwise leaving the Failure value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult.

        Returns
        -------
        try: Try[TypeResult]
            Returns an Success with the function result or otherwise Failure.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map(lambda x: x + 1)
        Success(6)
        >>> def risky(x: int) -> int:
        ...     raise ValueError("boom")
        >>> val.map(risky)
        Failure(boom)
        """
        try:
            return Success(function(self._success_value))
        except Exception as e:
            return Failure(e)

    def map_failure(
        self, function: Callable[[Exception], Exception]
    ) -> Try[TypeSource]:
        """Calls function to the Failure value if not Success, otherwise leaving the Success value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Exception]
            Function which takes an Exception and returns an Exception.

        Returns
        -------
        result: Try[TypeSource]
            Returns an Failure with the function result or otherwise Success.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map_failure(lambda e: ValueError("wrapped"))
        Success(5)
        """
        return self

    def bind(
        self, function: Callable[[TypeSource], Try[TypeResult]]
    ) -> Try[TypeResult]:
        """Calls function if Try Monad is Success, otherwise returns Failure.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeSource], Try[TypeResult]]
            Function which takes a value of TypeSource and returns a Try Monad of TypeResult.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the function result if Success, otherwise an Failure.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> def get_next(x: int) -> Try[int]:
        ...     return Success(x + 1) if x < 10 else Failure(ValueError("too big"))
        >>> val.bind(get_next)
        Success(6)
        >>> Success(10).bind(get_next)
        Failure(too big)
        """
        try:
            return function(self._success_value)
        except Exception as e:
            return Failure(e)

    def bind_failure(
        self, function: Callable[[Exception], Try[TypeSource]]
    ) -> Try[TypeSource]:
        """Calls function if Try Monad is an Failure, otherwise returns Success.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Try[TypeResult]]
            Function which takes an Exception and returns a Try Monad of TypeResult.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the function result if Failure, otherwise an Success.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.bind_failure(lambda e: Success(0))
        Success(5)
        """
        return self

    def apply(self, applicative: Try[Callable[..., TypeResult]]) -> Try[TypeResult]:
        """Applies the passed applicative wrapping a function if Try Monad is Success, otherwise returns Failure.

        Parameters
        ----------
        applicative: Try[Callable[Callable[[TypeSource], TypeResult]]
            Applicative Try Monad which contains a function.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the applied function if Success, otherwise an Failure.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val.apply(func)
        Success(20)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Try[TypeResult]:
            """Maps the applicative's function, curried, over this Success value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Success[Callable[..., TypeResult]],
        applicative_value: Try[Any],
    ) -> Try[TypeResult]:
        """Applies the passed Try Monad wrapping a value to the Try Monad containing a function if Success,
        otherwise returns Failure.

        Parameters
        ----------
        applicative_value: Try[TypePure]
            Try monad containing a value which will be applied to the Try Monad containing a function.

        Returns
        -------
        try: Try[TypeResult]:
            Returns a Try Monad from the applied function if Success, otherwise an Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val: Try[int] = Success(10)
        >>> func.apply2(val)
        Success(20)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Try[TypeResult]:
            """Maps the curried applicative function over the applicative's value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def is_success_and(self, function: Callable[[TypeSource], bool]) -> bool:
        """Returns True if the Try Monad is Success and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeSource], bool]
            Predicate function applied to the Success value.

        Returns
        -------
        result: bool
            Returns the predicate result.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.is_success_and(lambda x: x > 5)
        True
        """
        return function(self._success_value)

    def is_failure_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Try Monad is Success.

        Parameters
        ----------
        function: Callable[[Exception], bool]
            Predicate function which would be applied to the Failure value.

        Returns
        -------
        result: bool
            Returns False.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.is_failure_and(lambda e: True)
        False
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeSource], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Success value, or returns the default value if Failure.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Try Monad is Failure.
        function: Callable[[TypeSource], TypeResult]
            Function applied to the Success value.

        Returns
        -------
        result: TypeResult
            Returns the function result.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        return function(self._success_value)

    def and_(self, other: Try[TypeResult]) -> Try[TypeResult]:
        """Returns `other` if the Try Monad is Success, otherwise Failure.

        Parameters
        ----------
        other: Try[TypeResult]
            Try Monad to be returned if this Try Monad is Success.

        Returns
        -------
        result: Try[TypeResult]
            Returns `other`.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.and_(val2)
        Success(2)
        """
        return other

    def or_(self, other: Try[TypeSource]) -> Try[TypeSource]:
        """Returns this Try Monad if it is Success, otherwise `other`.

        Parameters
        ----------
        other: Try[TypeSource]
            Try Monad to be returned if this Try Monad is Failure.

        Returns
        -------
        result: Try[TypeSource]
            Returns this Success.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.or_(val2)
        Success(1)
        """
        return self

    def zip(self, other: Try[TypePure]) -> Try[Tuple[TypeSource, TypePure]]:
        """Combines this Try Monad with another into a Try Monad of a tuple, or Failure if either is Failure.

        Parameters
        ----------
        other: Try[TypePure]
            Try Monad to be zipped with this Try Monad.

        Returns
        -------
        result: Try[Tuple[TypeSource, TypePure]]
            Returns Success of a tuple of both values, or Failure.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.zip(val2)
        Success((1, 2))
        """
        return other.map(lambda o: (self._success_value, o))

    @overload
    def flatten(self: Success[Success[TypeResult]]) -> Try[TypeResult]: ...

    @overload
    def flatten(self: Success[Failure]) -> Try[Any]: ...

    def flatten(self) -> Try[Any]:
        """Flattens a nested Try Monad by one level.

        Returns
        -------
        result: Try[TypeResult]
            Returns the nested Try Monad.

        Examples
        --------
        >>> val: Try[Try[int]] = Success(Success(1))
        >>> val.flatten()
        Success(1)
        """
        return cast(Try[Any], self._success_value)

    def unwrap(self) -> TypeSource:
        """Returns the Success value if not Failure, or otherwise raises the Failure Exception.

        Returns
        -------
        result: TypeRight
            Returns the Success value or raises the Failure Exception.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap()
        1
        """
        return self._success_value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the Ok value if not Err, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeOk
            Default value of TypeOk

        Returns
        -------
        result: TypeOk
            Returns the Ok value or a default value.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap_or(0)
        1
        """
        return self._success_value

    def unwrap_or_else(
        self, failure_function: Callable[[Exception], TypeSource]
    ) -> TypeSource:
        """Returns the Success value if not Failure, or otherwise calls the provided function with the failure value.

        Parameters
        ----------
        failure_function: Callable[[Exception], TypeRight]
            Called with the failure value and must return a value of type TypeSource

        Returns
        -------
        result: TypeRight
            Returns the Success value or value from the function call.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap_or_else(lambda e: 0)
        1
        """
        return self._success_value

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        """Returns the Err value if not Ok, or otherwise a provided default Exception.

        Parameters
        ----------
        default_value: Exception
            Default value of type Exception

        Returns
        -------
        result: Exception
            Returns the Err value or a default value.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap_failure_or(ValueError("default"))
        ValueError('default')
        """
        return default_value

    def inspect(self, function: Callable[[TypeSource], None]) -> Try[TypeSource]:
        """Inspect the Try monad value of TypeSource

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            Inspection function which takes the success value of the Try monad

        Returns
        -------
        try: Try[TypeSource]

        Examples
        --------
        >>> val: Try[int] = Success(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Success(42)
        """
        function(self._success_value)
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Try[TypeSource]:
        """Inspect the Try monad Exception value

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception value of the Try monad

        Returns
        -------
        try: Try[TypeSource]

        Examples
        --------
        >>> val: Try[int] = Success(42)
        >>> val.inspect_failure(lambda e: print(f"Exception: {e}"))
        Success(42)
        """
        return self

    def match(
        self,
        failure_function: Callable[[Exception], TypeResult],
        success_function: Callable[[TypeSource], TypeResult],
    ) -> TypeResult:
        """Try Monad specific function to handle railroad orientated types.

        Parameters
        ----------
        failure_function: Callable[[TypeErr], TypeResult]
            Callback function for either monads of type Failure
        success_function: Callable[[TypeSource], TypeResult]
            Callback function for either monads of type Success

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.match(lambda e: "Error", lambda x: f"Success: {x}")
        'Success: 10'
        """
        return success_function(self._success_value)

    def to_either(self) -> Either[Exception, TypeSource]:
        """Try Monad specific function to return an Either Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_either()
        Right(10)
        """
        return Right(self._success_value)

    def to_result(self) -> Result[TypeSource, Exception]:
        """Try Monad specific function to return an Result Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_result()
        Ok(10)
        """
        return Ok(self._success_value)

    def is_success(self) -> bool:
        """Try monad is success function

        Returns
        -------
        result: bool
            True: if try monad is of type success, False: if try monad is of type failure

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_success()
        True
        """
        return True

    def is_failure(self) -> bool:
        """Try monad is failure function

        Returns
        -------
        result: bool
            True: if try monad is of type failure, False: if try monad is of type success

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_failure()
        False
        """
        return False

    def __str__(self) -> str:
        """Returns the string representation of the Success Monad.

        Examples
        --------
        >>> str(Success(42))
        'Success(42)'
        """
        return f"Success({self._success_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is a Success wrapping a value of the same type and string representation.

        Examples
        --------
        >>> Success(1) == Success(1)
        True
        >>> Success(1) == Success(2)
        False
        """
        if isinstance(other, Success):
            other_success = cast(Success[Any], other)
            return str(self) == str(other_success) and type(
                self._success_value
            ) is type(other_success._success_value)
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Success Monad (same as __str__).

        Examples
        --------
        >>> repr(Success(10))
        'Success(10)'
        """
        return str(self)


class Failure:
    """The Failure variant of the Try Monad.

    Represents a computation that raised an exception. It wraps the raised `Exception` and, like
    `Success`, provides a functional interface for chaining operations, though `map`/`bind` on a
    `Failure` are no-ops since there is no success value to transform.
    """

    __slots__ = ("_failure_value",)
    __match_args__ = ("_failure_value",)

    def __init__(self, value: Exception):
        """Failure Monad constructor which takes a value of type Exception.

        Parameters
        ----------
        value: Exception
            Exception to be stored in the Failure Monad. Must be an instance of Exception.

        Examples
        --------
        >>> failure = Failure(ValueError("boom"))
        >>> print(failure)
        Failure(boom)
        """
        assert isinstance(value, Exception), "Failure value must be of type Exception"
        self._failure_value = value

    def map(self, function: Callable[[Any], TypeResult]) -> Try[TypeResult]:
        """Leaves the Failure value untouched, since this Try Monad is already a Failure.

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            Function which would have been applied to the Success value.

        Returns
        -------
        result: Try[TypeResult]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.map(lambda x: x + 1)
        Failure(boom)
        """
        return self

    def map_failure(self, function: Callable[[Exception], Exception]) -> Try[Any]:
        """Calls function with the Failure value. The function is executed while checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Exception]
            Function which takes an Exception and returns an Exception.

        Returns
        -------
        result: Try[Any]
            Returns a Failure with the function result, or a Failure of the raised exception.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.map_failure(lambda e: TypeError(f"wrapped: {e}"))
        Failure(wrapped: boom)
        """
        try:
            return Failure(function(self._failure_value))
        except Exception as e:
            return Failure(e)

    def bind(self, function: Callable[[Any], Try[TypeResult]]) -> Try[TypeResult]:
        """Leaves this Failure untouched, since the bind function is only called for Success.

        Parameters
        ----------
        function: Callable[[Any], Try[TypeResult]]
            Function which would have been applied to the Success value.

        Returns
        -------
        result: Try[TypeResult]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.bind(lambda x: Success(x + 1))
        Failure(boom)
        """
        return self

    def bind_failure(
        self, function: Callable[[Exception], Try[TypeSource]]
    ) -> Try[TypeSource]:
        """Calls function with the Failure value. The bind function is executed while checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Try[TypeSource]]
            Function which takes an Exception and returns a Try Monad of TypeSource.

        Returns
        -------
        result: Try[TypeSource]
            Returns the Try Monad from the function result, or a Failure if it raised an exception.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.bind_failure(lambda e: Success(0))
        Success(0)
        """
        try:
            return function(self._failure_value)
        except Exception as e:
            return Failure(e)

    def apply(self, applicative: Try[Callable[..., TypeResult]]) -> Try[TypeResult]:
        """Leaves this Failure untouched, since applying a function only happens for Success.

        Parameters
        ----------
        applicative: Try[Callable[..., TypeResult]]
            Applicative Try Monad which contains a function.

        Returns
        -------
        result: Try[TypeResult]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val.apply(func)
        Failure(boom)
        """
        return self

    def apply2(self, applicative_value: Try[Any]) -> Try[Any]:
        """Leaves this Failure untouched, since applying a value only happens for Success.

        Parameters
        ----------
        applicative_value: Try[Any]
            Try Monad containing a value which would have been applied to the function.

        Returns
        -------
        result: Try[Any]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[Callable[[int], int]] = Failure(ValueError("boom"))
        >>> other: Try[int] = Success(5)
        >>> val.apply2(other)
        Failure(boom)
        """
        return self

    def is_success_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Try Monad is Failure.

        Parameters
        ----------
        function: Callable[[Any], bool]
            Predicate function which would be applied to the Success value.

        Returns
        -------
        result: bool
            Returns False.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_success_and(lambda x: x > 5)
        False
        """
        return False

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        """Returns True if the Try Monad is Failure and the predicate returns True for the contained exception.

        Parameters
        ----------
        function: Callable[[Exception], bool]
            Predicate function applied to the Failure value.

        Returns
        -------
        result: bool
            Returns the predicate result.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_failure_and(lambda e: isinstance(e, ValueError))
        True
        """
        return function(self._failure_value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """Returns the default value, since this Try Monad is Failure.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned since the Try Monad is Failure.
        function: Callable[[Any], TypeResult]
            Function which would have been applied to the Success value.

        Returns
        -------
        result: TypeResult
            Returns the default value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.map_or("Fallback", lambda x: x)
        'Fallback'
        """
        return default_value

    def and_(self, other: Try[Any]) -> Try[Any]:
        """Returns this Failure, since the Try Monad is not Success.

        Parameters
        ----------
        other: Try[Any]
            Try Monad which would have been returned if this Try Monad was Success.

        Returns
        -------
        result: Try[Any]
            Returns this Failure.

        Examples
        --------
        >>> val1: Try[int] = Failure(ValueError("boom"))
        >>> val2: Try[int] = Success(2)
        >>> val1.and_(val2)
        Failure(boom)
        """
        return self

    def or_(self, other: Try[TypeSource]) -> Try[TypeSource]:
        """Returns `other`, since this Try Monad is Failure.

        Parameters
        ----------
        other: Try[TypeSource]
            Try Monad to be returned since this Try Monad is Failure.

        Returns
        -------
        result: Try[TypeSource]
            Returns `other`.

        Examples
        --------
        >>> val1: Try[int] = Failure(ValueError("boom"))
        >>> val2: Try[int] = Success(2)
        >>> val1.or_(val2)
        Success(2)
        """
        return other

    def zip(self, other: Try[Any]) -> Try[Any]:
        """Returns this Failure, since combining values only happens for Success.

        Parameters
        ----------
        other: Try[Any]
            Try Monad which would have been zipped with this Try Monad.

        Returns
        -------
        result: Try[Any]
            Returns this Failure.

        Examples
        --------
        >>> val1: Try[int] = Failure(ValueError("boom"))
        >>> val2: Try[int] = Success(2)
        >>> val1.zip(val2)
        Failure(boom)
        """
        return self

    def flatten(self) -> Try[Any]:
        """Returns this Failure unchanged, since there is no nested Success value to flatten.

        Returns
        -------
        result: Try[Any]
            Returns this Failure.

        Examples
        --------
        >>> val: Try[Try[int]] = Failure(ValueError("boom"))
        >>> val.flatten()
        Failure(boom)
        """
        return self

    def unwrap(self) -> Any:
        """Raises the Failure Exception, since this Try Monad is not Success.

        Returns
        -------
        result: Any
            Never returns, always raises the stored Exception.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap()
        Traceback (most recent call last):
            ...
        ValueError: boom
        """
        raise self._failure_value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the provided default value, since this Try Monad is Failure.

        Parameters
        ----------
        default_value: TypeSource
            Default value of TypeSource.

        Returns
        -------
        result: TypeSource
            Returns the default value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or(0)
        0
        """
        return default_value

    def unwrap_or_else(
        self, failure_function: Callable[[Exception], TypeSource]
    ) -> TypeSource:
        """Calls the provided function with the Failure value and returns its result.

        Parameters
        ----------
        failure_function: Callable[[Exception], TypeSource]
            Called with the Failure value and must return a value of type TypeSource.

        Returns
        -------
        result: TypeSource
            Returns the result of the function call.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or_else(lambda e: 0)
        0
        """
        return failure_function(self._failure_value)

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        """Returns the Failure Exception, since this Try Monad is Failure.

        Parameters
        ----------
        default_value: Exception
            Default value of type Exception, ignored since this Try Monad is Failure.

        Returns
        -------
        result: Exception
            Returns the Failure value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_failure_or(TypeError("default"))
        ValueError('boom')
        """
        return self._failure_value

    def inspect(self, function: Callable[[Any], None]) -> Try[Any]:
        """Leaves this Failure untouched, since inspecting the Success value only happens for Success.

        Parameters
        ----------
        function: Callable[[Any], None]
            Inspection function which would have taken the Success value.

        Returns
        -------
        result: Try[Any]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.inspect(lambda x: print(f"Value: {x}"))
        Failure(boom)
        """
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Try[Any]:
        """Inspect the Try monad Exception value.

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception value of the Try monad.

        Returns
        -------
        result: Try[Any]
            Returns this Failure unchanged.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.inspect_failure(lambda e: print(f"Exception: {e}"))
        Exception: boom
        Failure(boom)
        """
        function(self._failure_value)
        return self

    def match(
        self,
        failure_function: Callable[[Exception], TypeResult],
        success_function: Callable[[Any], TypeResult],
    ) -> TypeResult:
        """Try Monad specific function to handle railroad orientated types.

        Parameters
        ----------
        failure_function: Callable[[Exception], TypeResult]
            Callback function for Try monads of type Failure.
        success_function: Callable[[Any], TypeResult]
            Callback function for Try monads of type Success.

        Returns
        -------
        result: TypeResult
            Returns the result of calling failure_function with the Failure value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.match(lambda e: f"Error: {e}", lambda x: "Success")
        'Error: boom'
        """
        return failure_function(self._failure_value)

    def to_either(self) -> Either[Exception, Any]:
        """Try Monad specific function to return an Either Monad.

        Returns
        -------
        either: Either[Exception, Any]
            Returns the Try Monad as a Left Either Monad.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.to_either()
        Left(boom)
        """
        return Left(self._failure_value)

    def to_result(self) -> Result[Any, Exception]:
        """Try Monad specific function to return a Result Monad.

        Returns
        -------
        result: Result[Any, Exception]
            Returns the Try Monad as an Err Result Monad.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.to_result()
        Err(boom)
        """
        return Err(self._failure_value)

    def is_success(self) -> bool:
        """Try monad is success function.

        Returns
        -------
        result: bool
            True: if try monad is of type success, False: if try monad is of type failure

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_success()
        False
        """
        return False

    def is_failure(self) -> bool:
        """Try monad is failure function.

        Returns
        -------
        result: bool
            True: if try monad is of type failure, False: if try monad is of type success

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_failure()
        True
        """
        return True

    def __str__(self) -> str:
        """Returns the string representation of the Failure Monad.

        Examples
        --------
        >>> str(Failure(ValueError("boom")))
        'Failure(boom)'
        """
        return f"Failure({self._failure_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is a Failure wrapping an exception of the same type and string representation.

        Examples
        --------
        >>> Failure(ValueError("boom")) == Failure(ValueError("boom"))
        True
        >>> Failure(ValueError("boom")) == Failure(TypeError("boom"))
        False
        """
        if isinstance(other, Failure):
            return str(self) == str(other) and type(self._failure_value) is type(
                other._failure_value
            )
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Failure Monad (same as __str__).

        Examples
        --------
        >>> repr(Failure(ValueError("boom")))
        'Failure(boom)'
        """
        return str(self)


Try: TypeAlias = Success[TypeSource] | Failure


def safe(function: Callable[[], TypeResult]) -> Try[TypeResult]:
    """Either Monad method (try_except) which wraps a function that may raise an exception.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        Callable function which may raise an exception

    Returns
    -------
    either: Either[Exception, TypeResult]
        Returns an Either Monad which contains either the function result or an Exception with a message added.

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> safe(risky_call)
    Failure(Boom)
    >>> def safe_call():
    ...     return 42
    >>> safe(safe_call)
    Success(42)
    """
    try:
        return Success(function())
    except Exception as e:
        return Failure(e)
