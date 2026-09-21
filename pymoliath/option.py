"""
# Option Monad

The Option Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Rust: [Option](https://doc.rust-lang.org/std/option/)
* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

This implementation is heavily inspired by the Rust `Option` type and the Haskell `Maybe` monad -
see `pymoliath.maybe` for the sibling implementation using `Just`/`Nothing` naming.

The `Option` type is a sum type that can be either `Some` or `Nil`.
In this implementation, it is represented as a Union type in Python.

```python
Option = Some[TypeSource] | Nil[TypeSource]
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
    .map(get_profile)
    .map(get_permission)
    .unwrap_or("Default Permission"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of an `Option` monad. Because the `Some` and `Nil` classes are designed to be compatible with Python's `match` statement, you can easily branch your logic based on whether a value exists without manually checking for `None` or using complex `if-is_some()` logic.

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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.result import Result

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")
TypeErr = TypeVar("TypeErr")


class Some(Generic[TypeSource]):
    """The Some variant of the Option Monad.

    Represents a computation that successfully yielded a value. It wraps a value of type
    `TypeSource` and provides a functional interface for chaining operations.

    Parameters
    ----------
    value: TypeSource
        Value to be stored in the Some Option Monad.
    """

    __slots__ = ("_value",)
    __match_args__ = ("_value",)

    def __init__(self, value: TypeSource):
        """Some Monad constructor which takes a value of type TypeSource.

        Parameters
        ----------
        value: TypeSource
            Value to be stored in the Some Monad.

        Examples
        --------
        >>> some = Some(42)
        >>> print(some._value)
        42
        """
        self._value = value

    def map(self, function: Callable[[TypeSource], TypeResult]) -> Option[TypeResult]:
        """Option monad functor interface (>=, map).

        Definition: M(a) >= f: a -> b => M(b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        option: Option[TypeResult]
            Returns a Option Monad with the function result if Monad is a Some or otherwise a Nothing.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.map(lambda x: x + 1)
        Some(6)
        >>> empty: Option[int] = Nil()
        >>> empty.map(lambda x: x + 1)
        Nil()
        >>> names: Option[str] = Some("alice")
        >>> names.map(str.upper)
        Some(ALICE)
        """
        return Some(function(self._value))

    def bind(
        self, function: Callable[[TypeSource], Option[TypeResult]]
    ) -> Option[TypeResult]:
        """Option Monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], Option[TypeResult]]
            Function which takes a value of type TypeSource and returns a option monad of type TypeResult.

        Returns
        -------
        option: Option[TypeResult]
            Returns a Option Monad with the function result if the Monad is a Some or otherwise a Nothing

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> def get_next(x: int) -> Option[int]:
        ...     return Some(x + 1) if x < 10 else Nil()
        >>> val.bind(get_next)
        Some(6)
        >>> val_large: Option[int] = Some(10)
        >>> val_large.bind(get_next)
        Nil()
        """
        return function(self._value)

    def apply(
        self, applicative: Option[Callable[..., TypeResult]]
    ) -> Option[TypeResult]:
        """Option Monad applicative interface for Option Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Option[TypeApplicative] (TypeApplicative: any callable type)
            Applicative option monad which contains a function and will be applied to the option monad containing
            a value.

        Returns
        -------
        option: Option[TypeResult]
            Applies a option monad containing a value of type TypeSource to a option monad containing a function.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> func: Option[Callable[[int], int]] = Some(lambda x: x * 2)
        >>> val.apply(func)
        Some(20)
        >>> empty: Option[int] = Nil()
        >>> val.apply(empty)
        Nil()
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Option[TypeResult]:
            """Maps the applicative's function, curried, over this Some value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Some[Callable[..., TypeResult]],
        applicative_value: Option[Any],
    ) -> Option[TypeResult]:
        """Option Monad applicative interface for Option Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Option[TypePure]
            Option monad value which will be applied to the option monad containing a function

        Returns
        -------
        option: Option[TypeResult]
            Applies a option monad containing a function to a option monad of type TypeSource (value or function).

        Examples
        --------
        >>> func_monad: Option[Callable[[int], int]] = Some(lambda y: 10 + y)
        >>> val_monad: Option[int] = Some(5)
        >>> func_monad.apply2(val_monad)
        Some(15)
        >>> empty_val: Option[int] = Nil()
        >>> func_monad.apply2(empty_val)
        Nil()
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Option[TypeResult]:
            """Maps the curried applicative function, held by this Some, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def filter(
        self, filter_function: Callable[[TypeSource], bool]
    ) -> Option[TypeSource]:
        """Returns a Some if filter function is True and Option Monad is of type Some, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            Filter function which will be applied to Some if not Nothing.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some if the Option Monad is of type Some and filter function returns True otherwise Nothing.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.filter(lambda x: x > 5)
        Some(10)
        >>> val.filter(lambda x: x < 5)
        Nil()
        >>> empty: Option[int] = Nil()
        >>> empty.filter(lambda x: x > 5)
        Nil()
        """
        if filter_function(self._value):
            return self
        return Nil()

    def is_some_and(self, function: Callable[[TypeSource], bool]) -> bool:
        """Returns True if the Option Monad is a Some and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeSource], bool]
            Predicate function applied to the Some value.

        Returns
        -------
        result: bool
            Returns the predicate result.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.is_some_and(lambda x: x > 5)
        True
        >>> val.is_some_and(lambda x: x < 5)
        False
        """
        return function(self._value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeSource], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Some value, or returns the default value if Nil.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Option Monad is Nil.
        function: Callable[[TypeSource], TypeResult]
            Function applied to the Some value.

        Returns
        -------
        result: TypeResult
            Returns the function result.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        >>> empty: Option[int] = Nil()
        >>> empty.map_or(0, lambda x: x * 2)
        0
        """
        return function(self._value)

    def and_(self, other: Option[TypeResult]) -> Option[TypeResult]:
        """Returns `other` if the Option Monad is a Some, otherwise Nil.

        Parameters
        ----------
        other: Option[TypeResult]
            Option Monad to be returned if this Option Monad is a Some.

        Returns
        -------
        result: Option[TypeResult]
            Returns `other`.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> other: Option[int] = Some(2)
        >>> val.and_(other)
        Some(2)
        >>> empty: Option[int] = Nil()
        >>> empty.and_(other)
        Nil()
        """
        return other

    def or_(self, other: Option[TypeSource]) -> Option[TypeSource]:
        """Returns this Option Monad if it is a Some, otherwise `other`.

        Parameters
        ----------
        other: Option[TypeSource]
            Option Monad to be returned if this Option Monad is a Nil.

        Returns
        -------
        result: Option[TypeSource]
            Returns this Some.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> other: Option[int] = Some(2)
        >>> val.or_(other)
        Some(1)
        >>> empty: Option[int] = Nil()
        >>> empty.or_(other)
        Some(2)
        """
        return self

    def zip(self, other: Option[TypePure]) -> Option[Tuple[TypeSource, TypePure]]:
        """Combines this Option Monad with another into an Option Monad of a tuple, or Nil if either is Nil.

        Parameters
        ----------
        other: Option[TypePure]
            Option Monad to be zipped with this Option Monad.

        Returns
        -------
        result: Option[Tuple[TypeSource, TypePure]]
            Returns Some of a tuple of both values, or Nil.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> other: Option[int] = Some(2)
        >>> val.zip(other)
        Some((1, 2))
        >>> empty: Option[int] = Nil()
        >>> val.zip(empty)
        Nil()
        """
        return other.map(lambda o: (self._value, o))

    @overload
    def flatten(self: Some[Some[TypeResult]]) -> Option[TypeResult]: ...

    @overload
    def flatten(self: Some[Nil[TypeResult]]) -> Option[TypeResult]: ...

    def flatten(self) -> Option[Any]:
        """Flattens a nested Option Monad by one level.

        Returns
        -------
        result: Option[TypeResult]
            Returns the nested Option Monad.

        Examples
        --------
        >>> val: Option[Option[int]] = Some(Some(1))
        >>> val.flatten()
        Some(1)
        >>> empty_inner: Option[Option[int]] = Some(Nil())
        >>> empty_inner.flatten()
        Nil()
        """
        return cast(Option[Any], self._value)

    def ok_or(self, err_value: TypeErr) -> Result[TypeSource, TypeErr]:
        """Converts the Option Monad into a Result Monad, using `err_value` as the Err value if Nil.

        Parameters
        ----------
        err_value: TypeErr
            Error value to be used if the Option Monad is Nil.

        Returns
        -------
        result: Result[TypeSource, TypeErr]
            Returns Ok with the Some value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.ok_or("Error")
        Ok(1)
        """
        from pymoliath.result import Ok

        return Ok(self._value)

    def ok_or_else(
        self, err_function: Callable[[], TypeErr]
    ) -> Result[TypeSource, TypeErr]:
        """Converts the Option Monad into a Result Monad, calling `err_function` for the Err value if Nil.

        Parameters
        ----------
        err_function: Callable[[], TypeErr]
            Function called to produce the error value if the Option Monad is Nil.

        Returns
        -------
        result: Result[TypeSource, TypeErr]
            Returns Ok with the Some value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.ok_or_else(lambda: "Error")
        Ok(1)
        """
        from pymoliath.result import Ok

        return Ok(self._value)

    def unwrap(self) -> TypeSource:
        """Returns the internal value of the Some or raises an exception if Nothing.

        Returns
        -------
        value: TypeSource
            Returns the Option value or a default value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.unwrap()
        1
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Unwrap error on Option monad
        """
        return self._value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the internal value of the Some or default value if the Monad is a Nothing.

        Parameters
        ----------
        default_value: TypeSource
            Default value to be returned if the Monad is a Nothing

        Returns
        -------
        value: TypeSource
            Returns the Option value or a default value.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.unwrap_or(0)
        1
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap_or(0)
        0
        """
        return self._value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """Returns the internal value of the Some or default value if the Monad is a Nothing.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            Function to be called when the Option value is of type Nothing

        Returns
        -------
        value: TypeSource
            Returns the Option value or calls the nothing function.

        Examples
        --------
        >>> val: Option[int] = Some(1)
        >>> val.unwrap_or_else(lambda: 0)
        1
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap_or_else(lambda: 0)
        0
        """
        return self._value

    def inspect(self, function: Callable[[TypeSource], None]) -> Option[TypeSource]:
        """Inspect the Option monad value of TypeSource

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            Inspection function which takes the option value if not nothing

        Returns
        -------
        option: Option[TypeSource]

        Examples
        --------
        >>> val: Option[int] = Some(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Some(42)
        >>> empty: Option[int] = Nil()
        >>> empty.inspect(lambda x: print(f"Value is: {x}"))
        Nil()
        """
        function(self._value)
        return self

    def match(
        self,
        some_function: Callable[[TypeSource], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """The option function takes a function and a default value. If the Option value is Nothing, the function returns
        the default value. Otherwise, it applies the function to the value inside a Some monad and returns the result.

        Parameter
        ---------
        callback: Callable[[TypeSource], TypeSource]
          Callback function if the Option Monad is a Some
        default: TypeResult
          Default value to be returned if the Option Monad is Nothing

        Returns
        -------
        result: TypeSource

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.match(lambda x: x * 2, lambda: 0)
        20
        >>> empty: Option[int] = Nil()
        >>> empty.match(lambda x: x * 2, lambda: 0)
        0
        """
        return some_function(self._value)

    def is_nothing(self) -> bool:
        """Returns False, since this Option Monad is a Some.

        Returns
        -------
        result: bool
            Returns False.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.is_nothing()
        False
        """
        return False

    def is_some(self) -> bool:
        """Returns True, since this Option Monad is a Some.

        Returns
        -------
        result: bool
            Returns True.

        Examples
        --------
        >>> val: Option[int] = Some(5)
        >>> val.is_some()
        True
        """
        return True

    def to_optional(self) -> TypeSource | None:
        """Converts the Option Monad into a standard Python optional value.

        Returns
        -------
        value: TypeSource | None
            Returns the Some value.

        Examples
        --------
        >>> val: Option[int] = Some(10)
        >>> val.to_optional()
        10
        >>> empty: Option[int] = Nil()
        >>> print(empty.to_optional())
        None
        """
        return self._value

    @staticmethod
    def from_optional(value: TypeSource | None) -> Option[TypeSource]:
        """Converts a standard Python optional value into an Option Monad.

        Parameters
        ----------
        value: TypeSource | None
            Optional value to be converted into an Option Monad.

        Returns
        -------
        option: Option[TypeSource]
            Returns Some if `value` is not None, otherwise Nil.

        Examples
        --------
        >>> Some(5) == Some.from_optional(5)
        True
        >>> Nil() == Some.from_optional(None)
        True
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Some Monad.

        Examples
        --------
        >>> str(Some(42))
        'Some(42)'
        """
        return f"Some({self._value})"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `other` is also a Some Monad with an equal string representation.

        Examples
        --------
        >>> Some(1) == Some(1)
        True
        >>> Some(1) == Some(2)
        False
        """
        return isinstance(__o, Some) and str(self) == str(__o)

    def __repr__(self) -> str:
        """Returns the string representation of the Some Monad (same as __str__).

        Examples
        --------
        >>> repr(Some(10))
        'Some(10)'
        """
        return str(self)


class Nil(Generic[TypeSource]):
    """The Nil variant of the Option Monad.

    Nil holds no data and is generic over `TypeSource` even though it does not store any internal
    value: a phantom type parameter (like Rust's `Option<T>::None`) that lets `map`/`bind`/`apply`/
    `filter`/`inspect` propagate real types instead of collapsing to `Any`.

    Examples
    -------
    >>> empty = Nil()
    >>> print(type(empty))
    <class 'pymoliath.option.Nil'>
    """

    __slots__ = ()

    _instance: ClassVar[Nil[Any] | None] = None

    def __new__(cls) -> Nil[TypeSource]:
        """
        Returns the single shared Nil instance, creating it on first call.

        Since `Nil` holds no data and is conceptually equivalent to any other `Nil` of the same
        type, this implementation uses a singleton pattern to optimize memory and performance by
        sharing a single instance.

        Returns
        -------
        Nil[TypeSource]
            The singleton instance of the Nil monad.

        Examples
        -------
        >>> n1 = Nil()
        >>> n2 = Nil()
        >>> n1 is n2
        True
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Nil[TypeSource], cls._instance)

    def map(self, function: Callable[[Any], TypeResult]) -> Option[TypeResult]:
        """Option monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            Function which would be applied to the value if this Option Monad were a Some.

        Returns
        -------
        option: Option[TypeResult]
            Returns Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.map(lambda x: x + 1)
        Nil()
        >>> empty.map(str.upper)
        Nil()
        """
        return Nil()

    def bind(self, function: Callable[[Any], Option[TypeResult]]) -> Option[TypeResult]:
        """Option Monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[Any], Option[TypeResult]]
            Function which would be applied to the value if this Option Monad were a Some.

        Returns
        -------
        option: Option[TypeResult]
            Returns Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> def get_next(x: int) -> Option[int]:
        ...     return Some(x + 1) if x < 10 else Nil()
        >>> empty.bind(get_next)
        Nil()
        """
        return Nil()

    def apply(
        self, applicative: Option[Callable[..., TypeResult]]
    ) -> Option[TypeResult]:
        """Option Monad applicative interface for Option Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Option[TypeApplicative] (TypeApplicative: any callable type)
            Applicative option monad which would be applied to the option monad containing a value.

        Returns
        -------
        option: Option[TypeResult]
            Returns Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> func: Option[Callable[[int], int]] = Some(lambda x: x * 2)
        >>> empty.apply(func)
        Nil()
        """
        return Nil()

    def apply2(self, applicative_value: Option[Any]) -> Option[Any]:
        """Option Monad applicative interface for Option Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Option[TypePure]
            Option monad value which would be applied to the option monad containing a function.

        Returns
        -------
        option: Option[Any]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> val: Option[int] = Some(5)
        >>> empty.apply2(val)
        Nil()
        """
        return self

    def filter(self, filter_function: Callable[[Any], bool]) -> Option[TypeSource]:
        """Returns a Some if filter function is True and Option Monad is of type Some, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[Any], bool]
            Filter function which would be applied to the value if this Option Monad were a Some.

        Returns
        -------
        result: Option[TypeSource]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.filter(lambda x: x > 5)
        Nil()
        """
        return self

    def is_some_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns True if the Option Monad is a Some and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[Any], bool]
            Predicate function which would be applied to the value if this Option Monad were a Some.

        Returns
        -------
        result: bool
            Returns False, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.is_some_and(lambda x: x > 5)
        False
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Some value, or returns the default value if Nil.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Option Monad is Nil.
        function: Callable[[Any], TypeResult]
            Function which would be applied to the value if this Option Monad were a Some.

        Returns
        -------
        result: TypeResult
            Returns `default_value`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.map_or(0, lambda x: x * 2)
        0
        """
        return default_value

    def and_(self, other: Option[Any]) -> Option[TypeSource]:
        """Returns `other` if the Option Monad is a Some, otherwise Nil.

        Parameters
        ----------
        other: Option[Any]
            Option Monad which would be returned if this Option Monad were a Some.

        Returns
        -------
        result: Option[TypeSource]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> other: Option[int] = Some(2)
        >>> empty.and_(other)
        Nil()
        """
        return self

    def or_(self, other: Option[TypeSource]) -> Option[TypeSource]:
        """Returns this Option Monad if it is a Some, otherwise `other`.

        Parameters
        ----------
        other: Option[TypeSource]
            Option Monad to be returned since this Option Monad is a Nil.

        Returns
        -------
        result: Option[TypeSource]
            Returns `other`.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> other: Option[int] = Some(2)
        >>> empty.or_(other)
        Some(2)
        """
        return other

    def zip(self, other: Option[Any]) -> Option[Any]:
        """Combines this Option Monad with another into an Option Monad of a tuple, or Nil if either is Nil.

        Parameters
        ----------
        other: Option[Any]
            Option Monad which would be zipped with this Option Monad if this Option Monad were a Some.

        Returns
        -------
        result: Option[Any]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> other: Option[int] = Some(2)
        >>> empty.zip(other)
        Nil()
        """
        return self

    def flatten(self) -> Option[TypeSource]:
        """Flattens a nested Option Monad by one level.

        Returns
        -------
        result: Option[TypeSource]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[Option[int]] = Nil()
        >>> empty.flatten()
        Nil()
        """
        return self

    def ok_or(self, err_value: TypeErr) -> Result[Any, TypeErr]:
        """Converts the Option Monad into a Result Monad, using `err_value` as the Err value if Nil.

        Parameters
        ----------
        err_value: TypeErr
            Error value to be used if the Option Monad is Nil.

        Returns
        -------
        result: Result[TypeSource, TypeErr]
            Returns Err with `err_value`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.ok_or("Error")
        Err(Error)
        """
        from pymoliath.result import Err

        return Err(err_value)

    def ok_or_else(self, err_function: Callable[[], TypeErr]) -> Result[Any, TypeErr]:
        """Converts the Option Monad into a Result Monad, calling `err_function` for the Err value if Nil.

        Parameters
        ----------
        err_function: Callable[[], TypeErr]
            Function called to produce the error value if the Option Monad is Nil.

        Returns
        -------
        result: Result[TypeSource, TypeErr]
            Returns Err with the result of `err_function`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.ok_or_else(lambda: "Error")
        Err(Error)
        """
        from pymoliath.result import Err

        return Err(err_function())

    def unwrap(self) -> TypeSource:
        """Raises an exception, since a Nil Monad has no internal value to unwrap.

        Returns
        -------
        value: TypeSource
            Never returns; always raises an Exception.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Unwrap error on Option monad
        """
        raise Exception("Unwrap error on Option monad")

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the internal value of the Some or default value if the Monad is a Nothing.

        Parameters
        ----------
        default_value: TypeSource
            Default value to be returned if the Monad is a Nothing

        Returns
        -------
        value: TypeSource
            Returns `default_value`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap_or(0)
        0
        """
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """Returns the internal value of the Some or default value if the Monad is a Nothing.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            Function to be called when the Option value is of type Nothing

        Returns
        -------
        value: TypeSource
            Returns the result of calling `nothing_function`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.unwrap_or_else(lambda: 0)
        0
        """
        return nothing_function()

    def inspect(self, function: Callable[[Any], None]) -> Option[TypeSource]:
        """Inspect the Option monad value of TypeSource

        Parameters
        ----------
        function: Callable[[Any], None]
            Inspection function which would be called with the value if this Option Monad were a Some.

        Returns
        -------
        option: Option[TypeSource]
            Returns this Nil, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.inspect(lambda x: print(f"Value is: {x}"))
        Nil()
        """
        return self

    def match(
        self,
        some_function: Callable[[Any], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """The option function takes a function and a default value. If the Option value is Nothing, the function returns
        the default value. Otherwise, it applies the function to the value inside a Some monad and returns the result.

        Parameter
        ---------
        callback: Callable[[TypeSource], TypeSource]
          Callback function if the Option Monad is a Some
        default: TypeResult
          Default value to be returned if the Option Monad is Nothing

        Returns
        -------
        result: TypeSource
            Returns the result of calling `nothing_function`, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.match(lambda x: x * 2, lambda: 0)
        0
        """
        return nothing_function()

    def is_nothing(self) -> bool:
        """Returns True, since this Option Monad is a Nil.

        Returns
        -------
        result: bool
            Returns True.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.is_nothing()
        True
        """
        return True

    def is_some(self) -> bool:
        """Returns False, since this Option Monad is a Nil.

        Returns
        -------
        result: bool
            Returns False.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> empty.is_some()
        False
        """
        return False

    def to_optional(self) -> TypeSource | None:
        """Converts the Option Monad into a standard Python optional value.

        Returns
        -------
        value: TypeSource | None
            Returns None, since this Option Monad is Nil.

        Examples
        --------
        >>> empty: Option[int] = Nil()
        >>> print(empty.to_optional())
        None
        """
        return None

    @staticmethod
    def from_optional(value: TypeSource | None) -> Option[TypeSource]:
        """Converts a standard Python optional value into an Option Monad.

        Parameters
        ----------
        value: TypeSource | None
            Optional value to be converted into an Option Monad.

        Returns
        -------
        option: Option[TypeSource]
            Returns Some if `value` is not None, otherwise Nil.

        Examples
        --------
        >>> Some(5) == Nil.from_optional(5)
        True
        >>> Nil() == Nil.from_optional(None)
        True
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Nil Monad.

        Examples
        --------
        >>> str(Nil())
        'Nil()'
        """
        return "Nil()"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `other` is also a Nil Monad.

        Examples
        --------
        >>> Nil() == Nil()
        True
        >>> Nil() == Some(1)
        False
        """
        return isinstance(__o, Nil)

    def __repr__(self) -> str:
        """Returns the string representation of the Nil Monad (same as __str__).

        Examples
        --------
        >>> repr(Nil())
        'Nil()'
        """
        return str(self)


Option: TypeAlias = Some[TypeSource] | Nil[TypeSource]


def from_optional(value: TypeSource | None) -> Option[TypeSource]:
    """Converts a standard Python optional value into an Option Monad.

    Parameters
    ----------
    value: TypeSource | None
        Optional value to be converted into an Option Monad.

    Returns
    -------
    option: Option[TypeSource]
        Returns Some containing `value`, or Nil if `value` is None.

    Examples
    --------
    >>> from_optional("hello")
    Some(hello)
    >>> from_optional(None)
    Nil()
    """
    if value is None:
        return Nil()
    return Some(value)


def safe(function: Callable[[], TypeResult]) -> Option[TypeResult]:
    """Calls a function which might raise an Exception and returns Some with the result, otherwise Nil.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        Callable function which may raise an exception.

    Returns
    -------
    option: Option[TypeResult]
        Returns Some containing the function result, or Nil if an exception was raised.

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> safe(risky_call)
    Nil()
    >>> def safe_call():
    ...     return 42
    >>> safe(safe_call)
    Some(42)
    """
    try:
        return Some(function())
    except Exception:
        return Nil()
