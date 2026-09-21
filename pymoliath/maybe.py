"""
# Maybe Monad

The Maybe Monad is a container used to represent computations that may fail or return nothing.
It encapsulates values that could be `None`, allowing for a functional approach to handling optionality by
chaining operations without constant explicit null checks.

* Haskell: [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data.Maybe.html)
* Rust: [Option](https://doc.rust-lang.org/std/option/)

This implementation is heavily inspired by the Haskell `Maybe` monad and the Rust `Option` type.

The `Maybe` type is a sum type that can be either `Just` or `Nothing`.
In this implementation, it is represented as a Union type in Python.

```python
Maybe = Just[TypeSource] | Nothing[TypeSource]
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
    .map(get_profile)
    .map(get_permission)
    .unwrap_or("Default Permission"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Maybe` monad. Because the `Just` and `Nothing` classes are designed to be compatible with Python's `match` statement, you can easily branch your logic based on whether a value exists without manually checking for `None` or using complex `if-is_just()` logic.

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

from pymoliath.either import Right
from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.either import Either

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")
TypeLeft = TypeVar("TypeLeft")


class Just(Generic[TypeSource]):
    """
    The Just variant of the Maybe Monad.

    Represents a computation that successfully yielded a value. It wraps a
    value of type `TypeSource` and provides a functional interface for
    chaining operations.

    Parameters
    ----------
    value: TypeSource
        The value contained within the Just monad.
    """

    __slots__ = ("_value",)
    __match_args__ = ("_value",)

    def __init__(self, value: TypeSource):
        """Just Monad constructor which takes a value of type TypeSource.

        Parameters
        ----------
        value: TypeSource
            Value to be stored in the Just Monad.

        Examples
        --------
        >>> just = Just(42)
        >>> print(just._value)
        42
        """
        self._value = value

    def map(self, function: Callable[[TypeSource], TypeResult]) -> Maybe[TypeResult]:
        """Apply a function to the value inside the Just container.

        The `map` method is the standard Functor interface for the Maybe monad.
        It allows for transforming the internal value while preserving the
        structure of the Monad. If the container is `Just`, the function
        is applied to the value; if it is `Nothing`, the result is
        automatically returned as `Nothing`.

        Definition: M(a) >= f: a -> b => M(b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            A function that accepts a value of type `TypeSource` and
            returns a value of type `TypeResult`.

        Returns
        -------
        maybe: Maybe[TypeResult]
            A new `Maybe` containing the result of the function application
            if the current instance is `Just`, otherwise returns `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.map(lambda x: x + 1)
        Just(6)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map(lambda x: x + 1)
        Nothing()
        >>> names: Maybe[str] = Just("alice")
        >>> names.map(str.upper)
        Just(ALICE)
        """
        return Just(function(self._value))

    def bind(
        self, function: Callable[[TypeSource], Maybe[TypeResult]]
    ) -> Maybe[TypeResult]:
        """Maybe Monad bind interface (>>=, bind, flatMap).

        The bind method allows for chaining operations where the transformation
        function itself returns a `Maybe` type. This prevents nested `Maybe`
        types (e.g., `Just(Just(x))`) by extracting the inner value.

        Parameters
        ----------
        function: Callable[[TypeSource], Maybe[TypeResult]]
            A function that accepts a value of type `TypeSource` and returns
            a `Maybe` monad of type `TypeResult`.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns the result of applying the function to the internal value.
            If the monad is `Just`, the function is executed; if `Nothing`,
            the function is skipped and `Nothing` is returned.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> def get_next_maybe(x: int) -> Maybe[int]:
        ...     return Just(x + 1) if x < 10 else Nothing()
        >>> val.bind(get_next_maybe)
        Just(6)
        >>> val_large: Maybe[int] = Just(10)
        >>> val_large.bind(get_next_maybe)
        Nothing()
        """
        return function(self._value)

    def apply(self, applicative: Maybe[Callable[..., TypeResult]]) -> Maybe[TypeResult]:
        """Maybe Monad applicative interface for Maybe Monads containing a value (<*>).

        The apply method takes a `Maybe` containing a function and applies it
        to the value contained within this `Just` monad.

        Parameters
        ----------
        applicative: Maybe[Callable[..., TypeResult]]
            An applicative maybe monad containing a callable function.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Applies the function contained in the `applicative` monad to the
            value contained in this monad. If either monad is `Nothing`,
            the result is `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> func: Maybe[Callable[[int], int]] = Just(lambda x: x * 2)
        >>> val.apply(func)
        Just(20)
        >>> empty: Maybe[int] = Nothing()
        >>> val.apply(empty)
        Nothing()
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Maybe[TypeResult]:
            """Maps the applicative's function, curried, over this Just value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Just[Callable[..., TypeResult]], applicative_value: Maybe[Any]
    ) -> Maybe[TypeResult]:
        """Maybe Monad applicative interface for Maybe Monads containing a function (<*>).

        The `apply2` method allows for applying a function contained within a
        `Just` monad to a value contained within another `Maybe` monad.
        This is a key part of the Applicative Functor pattern, allowing for
        independent computations to be combined.

        Parameters
        ----------
        applicative_value: Maybe[Any]
            The Maybe monad containing the value to which the function will be applied.

        Returns
        -------
        maybe: Maybe[TypeResult]
            The result of applying the function contained in this monad to the
            value contained in `applicative_value`. If either monad is `Nothing`,
            the result is `Nothing`.

        Examples
        --------
        >>> # A function that adds two numbers
        >>> def add(x: int, y: int) -> int:
        ...     return x + y
        >>>
        >>> # A Just monad containing the function (curried)
        >>> func_monad: Maybe[Callable[[int], int]] = Just(lambda y: add(10, y))
        >>>
        >>> # A value monad
        >>> val_monad: Maybe[int] = Just(5)
        >>>
        >>> # Apply the function to the value
        >>> func_monad.apply2(val_monad)
        Just(15)
        >>>
        >>> # If either is Nothing, the result is Nothing
        >>> empty_val: Maybe[int] = Nothing()
        >>> func_monad.apply2(empty_val)
        Nothing()
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Maybe[TypeResult]:
            """Maps the curried applicative function, held by this Just, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def filter(
        self, filter_function: Callable[[TypeSource], bool]
    ) -> Maybe[TypeSource]:
        """Filters the contained value based on a predicate.

        If the current Maybe instance is `Just` and the `filter_function`
        returns `True` for the contained value, the original `Just` is
        returned. Otherwise, it returns `Nothing`.

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            A predicate function that accepts the value inside the `Just`
            monad and returns a boolean.

        Returns
        -------
        result: Maybe[TypeSource]
            The original `Just` instance if the predicate is met,
            otherwise `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.filter(lambda x: x > 5)
        Just(10)
        >>> val.filter(lambda x: x < 5)
        Nothing()
        >>> empty: Maybe[int] = Nothing()
        >>> empty.filter(lambda x: x > 5)
        Nothing()
        """
        if filter_function(self._value):
            return self
        return Nothing()

    def is_just_and(self, function: Callable[[TypeSource], bool]) -> bool:
        """
        Checks if the Maybe Monad is a Just and satisfies a given predicate.

        This method evaluates a predicate function against the value contained
        within the Just monad. If the monad is Just, it returns the boolean
        result of the function. If the monad is Nothing, it returns False.

        Parameters
        ----------
        function: Callable[[TypeSource], bool]
            A predicate function that accepts the value inside the Just
            monad and returns a boolean result.

        Returns
        -------
        result: bool
            Returns True if the monad is Just and the predicate is satisfied,
            otherwise returns False.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.is_just_and(lambda x: x > 5)
        True
        >>> val.is_just_and(lambda x: x < 5)
        False
        >>> empty: Maybe[int] = Nothing()
        >>> empty.is_just_and(lambda x: x > 5)
        False
        """
        return function(self._value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeSource], TypeResult]
    ) -> TypeResult:
        """Apply a transformation function to the Just value, providing a fallback value if Nothing.

        This method is a convenience shorthand for chaining a `map` operation followed by
        an `unwrap_or` call. It allows you to transform the internal value of a
        Just monad and specify a default result for the case where the monad is Nothing
        in a single, expressive operation.

        Parameters
        ----------
        default_value: TypeResult
            The fallback value to be returned if the current Maybe instance is Nothing.
        function: Callable[[TypeSource], TypeResult]
            A transformation function that accepts the value inside the Just
            monad and returns a value of type `TypeResult`.

        Returns
        -------
        result: TypeResult
            The result of applying `function` to the internal value if the
            monad is Just; otherwise, the `default_value`.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map_or(0, lambda x: x * 2)
        0
        >>> name: Maybe[str] = Just("alice")
        >>> name.map_or("Guest", str.upper)
        'ALICE'
        >>> missing: Maybe[str] = Nothing()
        >>> missing.map_or("Guest", str.upper)
        'Guest'
        """
        return function(self._value)

    def and_(self, other: Maybe[TypeResult]) -> Maybe[TypeResult]:
        """
        Returns `other` if the Maybe Monad is a Just, otherwise Nothing.

        This operation allows for chaining computations where the continuation
        depends on the success of the preceding operation. If the current
        monad is `Nothing`, the operation short-circuits and returns `Nothing`.

        Parameters
        ----------
        other : Maybe[TypeResult]
            The Maybe monad to be returned if the current instance is `Just`.

        Returns
        -------
        result : Maybe[TypeResult]
            Returns the `other` Maybe monad if the current instance is `Just`,
            otherwise returns `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> other: Maybe[int] = Just(2)
        >>> val.and_(other)
        Just(2)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.and_(other)
        Nothing()
        """
        return other

    def or_(self, other: Maybe[TypeSource]) -> Maybe[TypeSource]:
        """Returns this Maybe Monad if it is a Just, otherwise returns `other`.

        This operation provides a fallback mechanism for the Maybe Monad. If the
        current instance contains a value (`Just`), the original value is preserved;
        if the current instance is `Nothing`, the `other` Monad is returned.

        Parameters
        ----------
        other : Maybe[TypeSource]
            The Maybe Monad to be returned if the current instance is `Nothing`.

        Returns
        -------
        result : Maybe[TypeSource]
            Returns the current Just instance if it contains a value,
            otherwise returns the `other` Maybe Monad.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> other: Maybe[int] = Just(2)
        >>> val.or_(other)
        Just(1)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.or_(other)
        Just(2)
        """
        return self

    def zip(self, other: Maybe[TypePure]) -> Maybe[Tuple[TypeSource, TypePure]]:
        """Combines this Maybe Monad with another into a Maybe Monad of a tuple.

        The `zip` method combines the values of two `Maybe` instances. It returns a `Just`
        containing a tuple of both values only if both instances contain values. If either
        instance is `Nothing`, the result is `Nothing`.

        Parameters
        ----------
        other : Maybe[TypePure]
            The Maybe monad to be zipped with the current instance.

        Returns
        -------
        result : Maybe[Tuple[TypeSource, TypePure]]
            A Maybe monad containing a tuple of the two values if both are present,
            otherwise returns `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> other: Maybe[int] = Just(2)
        >>> val.zip(other)
        Just((1, 2))
        >>> empty: Maybe[int] = Nothing()
        >>> val.zip(empty)
        Nothing()
        >>> empty_other: Maybe[int] = Nothing()
        >>> empty.zip(empty_other)
        Nothing()
        """
        return other.map(lambda o: (self._value, o))

    @overload
    def flatten(self: Just[Just[TypeResult]]) -> Maybe[TypeResult]: ...

    @overload
    def flatten(self: Just[Nothing[TypeResult]]) -> Maybe[TypeResult]: ...

    def flatten(self) -> Maybe[Any]:
        """Flattens a nested Maybe Monad by one level.

        This method is used to "unwrap" a Monad that contains another Monad.
        In functional programming, this is often referred to as `flatten` or `join`.
        It ensures that nested structures like `Just(Just(x))` are reduced to
        `Just(x)`.

        Returns
        -------
        result: Maybe[TypeResult]
            The inner Maybe Monad extracted from the outer container.

        Examples
        --------
        >>> val: Maybe[Maybe[int]] = Just(Just(1))
        >>> val.flatten()
        Just(1)
        >>> empty_inner: Maybe[Maybe[int]] = Just(Nothing())
        >>> empty_inner.flatten()
        Nothing()
        """
        return cast(Maybe[Any], self._value)

    def right_or(self, left_value: TypeLeft) -> Either[TypeLeft, TypeSource]:
        """
        Converts the Maybe Monad into an Either Monad, using `left_value` as the
        Left value if the current monad is `Nothing`.

        This method bridges the gap between optionality and error-handling. If the
        current instance is `Just`, it wraps the internal value in a `Right`.
        If the instance is `Nothing`, it returns a `Left` containing the
        provided `left_value`.

        Parameters
        ----------
        left_value : TypeLeft
            The value to be encapsulated in a `Left` instance if the
            Maybe Monad is `Nothing`.

        Returns
        -------
        either : Either[TypeLeft, TypeSource]
            An `Either` monad where the `Right` side contains the value if
            the current monad is `Just`, and the `Left` side contains `left_value`
            if the current monad is `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.right_or("Error")
        Right(1)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.right_or("Error")
        Left(Error)
        """
        return Right(self._value)

    def right_or_else(
        self, left_function: Callable[[], TypeLeft]
    ) -> Either[TypeLeft, TypeSource]:
        """Convert the Maybe Monad into an Either Monad using a generator function for the Left value.

        This method bridges the gap between optionality and error handling by allowing the
        caller to provide a computation that generates a Left value if the current
        Maybe instance is `Nothing`. If the instance is `Just`, it returns a
        `Right` containing the internal value.

        Parameters
        ----------
        left_function : Callable[[], TypeLeft]
            A function that produces a value of type `TypeLeft` to be returned
            when the Maybe Monad is `Nothing`.

        Returns
        -------
        either : Either[TypeLeft, TypeSource]
            An `Either` monad where the `Right` side contains the value if the
            current monad is `Just`, and the `Left` side contains the result of
            `left_function()` if the current monad is `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.right_or_else(lambda: "Error")
        Right(1)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.right_or_else(lambda: "Error")
        Left(Error)
        """
        return Right(self._value)

    def unwrap(self) -> TypeSource:
        """
        Extracts and returns the internal value from the Just monad.

        If the current instance is a `Just` monad, it returns the encapsulated
        value. If the instance is `Nothing`, this method will result in an
        unexpected behavior (accessing an internal private member), as `unwrap`
        is intended to be used only when the presence of a value is guaranteed.

        Returns
        -------
        value: TypeSource
            The value contained within the Just monad.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.unwrap()
        1
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Unwrap error on Maybe monad
        """
        return self._value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """
        Extracts the value from the Just monad, providing a fallback value if it is Nothing.

        This method provides a safe way to access the underlying value of the Maybe
        monad. If the monad contains a value (Just), the value is returned.
        If the monad is Nothing, the provided `default_value` is returned instead.

        Parameters
        ----------
        default_value : TypeSource
            The value to return if the current Maybe instance is `Nothing`.

        Returns
        -------
        TypeSource
            The value contained within the Just monad, or the `default_value`
            if the monad is `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.unwrap_or(0)
        1
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap_or(0)
        0
        """
        return self._value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """Returns the internal value of the Just or a default value generated by a function if the Monad is Nothing.

        This method allows for the extraction of a value from a `Just` container. If the container is `Nothing`,
        instead of raising an error or returning a static default, it executes a provided callback function
        to generate a fallback value.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            A callable that produces a value of type `TypeSource` to be returned
            if the current Maybe instance is `Nothing`.

        Returns
        -------
        value: TypeSource
            The value contained within the Just monad, or the result of calling
            `nothing_function` if the monad is Nothing.

        Examples
        --------
        >>> val: Maybe[int] = Just(1)
        >>> val.unwrap_or_else(lambda: 0)
        1
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap_or_else(lambda: 0)
        0
        """
        return self._value

    def inspect(self, function: Callable[[TypeSource], None]) -> Maybe[TypeSource]:
        """Perform an inspection or side-effect on the value inside the Just monad.

        This method executes a provided function (such as printing or logging) on the
        internal value if the monad is `Just`. If the monad is `Nothing`, the
        function is skipped, and the `Nothing` instance is returned. This is
        useful for debugging or triggering side-effects without consuming the monad.

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            A function that accepts the value inside the `Just` monad and
            performs an operation (e.g., printing).

        Returns
        -------
        maybe: Maybe[TypeSource]
            Returns the current Maybe instance, allowing for method chaining.

        Examples
        --------
        >>> val: Maybe[int] = Just(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Just(42)
        >>> empty: Maybe[int] = Nothing()
        >>> empty.inspect(lambda x: print(f"Value is: {x}"))
        Nothing()
        """
        function(self._value)
        return self

    def match(
        self,
        just_function: Callable[[TypeSource], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """
        Execute a function based on whether the Monad contains a value.

        This method provides a clean, functional way to branch logic. If the
        monad is `Just`, the `just_function` is applied to the internal value.
        If the monad is `Nothing`, the `nothing_function` is executed.

        Parameters
        ----------
        just_function: Callable[[TypeSource], TypeResult]
            A function to execute if the monad contains a value.
        nothing_function: Callable[[], TypeResult]
            A function to execute if the monad is `Nothing`.

        Returns
        -------
        result: TypeResult
            The result of the executed function.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.match(lambda x: x * 2, lambda: 0)
        20
        >>> empty: Maybe[int] = Nothing()
        >>> empty.match(lambda x: x * 2, lambda: 0)
        0
        """
        return just_function(self._value)

    def is_nothing(self) -> bool:
        """
        Check if the Maybe Monad is Nothing.

        Returns
        -------
        result: bool
            Returns False, since this instance is a `Just` monad.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.is_nothing()
        False
        """
        return False

    def is_just(self) -> bool:
        """
        Check if the Maybe Monad is Just.

        Returns
        -------
        result: bool
            Returns True, since this instance is a `Just` monad.

        Examples
        --------
        >>> val: Maybe[int] = Just(5)
        >>> val.is_just()
        True
        """
        return True

    def to_optional(self) -> TypeSource | None:
        """
        Convert the Maybe Monad to a standard Python optional value.

        This bridges the gap between the functional `Maybe` type and standard
        Python `None` handling.

        Returns
        -------
        value: TypeSource | None
            The value contained within the `Just` monad, or `None` if `Nothing`.

        Examples
        --------
        >>> val: Maybe[int] = Just(10)
        >>> val.to_optional()
        10
        >>> empty: Maybe[int] = Nothing()
        >>> print(empty.to_optional())
        None
        """
        return self._value

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        """
        Create a Maybe Monad from a standard Python optional value.

        Parameters
        ----------
        value: TypeSource | None
            The optional value to convert.

        Returns
        -------
        maybe: Maybe[TypeSource]
            A `Just` monad if the value is not `None`, otherwise a `Nothing` monad.

        Examples
        --------
        >>> Just.from_optional(5)
        Just(5)
        >>> Just.from_optional(None)
        Nothing()
        """
        return from_optional(value)

    def __str__(self) -> str:
        """
        Return the string representation of the Just Monad.

        Returns
        -------
        str
            A string representation of the internal value wrapped in `Just`.

        Examples
        --------
        >>> str(Just(42))
        'Just(42)'
        """
        return f"Just({self._value})"

    def __eq__(self, __o: object) -> bool:
        """
        Compare the Just Monad to another object for equality.

        Returns
        -------
        bool
            True if the other object is a `Just` monad with the same
            string representation.

        Examples
        --------
        >>> Just(1) == Just(1)
        True
        >>> Just(1) == Just(2)
        False
        """
        return isinstance(__o, Just) and str(self) == str(__o)

    def __repr__(self) -> str:
        """
        Return the official string representation of the Just Monad.

        Returns
        -------
        str
            The same as `__str__`.

        Examples
        --------
        >>> repr(Just(10))
        'Just(10)'
        """
        return str(self)


class Nothing(Generic[TypeSource]):
    """
    The Nothing variant of the Maybe Monad.

    The `Nothing` class represents the absence of a value. It is a phantom type,
    meaning it is generic over `TypeSource` even though it does not store any
    internal data. This ensures that operations like `map`, `bind`, and `apply`
    properly propagate the expected types instead of collapsing to `Any`.

    Examples
    -------
    >>> empty = Nothing()
    >>> print(type(empty))
    <class 'pymoliath.maybe.Nothing'>
    """

    __slots__ = ()

    _instance: ClassVar[Nothing[Any] | None] = None

    def __new__(cls) -> Nothing[TypeSource]:
        """
        Returns the single shared Nothing instance, creating it on first call.

        Since `Nothing` holds no data and is conceptually equivalent to any other
        `Nothing` of the same type, this implementation uses a singleton pattern
        to optimize memory and performance by sharing a single instance.

        Returns
        -------
        Nothing[TypeSource]
            The singleton instance of the Nothing monad.

        Examples
        -------
        >>> n1 = Nothing()
        >>> n2 = Nothing()
        >>> n1 is n2
        True
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Nothing[TypeSource], cls._instance)

    def map(self, function: Callable[[Any], TypeResult]) -> Maybe[TypeResult]:
        """Apply a function to the value inside the Nothing container.

        The `map` method is the standard Functor interface for the Maybe monad.
        Since this instance is `Nothing`, the function is not applied, and the
        operation returns a new `Nothing` instance while preserving the type
        structure of the result.

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            A function that accepts a value and returns a value of type `TypeResult`.

        Returns
        -------
        maybe: Maybe[TypeResult]
            A `Nothing` instance of type `TypeResult`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map(lambda x: x + 1)
        Nothing()
        >>> empty.map(str.upper)
        Nothing()
        """
        return Nothing()

    def bind(self, function: Callable[[Any], Maybe[TypeResult]]) -> Maybe[TypeResult]:
        """Maybe Monad bind interface (>>=, bind, flatMap).

        The `bind` method allows for chaining operations where the transformation
        function itself returns a `Maybe` type. Since this instance is `Nothing`,
        the function is skipped, and the result is automatically returned as `Nothing`.

        Parameters
        ----------
        function: Callable[[Any], Maybe[TypeResult]]
            A function that accepts a value and returns a `Maybe` monad of type
            `TypeResult`.

        Returns
        -------
        maybe: Maybe[TypeResult]
            A `Nothing` instance of type `TypeResult`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> def get_next_maybe(x: int) -> Maybe[int]:
        ...     return Just(x + 1) if x < 10 else Nothing()
        >>> empty.bind(get_next_maybe)
        Nothing()
        """
        return Nothing()

    def apply(self, applicative: Maybe[Callable[..., TypeResult]]) -> Maybe[TypeResult]:
        """
        Maybe Monad applicative interface for Maybe Monads containing a value (<*>).

        The `apply` method takes a `Maybe` containing a function and applies it
        to the value contained within this `Nothing` monad. Since the current
        instance is `Nothing`, the result is `Nothing`.

        Parameters
        ----------
        applicative: Maybe[Callable[..., TypeResult]]
            An applicative maybe monad containing a callable function.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> func: Maybe[Callable[[int], int]] = Just(lambda x: x * 2)
        >>> empty.apply(func)
        Nothing()
        """
        return Nothing()

    def apply2(self, applicative_value: Maybe[Any]) -> Maybe[Any]:
        """
        Maybe Monad applicative interface for Maybe Monads containing a function (<*>).

        The `apply2` method allows for applying a function contained within a
        `Nothing` monad to a value contained within another `Maybe` monad.
        Since the current instance is `Nothing`, the result is `Nothing`.

        Parameters
        ----------
        applicative_value: Maybe[Any]
            The Maybe monad containing the value to which the function will be applied.

        Returns
        -------
        maybe: Maybe[Any]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> val: Maybe[int] = Just(5)
        >>> empty.apply2(val)
        Nothing()
        """
        return self

    def filter(self, filter_function: Callable[[Any], bool]) -> Maybe[TypeSource]:
        """
        Filters the contained value based on a predicate.

        Since the current instance is `Nothing`, the `filter_function` is
        not applied, and the operation returns `Nothing`.

        Parameters
        ----------
        filter_function: Callable[[Any], bool]
            A predicate function that accepts the value inside the `Nothing`
            monad and returns a boolean.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.filter(lambda x: x > 5)
        Nothing()
        """
        return self

    def is_just_and(self, function: Callable[[Any], bool]) -> bool:
        """
        Checks if the Maybe Monad is a Just and satisfies a given predicate.

        Since the current instance is `Nothing`, the predicate is not evaluated,
        and the method returns `False`.

        Parameters
        ----------
        function: Callable[[Any], bool]
            A predicate function that accepts the value inside the `Nothing`
            monad and returns a boolean result.

        Returns
        -------
        result: bool
            Returns `False`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.is_just_and(lambda x: x > 5)
        False
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """
        Apply a transformation function to the Just value, providing a fallback value if Nothing.

        Since the current instance is `Nothing`, the transformation function is
        skipped and the `default_value` is returned.

        Parameters
        ----------
        default_value: TypeResult
            The fallback value to be returned if the current Maybe instance is Nothing.
        function: Callable[[Any], TypeResult]
            A transformation function that accepts the value inside the `Nothing`
            monad and returns a value of type `TypeResult`.

        Returns
        -------
        result: TypeResult
            Returns `default_value`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.map_or(0, lambda x: x * 2)
        0
        """
        return default_value

    def and_(self, other: Maybe[Any]) -> Maybe[TypeSource]:
        """
        Returns `other` if the Maybe Monad is a Just, otherwise Nothing.

        Since the current instance is `Nothing`, the operation short-circuits
        and returns `Nothing`.

        Parameters
        ----------
        other: Maybe[Any]
            The Maybe monad to be returned if the current instance is `Just`.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> other: Maybe[int] = Just(2)
        >>> empty.and_(other)
        Nothing()
        """
        return self

    def or_(self, other: Maybe[TypeSource]) -> Maybe[TypeSource]:
        """
        Returns this Maybe Monad if it is a Just, otherwise returns `other`.

        Since the current instance is `Nothing`, the `other` monad is returned.

        Parameters
        ----------
        other: Maybe[TypeSource]
            The Maybe Monad to be returned if the current instance is `Nothing`.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns `other`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> other: Maybe[int] = Just(2)
        >>> empty.or_(other)
        Just(2)
        """
        return other

    def zip(self, other: Maybe[Any]) -> Maybe[Any]:
        """
        Combines this Maybe Monad with another into a Maybe Monad of a tuple.

        Since the current instance is `Nothing`, the result is `Nothing`.

        Parameters
        ----------
        other: Maybe[Any]
            The Maybe monad to be zipped with the current instance.

        Returns
        -------
        result: Maybe[Any]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> other: Maybe[int] = Just(2)
        >>> empty.zip(other)
        Nothing()
        """
        return self

    def flatten(self) -> Maybe[TypeSource]:
        """
        Flattens a nested Maybe Monad by one level.

        Since the current instance is `Nothing`, the result is `Nothing`.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[Maybe[int]] = Nothing()
        >>> empty.flatten()
        Nothing()
        """
        return self

    def right_or(self, left_value: TypeLeft) -> Either[TypeLeft, Any]:
        """
        Converts the Maybe Monad into an Either Monad, using `left_value` as the
        Left value if the current monad is `Nothing`.

        Since the current instance is `Nothing`, it returns a `Left` containing
        the provided `left_value`.

        Parameters
        ----------
        left_value: TypeLeft
            The value to be encapsulated in a `Left` instance if the
            Maybe Monad is `Nothing`.

        Returns
        -------
        either: Either[TypeLeft, Any]
            Returns `Left` with `left_value`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.right_or("Error")
        Left(Error)
        """
        from pymoliath.either import Left

        return Left(left_value)

    def right_or_else(
        self, left_function: Callable[[], TypeLeft]
    ) -> Either[TypeLeft, Any]:
        """
        Converts the Maybe Monad into an Either Monad using a generator function for the
        Left value if the current Maybe instance is `Nothing`.

        Since the current instance is `Nothing`, it returns a `Left` containing
        the result of `left_function()`.

        Parameters
        ----------
        left_function: Callable[[], TypeLeft]
            A function that produces a value of type `TypeLeft` to be returned
            when the Maybe Monad is `Nothing`.

        Returns
        -------
        either: Either[TypeLeft, Any]
            Returns `Left` with the result of `left_function()`, since this
            Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.right_or_else(lambda: "Error")
        Left(Error)
        """
        from pymoliath.either import Left

        return Left(left_function())

    def unwrap(self) -> TypeSource:
        """
        Extracts and returns the internal value from the Just monad.

        Since the current instance is `Nothing`, this method will result in an
        Exception being raised.

        Returns
        -------
        value: TypeSource
            Never returns; always raises an Exception.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Unwrap error on Maybe monad
        """
        raise Exception("Unwrap error on Maybe monad")

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """
        Extracts the value from the Just monad, providing a fallback value if it is Nothing.

        Since the current instance is `Nothing`, the `default_value` is returned.

        Parameters
        ----------
        default_value: TypeSource
            The value to return if the current Maybe instance is `Nothing`.

        Returns
        -------
        value: TypeSource
            Returns `default_value`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap_or(0)
        0
        """
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """
        Returns the internal value of the Just or a default value generated by a function
        if the Monad is Nothing.

        Since the current instance is `Nothing`, the result of calling
        `nothing_function` is returned.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            A callable that produces a value of type `TypeSource` to be returned
            if the current Maybe instance is `Nothing`.

        Returns
        -------
        value: TypeSource
            Returns the result of calling `nothing_function`, since this
            Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.unwrap_or_else(lambda: 0)
        0
        """
        return nothing_function()

    def inspect(self, function: Callable[[Any], None]) -> Maybe[TypeSource]:
        """
        Perform an inspection or side-effect on the value inside the Just monad.

        Since the current instance is `Nothing`, the provided function is skipped,
        and the `Nothing` instance is returned.

        Parameters
        ----------
        function: Callable[[Any], None]
            A function that accepts the value inside the `Nothing`
            monad and performs an operation (e.g., printing).

        Returns
        -------
        maybe: Maybe[TypeSource]
            Returns this `Nothing`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.inspect(lambda x: print(f"Value is: {x}"))
        Nothing()
        """
        return self

    def match(
        self,
        just_function: Callable[[Any], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """
        Execute a function based on whether the Monad contains a value.

        Since the current instance is `Nothing`, the `nothing_function` is
        executed and its result is returned.

        Parameters
        ----------
        just_function: Callable[[Any], TypeResult]
            A function to execute if the monad contains a value.
        nothing_function: Callable[[], TypeResult]
            A function to execute if the monad is `Nothing`.

        Returns
        -------
        result: TypeResult
            Returns the result of calling `nothing_function`, since this
            Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.match(lambda x: x * 2, lambda: 0)
        0
        """
        return nothing_function()

    def is_nothing(self) -> bool:
        """
        Check if the Maybe Monad is Nothing.

        Returns
        -------
        result: bool
            Returns `True`, since this instance is a `Nothing` monad.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.is_nothing()
        True
        """
        return True

    def is_just(self) -> bool:
        """
        Check if the Maybe Monad is Just.

        Returns
        -------
        result: bool
            Returns `False`, since this instance is a `Nothing` monad.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> empty.is_just()
        False
        """
        return False

    def to_optional(self) -> TypeSource | None:
        """
        Convert the Maybe Monad to a standard Python optional value.

        Returns
        -------
        value: TypeSource | None
            Returns `None`, since this Maybe Monad is `Nothing`.

        Examples
        --------
        >>> empty: Maybe[int] = Nothing()
        >>> print(empty.to_optional())
        None
        """
        return None

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        """
        Create a Maybe Monad from a standard Python optional value.

        Parameters
        ----------
        value: TypeSource | None
            The optional value to convert.

        Returns
        -------
        maybe: Maybe[TypeSource]
            A `Just` monad if the value is not `None`, otherwise a `Nothing`
            monad.

        Examples
        --------
        >>> Nothing.from_optional(5)
        Just(5)
        >>> Nothing.from_optional(None)
        Nothing()
        """
        return from_optional(value)

    def __str__(self) -> str:
        """
        Return the string representation of the Nothing Monad.

        Returns
        -------
        str
            The string representation of the Nothing monad.

        Examples
        --------
        >>> str(Nothing())
        'Nothing()'
        """
        return "Nothing()"

    def __eq__(self, __o: object) -> bool:
        """
        Compare the Nothing Monad to another object for equality.

        Returns
        -------
        bool
            True if the other object is also a `Nothing` monad.

        Examples
        --------
        >>> Nothing() == Nothing()
        True
        >>> Nothing() == Just(1)
        False
        """
        return isinstance(__o, Nothing)

    def __repr__(self) -> str:
        """
        Return the official string representation of the Nothing Monad.

        Returns
        -------
        str
            The same as `__str__`.

        Examples
        --------
        >>> repr(Nothing())
        'Nothing()'
        """
        return str(self)


Maybe: TypeAlias = Just[TypeSource] | Nothing[TypeSource]


def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
    """Converts a standard Python optional value into a Maybe Monad.

    This helper utility allows for seamless integration between standard Python
    `None` handling and the `Maybe` functional pattern. It wraps a value
    in a `Just` container if it exists, or returns `Nothing` if the
    input is `None`.

    Parameters
    ----------
    value: TypeSource | None
        The optional value to be converted into a Maybe Monad.

    Returns
    -------
    maybe: Maybe[TypeSource]
        A `Just` instance containing the value, or a `Nothing` instance
        if the input was `None`.

    Examples
    --------
    >>> from_optional("hello")
    Just(hello)
    >>> from_optional(None)
    Nothing()
    """
    if value is None:
        return Nothing()
    return Just(value)


def safe(function: Callable[[], TypeResult]) -> Maybe[TypeResult]:
    """Executes a potentially unsafe function and captures its result in a Maybe Monad.

    This utility is used to encapsulate "dangerous" operations that might
    raise exceptions (like network requests, file I/O, or parsing). Instead
    of allowing the exception to crash the program, it catches the error
    and returns `Nothing`, allowing for safer, more declarative error handling.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        A callable that may raise an exception during execution.

    Returns
    -------
    maybe: Maybe[TypeResult]
        A `Just` instance containing the result of the function call if
        it succeeds, otherwise a `Nothing` instance if an exception is caught.

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> safe(risky_call)
    Nothing()
    >>> def safe_call():
    ...     return 42
    >>> safe(safe_call)
    Just(42)
    """
    try:
        return Just(function())
    except Exception:
        return Nothing()
