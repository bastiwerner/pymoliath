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
    from pymoliath.either import Either

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")
TypeLeft = TypeVar("TypeLeft")


class Just(Generic[TypeSource]):
    """Just Maybe Monad class

    Parameters
    ----------
    value: TypeSource
        Value to be stored in the Just Maybe Monad.
    """

    __match_args__ = ("_value",)

    def __init__(self, value: TypeSource):
        """Just Monad constructor which takes a value of type TypeSource.

        Parameters
        ----------
        value: TypeSource
            Value to be stored in the Just Monad.
        """
        self._value = value

    def map(self, function: Callable[[TypeSource], TypeResult]) -> Maybe[TypeResult]:
        """Maybe monad functor interface (>=, map).

        Definition: M(a) >= f: a -> b => M(b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns a Maybe Monad with the function result if Monad is a Just or otherwise a Nothing.
        """
        return Just(function(self._value))

    def bind(
        self, function: Callable[[TypeSource], Maybe[TypeResult]]
    ) -> Maybe[TypeResult]:
        """Maybe Monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], Maybe[TypeResult]]
            Function which takes a value of type TypeSource and returns a maybe monad of type TypeResult.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns a Maybe Monad with the function result if the Monad is a Just or otherwise a Nothing
        """
        return function(self._value)

    def apply(self, applicative: Maybe[Callable[..., TypeResult]]) -> Maybe[TypeResult]:
        """Maybe Monad applicative interface for Maybe Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Maybe[TypeApplicative] (TypeApplicative: any callable type)
            Applicative maybe monad which contains a function and will be applied to the maybe monad containing
            a value.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Applies a maybe monad containing a value of type TypeSource to a maybe monad containing a function.
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

        Parameters
        ----------
        applicative_value: Maybe[TypePure]
            Maybe monad value which will be applied to the maybe monad containing a function

        Returns
        -------
        maybe: Maybe[TypeResult]
            Applies a maybe monad containing a function to a maybe monad of type TypeSource (value or function).
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
        """Returns a Just if filter function is True and Maybe Monad is of type Just, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            Filter function which will be applied to Just if not Nothing.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns Just if the Maybe Monad is of type Just and filter function returns True otherwise Nothing.
        """
        if filter_function(self._value):
            return self
        return Nothing()

    def is_just_and(self, function: Callable[[TypeSource], bool]) -> bool:
        """Returns True if the Maybe Monad is a Just and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeSource], bool]
            Predicate function applied to the Just value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeSource], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Just value, or returns the default value if Nothing.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Maybe Monad is Nothing.
        function: Callable[[TypeSource], TypeResult]
            Function applied to the Just value.

        Returns
        -------
        result: TypeResult
            Returns the function result.
        """
        return function(self._value)

    def and_(self, other: Maybe[TypeResult]) -> Maybe[TypeResult]:
        """Returns `other` if the Maybe Monad is a Just, otherwise Nothing.

        Parameters
        ----------
        other: Maybe[TypeResult]
            Maybe Monad to be returned if this Maybe Monad is a Just.

        Returns
        -------
        result: Maybe[TypeResult]
            Returns `other`.
        """
        return other

    def or_(self, other: Maybe[TypeSource]) -> Maybe[TypeSource]:
        """Returns this Maybe Monad if it is a Just, otherwise `other`.

        Parameters
        ----------
        other: Maybe[TypeSource]
            Maybe Monad to be returned if this Maybe Monad is a Nothing.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this Just.
        """
        return self

    def zip(self, other: Maybe[TypePure]) -> Maybe[Tuple[TypeSource, TypePure]]:
        """Combines this Maybe Monad with another into a Maybe Monad of a tuple, or Nothing if either is Nothing.

        Parameters
        ----------
        other: Maybe[TypePure]
            Maybe Monad to be zipped with this Maybe Monad.

        Returns
        -------
        result: Maybe[Tuple[TypeSource, TypePure]]
            Returns Just of a tuple of both values, or Nothing.
        """
        return other.map(lambda o: (self._value, o))

    @overload
    def flatten(self: Just[Just[TypeResult]]) -> Maybe[TypeResult]: ...

    @overload
    def flatten(self: Just[Nothing[TypeResult]]) -> Maybe[TypeResult]: ...

    def flatten(self) -> Maybe[Any]:
        """Flattens a nested Maybe Monad by one level.

        Returns
        -------
        result: Maybe[TypeResult]
            Returns the nested Maybe Monad.
        """
        return cast(Maybe[Any], self._value)

    def right_or(self, left_value: TypeLeft) -> Either[TypeLeft, TypeSource]:
        """Converts the Maybe Monad into an Either Monad, using `left_value` as the Left value if Nothing.

        Parameters
        ----------
        left_value: TypeLeft
            Left value to be used if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[TypeLeft, TypeSource]
            Returns Right with the Just value.
        """
        from pymoliath.either import Right

        return Right(self._value)

    def right_or_else(
        self, left_function: Callable[[], TypeLeft]
    ) -> Either[TypeLeft, TypeSource]:
        """Converts the Maybe Monad into an Either Monad, calling `left_function` for the Left value if Nothing.

        Parameters
        ----------
        left_function: Callable[[], TypeLeft]
            Function called to produce the left value if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[TypeLeft, TypeSource]
            Returns Right with the Just value.
        """
        from pymoliath.either import Right

        return Right(self._value)

    def unwrap(self) -> TypeSource:
        """Returns the internal value of the Just or raises an exception if Nothing.

        Returns
        -------
        value: TypeSource
            Returns the Maybe value or a default value.
        """
        return self._value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the internal value of the Just or default value if the Monad is a Nothing.

        Parameters
        ----------
        default_value: TypeSource
            Default value to be returned if the Monad is a Nothing

        Returns
        -------
        value: TypeSource
            Returns the Maybe value or a default value.
        """
        return self._value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """Returns the internal value of the Just or default value if the Monad is a Nothing.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            Function to be called when the Maybe value is of type Nothing

        Returns
        -------
        value: TypeSource
            Returns the Maybe value or calls the nothing function.
        """
        return self._value

    def inspect(self, function: Callable[[TypeSource], None]) -> Maybe[TypeSource]:
        """Inspect the Maybe monad value of TypeSource

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            Inspection function which takes the maybe value if not nothing

        Returns
        -------
        maybe: Maybe[TypeSource]
        """
        function(self._value)
        return self

    def match(
        self,
        just_function: Callable[[TypeSource], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """The maybe function takes a function and a default value. If the Maybe value is Nothing, the function returns
        the default value. Otherwise, it applies the function to the value inside a Just monad and returns the result.

        Parameter
        ---------
        callback: Callable[[TypeSource], TypeSource]
          Callback function if the Maybe Monad is a Just
        default: TypeResult
          Default value to be returned if the Maybe Monad is Nothing

        Returns
        -------
        result: TypeSource
        """
        return just_function(self._value)

    def is_nothing(self) -> bool:
        """Returns False, since this Maybe Monad is a Just.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def is_just(self) -> bool:
        """Returns True, since this Maybe Monad is a Just.

        Returns
        -------
        result: bool
            Returns True.
        """
        return True

    def to_optional(self) -> TypeSource | None:
        """Converts the Maybe Monad into a standard Python optional value.

        Returns
        -------
        value: TypeSource | None
            Returns the Just value.
        """
        return self._value

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        """Converts a standard Python optional value into a Maybe Monad.

        Parameters
        ----------
        value: TypeSource | None
            Optional value to be converted into a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeSource]
            Returns Just if `value` is not None, otherwise Nothing.
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Just Monad."""
        return f"Just({self._value})"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `other` is also a Just Monad with an equal string representation."""
        return isinstance(__o, Just) and str(self) == str(__o)

    def __repr__(self) -> str:
        """Returns the string representation of the Just Monad (same as __str__)."""
        return str(self)


class Nothing(Generic[TypeSource]):
    """Nothing Maybe Monad class

    Generic over TypeSource even though it stores no value: a phantom type
    parameter (like Rust's Option<T>::None) that lets map/bind/apply/filter/
    inspect propagate real types instead of collapsing to Any.
    """

    def map(self, function: Callable[[Any], TypeResult]) -> Maybe[TypeResult]:
        """Maybe monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[Any], TypeResult]
            Function which would be applied to the value if this Maybe Monad were a Just.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns Nothing, since this Maybe Monad is Nothing.
        """
        return Nothing()

    def bind(self, function: Callable[[Any], Maybe[TypeResult]]) -> Maybe[TypeResult]:
        """Maybe Monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[Any], Maybe[TypeResult]]
            Function which would be applied to the value if this Maybe Monad were a Just.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns Nothing, since this Maybe Monad is Nothing.
        """
        return Nothing()

    def apply(self, applicative: Maybe[Callable[..., TypeResult]]) -> Maybe[TypeResult]:
        """Maybe Monad applicative interface for Maybe Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Maybe[TypeApplicative] (TypeApplicative: any callable type)
            Applicative maybe monad which would be applied to the maybe monad containing a value.

        Returns
        -------
        maybe: Maybe[TypeResult]
            Returns Nothing, since this Maybe Monad is Nothing.
        """
        return Nothing()

    def apply2(self, applicative_value: Maybe[Any]) -> Maybe[Any]:
        """Maybe Monad applicative interface for Maybe Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Maybe[TypePure]
            Maybe monad value which would be applied to the maybe monad containing a function.

        Returns
        -------
        maybe: Maybe[Any]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def filter(self, filter_function: Callable[[Any], bool]) -> Maybe[TypeSource]:
        """Returns a Just if filter function is True and Maybe Monad is of type Just, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[Any], bool]
            Filter function which would be applied to the value if this Maybe Monad were a Just.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def is_just_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns True if the Maybe Monad is a Just and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[Any], bool]
            Predicate function which would be applied to the value if this Maybe Monad were a Just.

        Returns
        -------
        result: bool
            Returns False, since this Maybe Monad is Nothing.
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Just value, or returns the default value if Nothing.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Maybe Monad is Nothing.
        function: Callable[[Any], TypeResult]
            Function which would be applied to the value if this Maybe Monad were a Just.

        Returns
        -------
        result: TypeResult
            Returns `default_value`, since this Maybe Monad is Nothing.
        """
        return default_value

    def and_(self, other: Maybe[Any]) -> Maybe[TypeSource]:
        """Returns `other` if the Maybe Monad is a Just, otherwise Nothing.

        Parameters
        ----------
        other: Maybe[Any]
            Maybe Monad which would be returned if this Maybe Monad were a Just.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def or_(self, other: Maybe[TypeSource]) -> Maybe[TypeSource]:
        """Returns this Maybe Monad if it is a Just, otherwise `other`.

        Parameters
        ----------
        other: Maybe[TypeSource]
            Maybe Monad to be returned since this Maybe Monad is a Nothing.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns `other`.
        """
        return other

    def zip(self, other: Maybe[Any]) -> Maybe[Any]:
        """Combines this Maybe Monad with another into a Maybe Monad of a tuple, or Nothing if either is Nothing.

        Parameters
        ----------
        other: Maybe[Any]
            Maybe Monad which would be zipped with this Maybe Monad if this Maybe Monad were a Just.

        Returns
        -------
        result: Maybe[Any]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def flatten(self) -> Maybe[TypeSource]:
        """Flattens a nested Maybe Monad by one level.

        Returns
        -------
        result: Maybe[TypeSource]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def right_or(self, left_value: TypeLeft) -> Either[TypeLeft, Any]:
        """Converts the Maybe Monad into an Either Monad, using `left_value` as the Left value if Nothing.

        Parameters
        ----------
        left_value: TypeLeft
            Left value to be used if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[TypeLeft, Any]
            Returns Left with `left_value`, since this Maybe Monad is Nothing.
        """
        from pymoliath.either import Left

        return Left(left_value)

    def right_or_else(self, left_function: Callable[[], TypeLeft]) -> Either[TypeLeft, Any]:
        """Converts the Maybe Monad into an Either Monad, calling `left_function` for the Left value if Nothing.

        Parameters
        ----------
        left_function: Callable[[], TypeLeft]
            Function called to produce the left value if the Maybe Monad is Nothing.

        Returns
        -------
        either: Either[TypeLeft, Any]
            Returns Left with the result of `left_function`, since this Maybe Monad is Nothing.
        """
        from pymoliath.either import Left

        return Left(left_function())

    def unwrap(self) -> TypeSource:
        """Raises an exception, since a Nothing Monad has no internal value to unwrap.

        Returns
        -------
        value: TypeSource
            Never returns; always raises an Exception.
        """
        raise Exception("Unwrap error on Maybe monad")

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        """Returns the internal value of the Just or default value if the Monad is a Nothing.

        Parameters
        ----------
        default_value: TypeSource
            Default value to be returned if the Monad is a Nothing

        Returns
        -------
        value: TypeSource
            Returns `default_value`, since this Maybe Monad is Nothing.
        """
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        """Returns the internal value of the Just or default value if the Monad is a Nothing.

        Parameters
        ----------
        nothing_function: Callable[[], TypeSource]
            Function to be called when the Maybe value is of type Nothing

        Returns
        -------
        value: TypeSource
            Returns the result of calling `nothing_function`, since this Maybe Monad is Nothing.
        """
        return nothing_function()

    def inspect(self, function: Callable[[Any], None]) -> Maybe[TypeSource]:
        """Inspect the Maybe monad value of TypeSource

        Parameters
        ----------
        function: Callable[[Any], None]
            Inspection function which would be called with the value if this Maybe Monad were a Just.

        Returns
        -------
        maybe: Maybe[TypeSource]
            Returns this Nothing, since this Maybe Monad is Nothing.
        """
        return self

    def match(
        self,
        just_function: Callable[[Any], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        """The maybe function takes a function and a default value. If the Maybe value is Nothing, the function returns
        the default value. Otherwise, it applies the function to the value inside a Just monad and returns the result.

        Parameter
        ---------
        callback: Callable[[TypeSource], TypeSource]
          Callback function if the Maybe Monad is a Just
        default: TypeResult
          Default value to be returned if the Maybe Monad is Nothing

        Returns
        -------
        result: TypeSource
            Returns the result of calling `nothing_function`, since this Maybe Monad is Nothing.
        """
        return nothing_function()

    def is_nothing(self) -> bool:
        """Returns True, since this Maybe Monad is a Nothing.

        Returns
        -------
        result: bool
            Returns True.
        """
        return True

    def is_just(self) -> bool:
        """Returns False, since this Maybe Monad is a Nothing.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def to_optional(self) -> TypeSource | None:
        """Converts the Maybe Monad into a standard Python optional value.

        Returns
        -------
        value: TypeSource | None
            Returns None, since this Maybe Monad is Nothing.
        """
        return None

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        """Converts a standard Python optional value into a Maybe Monad.

        Parameters
        ----------
        value: TypeSource | None
            Optional value to be converted into a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeSource]
            Returns Just if `value` is not None, otherwise Nothing.
        """
        return from_optional(value)

    def __str__(self) -> str:
        """Returns the string representation of the Nothing Monad."""
        return "Nothing()"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `other` is also a Nothing Monad."""
        return isinstance(__o, Nothing)

    def __repr__(self) -> str:
        """Returns the string representation of the Nothing Monad (same as __str__)."""
        return str(self)


Maybe: TypeAlias = Just[TypeSource] | Nothing[TypeSource]


def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
    """Converts a standard Python optional value into a Maybe Monad.

    Parameters
    ----------
    value: TypeSource | None
        Optional value to be converted into a Maybe Monad.

    Returns
    -------
    maybe: Maybe[TypeSource]
        Returns Just containing `value`, or Nothing if `value` is None.
    """
    if value is None:
        return Nothing()
    return Just(value)


def safe(function: Callable[[], TypeResult]) -> Maybe[TypeResult]:
    """Calls a function which might raise an Exception and returns Just with the result, otherwise Nothing.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        Callable function which may raise an exception.

    Returns
    -------
    maybe: Maybe[TypeResult]
        Returns Just containing the function result, or Nothing if an exception was raised.
    """
    try:
        return Just(function())
    except Exception:
        return Nothing()
