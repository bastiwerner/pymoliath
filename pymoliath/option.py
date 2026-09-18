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
    from pymoliath.result import Result

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")
TypeErr = TypeVar("TypeErr")


class Some(Generic[TypeSource]):
    """Some Option Monad class

    Parameters
    ----------
    value: TypeSource
        Value to be stored in the Some Option Monad.
    """

    __match_args__ = ("_value",)

    def __init__(self, value: TypeSource):
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
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Option[TypeResult]:
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
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Option[TypeResult]:
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
        """
        from pymoliath.result import Ok

        return Ok(self._value)

    def unwrap(self) -> TypeSource:
        """Returns the internal value of the Some or raises an exception if Nothing.

        Returns
        -------
        value: TypeSource
            Returns the Option value or a default value.
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
        """
        return some_function(self._value)

    def is_nothing(self) -> bool:
        return False

    def is_some(self) -> bool:
        return True

    def to_optional(self) -> TypeSource | None:
        return self._value

    @staticmethod
    def from_optional(value: TypeSource | None) -> Option[TypeSource]:
        return from_optional(value)

    def __str__(self) -> str:
        return f"Some({self._value})"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Some) and str(self) == str(__o)

    def __repr__(self) -> str:
        return str(self)


class Nil(Generic[TypeSource]):
    """Nothing Option Monad class

    Generic over TypeSource even though it stores no value: a phantom type
    parameter (like Rust's Option<T>::None) that lets map/bind/apply/filter/
    inspect propagate real types instead of collapsing to Any.
    """

    def map(self, function: Callable[[Any], TypeResult]) -> Option[TypeResult]:
        return Nil()

    def bind(self, function: Callable[[Any], Option[TypeResult]]) -> Option[TypeResult]:
        return Nil()

    def apply(
        self, applicative: Option[Callable[..., TypeResult]]
    ) -> Option[TypeResult]:
        return Nil()

    def apply2(self, applicative_value: Option[Any]) -> Option[Any]:
        return self

    def filter(self, filter_function: Callable[[Any], bool]) -> Option[TypeSource]:
        return self

    def is_some_and(self, function: Callable[[Any], bool]) -> bool:
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        return default_value

    def and_(self, other: Option[Any]) -> Option[TypeSource]:
        return self

    def or_(self, other: Option[TypeSource]) -> Option[TypeSource]:
        return other

    def zip(self, other: Option[Any]) -> Option[Any]:
        return self

    def flatten(self) -> Option[TypeSource]:
        return self

    def ok_or(self, err_value: TypeErr) -> Result[Any, TypeErr]:
        from pymoliath.result import Err

        return Err(err_value)

    def ok_or_else(self, err_function: Callable[[], TypeErr]) -> Result[Any, TypeErr]:
        from pymoliath.result import Err

        return Err(err_function())

    def unwrap(self) -> TypeSource:
        raise Exception("Unwrap error on Option monad")

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        return nothing_function()

    def inspect(self, function: Callable[[Any], None]) -> Option[TypeSource]:
        return self

    def match(
        self,
        some_function: Callable[[Any], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        return nothing_function()

    def is_nothing(self) -> bool:
        return True

    def is_some(self) -> bool:
        return False

    def to_optional(self) -> TypeSource | None:
        return None

    @staticmethod
    def from_optional(value: TypeSource | None) -> Option[TypeSource]:
        return from_optional(value)

    def __str__(self) -> str:
        return "Nil()"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Nil)

    def __repr__(self) -> str:
        return str(self)


Option: TypeAlias = Some[TypeSource] | Nil[TypeSource]


def from_optional(value: TypeSource | None) -> Option[TypeSource]:
    if value is None:
        return Nil()
    return Some(value)


def safe(function: Callable[[], TypeResult]) -> Option[TypeResult]:
    try:
        return Some(function())
    except Exception:
        return Nil()
