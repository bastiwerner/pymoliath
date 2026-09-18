from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, TypeAlias, TypeVar

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Just(Generic[TypeSource]):
    """Just Maybe Monad class

    Parameters
    ----------
    value: TypeSource
        Value to be stored in the Just Maybe Monad.
    """

    __match_args__ = ("_value",)

    def __init__(self, value: TypeSource):
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
            def inner(x: Any) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

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
            def inner(x: Any) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

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
            return Nothing()
        return Just(self._value)

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
        return False

    def is_just(self) -> bool:
        return True

    def to_optional(self) -> TypeSource | None:
        return self._value

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        return from_optional(value)

    def __str__(self) -> str:
        return f"Just({self._value})"

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def __repr__(self) -> str:
        return str(self)


class Nothing(Generic[TypeSource]):
    """Nothing Maybe Monad class

    Generic over TypeSource even though it stores no value: a phantom type
    parameter (like Rust's Option<T>::None) that lets map/bind/apply/filter/
    inspect propagate real types instead of collapsing to Any.
    """

    def map(self, function: Callable[[Any], TypeResult]) -> Maybe[TypeResult]:
        return Nothing()

    def bind(self, function: Callable[[Any], Maybe[TypeResult]]) -> Maybe[TypeResult]:
        return Nothing()

    def apply(self, applicative: Maybe[Callable[..., TypeResult]]) -> Maybe[TypeResult]:
        return Nothing()

    def apply2(self, applicative_value: Maybe[Any]) -> Maybe[Any]:
        return self

    def filter(self, filter_function: Callable[[Any], bool]) -> Maybe[TypeSource]:
        return self

    def unwrap(self) -> TypeSource:
        raise Exception("Unwrap error on Maybe monad")

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        return default_value

    def unwrap_or_else(self, nothing_function: Callable[[], TypeSource]) -> TypeSource:
        return nothing_function()

    def inspect(self, function: Callable[[Any], None]) -> Maybe[TypeSource]:
        return self

    def match(
        self,
        just_function: Callable[[Any], TypeResult],
        nothing_function: Callable[[], TypeResult],
    ) -> TypeResult:
        return nothing_function()

    def is_nothing(self) -> bool:
        return True

    def is_just(self) -> bool:
        return False

    def to_optional(self) -> TypeSource | None:
        return None

    @staticmethod
    def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
        return from_optional(value)

    def __str__(self) -> str:
        return "Nothing()"

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def __repr__(self) -> str:
        return str(self)


Maybe: TypeAlias = Just[TypeSource] | Nothing[TypeSource]


def from_optional(value: TypeSource | None) -> Maybe[TypeSource]:
    if value is None:
        return Nothing()
    return Just(value)


def safe(function: Callable[[], TypeResult]) -> Maybe[TypeResult]:
    try:
        return Just(function())
    except Exception:
        return Nothing()
