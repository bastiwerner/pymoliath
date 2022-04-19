from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Any, Optional, Type, TypeVar, Generic

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Maybe(Generic[TypeSource], abc.ABC):
    """Maybe monad abstract class type

    The Maybe type encapsulates an optional value. A value of type Maybe either contains a value of type a
    (represented as Just a), or it is empty (represented as Nothing). Using Maybe is a good way to deal with errors
    or exceptional cases without resorting to drastic measures such as error.

    The Maybe type is also a monad. It is a simple kind of error monad, where all errors are represented by Nothing.
    A richer error monad can be built using the Either type.

    TypeSource: any type which will be used for monad computation (bind, map, apply)

    Instances:
        * Maybe: Just, Nothing
    """
    _value: TypeSource  # Private maybe monad value which should not be modified
    _is_nothing: bool  # Private maybe monad is nothing flag which should not be modified

    def map(self: Maybe[TypeSource], function: Callable[[TypeSource], TypeResult]) -> Maybe[TypeResult]:
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
        if self._is_nothing:
            return Nothing()
        return Just(function(self._value))

    def bind(self: Maybe[TypeSource], function: Callable[[TypeSource], Maybe[TypeResult]]) -> Maybe[TypeResult]:
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
        if self._is_nothing:
            return Nothing()
        return function(self._value)

    def apply(self: Maybe[TypeSource], applicative: Maybe[Callable[[TypeSource], TypeResult]]) -> Maybe[TypeResult]:
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

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> Maybe[TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: Maybe[Callable[[TypePure], TypeResult]], applicative_value: Maybe[TypePure]) -> Maybe[TypeResult]:
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

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> Maybe[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def filter(self: Maybe[TypeSource], filter_function: Callable[[TypeSource], bool]) -> Maybe[TypeSource]:
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
        if self._is_nothing or filter_function(self._value):
            return Nothing()
        return Just(self._value)

    def unwrap(self: Maybe[TypeSource]) -> TypeSource:
        """Returns the internal value of the Just or raises an exception if Nothing.

        Returns
        -------
        value: TypeSource
            Returns the Maybe value or a default value.
        """
        if self._is_nothing:
            raise Exception("Unwrap error on Maybe monad")
        return self._value

    def unwrap_or(self: Maybe[TypeSource], default_value: TypeSource) -> TypeSource:
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
        if self._is_nothing:
            return default_value
        return self._value

    def unwrap_or_else(self: Maybe[TypeSource], nothing_function: Callable[[], TypeSource]) -> TypeSource:
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
        if self._is_nothing:
            return nothing_function()
        return self._value

    def match(self: Maybe[TypeSource],
              just_function: Callable[[TypeSource], TypeResult],
              nothing_function: Callable[[], TypeResult]) -> TypeResult:
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
        if self._is_nothing:
            return nothing_function()
        return just_function(self._value)

    def is_nothing(self):
        return self._is_nothing

    def is_just(self):
        return not self._is_nothing

    @classmethod
    def from_optional(cls: Type[Maybe[TypeSource]], value: Optional[TypeSource]) -> Maybe[TypeSource]:
        if value:
            return Just(value)
        return Nothing()

    @abc.abstractmethod
    def __str__(self: Maybe[TypeSource]) -> str:
        pass

    def __eq__(self: Maybe[TypeSource], __o: object) -> bool:
        return str(self) == str(__o)

    def __repr__(self: Maybe[TypeSource]) -> str:
        return str(self)


class Just(Maybe[TypeSource]):
    """Just Maybe Monad class

    Parameters
    ----------
    value: TypeSource
        Value to be stored in the Just Maybe Monad.
    """

    def __init__(self, value: TypeSource):
        self._value = value
        self._is_nothing = False

    def __str__(self: Just[TypeSource]) -> str:
        return f"Just({self._value})"


class Nothing(Maybe[Any]):
    """Nothing Maybe Monad class
    """

    def __init__(self):
        self._value = Any
        self._is_nothing = True

    def __str__(self: Nothing) -> str:
        return "Nothing()"
