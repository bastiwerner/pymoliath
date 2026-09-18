from __future__ import annotations

from typing import Any, Callable, Generic, TypeAlias, TypeVar

from pymoliath.util import curry

TypeLeft = TypeVar("TypeLeft")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Right(Generic[TypeRight]):
    """The Either Monad represents values with two possibilities: either Left[A] or Right[B]."""

    __match_args__ = ("_right_value",)

    def __init__(self, value: TypeRight) -> None:
        self._right_value = value

    def map(
        self, function: Callable[[TypeRight], TypeResult]
    ) -> Either[Any, TypeResult]:
        """Calls function to the a wrapped Right value if not Left, otherwise leaving the Left value untouched.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which takes a value of TypeRight and returns a value of type TypeResult.

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
            Returns Right with the function result or otherwise Left.
        """
        return Right(function(self._right_value))

    def map_left(
        self, function: Callable[[Any], TypeResult]
    ) -> Either[TypeResult, TypeRight]:
        """Calls function to the a wrapped Left value if not Right, otherwise leaving the Right value untouched.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which takes a value of TypeLeft and returns a value of type TypeResult.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns Left with the function result or otherwise Right.
        """
        return self

    def bind(
        self, function: Callable[[TypeRight], Either[Any, TypeResult]]
    ) -> Either[Any, TypeResult]:
        """Calls function if Either Monad is Right, otherwise returns Left.

        Parameters
        ----------
        function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]
            Function which takes a value of TypeRight and returns a Result Monad of TypeResult.

        Returns
        -------
        either: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the function result if Right, otherwise Left.
        """
        return function(self._right_value)

    def bind_left(
        self, function: Callable[[Any], Either[TypeResult, TypeRight]]
    ) -> Either[TypeResult, TypeRight]:
        """Calls function if Either Monad is Left, otherwise returns Right.

        Parameters
        ----------
        function: Callable[[TypeLeft], Either[TypeResult, TypeRight]]
            Function which takes a value of TypeLeft and returns a Result Monad of TypeResult.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns a Either Monad from the function result if Left, otherwise Right.
        """
        return self

    def apply(
        self, applicative: Either[Any, Callable[..., TypeResult]]
    ) -> Either[Any, TypeResult]:
        """Applies the passed applicative wrapping a function if Either Monad is Right, otherwise returns Left.

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[Callable[[TypeSource], TypeResult]]
            Applicative Either Monad which contains a function.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the applied function if Right, otherwise Left.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Either[Any, TypeResult]:
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Right[Callable[..., TypeResult]],
        applicative_value: Either[Any, Any],
    ) -> Either[Any, TypeResult]:
        """Applies the passed Either Monad wrapping a value to the Either Monad containing a function if Right,
        otherwise returns Left.

        Parameters
        ----------
        applicative_value: Either[TypeLeft, TypePure]
            Result Monad which contains a value.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the applied function if Right, otherwise Left.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Either[Any, TypeResult]:
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def unwrap(self) -> TypeRight:
        """Returns the Right value if not Left, or otherwise raises an Exception containing the left value.

        Returns
        -------
        result: TypeRight
            Returns the Right value or raises an Exception.
        """
        return self._right_value

    def unwrap_or(self, default_value: TypeRight) -> TypeRight:
        """Returns the Right value if not Left, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeRight
            Default value of TypeRight

        Returns
        -------
        result: TypeRight
            Returns the Right value or a default value.
        """
        return self._right_value

    def unwrap_or_else(self, left_function: Callable[[Any], TypeRight]) -> TypeRight:
        """Returns the Right value if not Left, or otherwise a provided function which will be called with the left.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeRight]
            Called with the left value and must return a value of type TypeRight

        Returns
        -------
        result: TypeRight
            Returns the Right value or value from the function call.
        """
        return self._right_value

    def unwrap_left_or(self, default_value: TypeLeft) -> TypeLeft:
        """Returns the Left value if not Right, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeLeft
            Default value of TypeLeft

        Returns
        -------
        result: TypeLeft
            Returns the Left value or a default value.
        """
        return default_value

    def inspect(self, function: Callable[[TypeRight], None]) -> Either[Any, TypeRight]:
        """Inspect the Either value of TypeRight

        Parameters
        ----------
        function: Callable[[TypeRight], None]
            Inspection function which takes the right value of the Either monad

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
        """
        function(self._right_value)
        return self

    def inspect_left(self, function: Callable[[Any], None]) -> Either[Any, TypeRight]:
        """Inspect the Either value of TypeLeft

        Parameters
        ----------
        function: Callable[[TypeLeft], None]
            Inspection function which takes the left value of the Either monad

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
        """
        return self

    def match(
        self,
        left_function: Callable[[Any], TypeResult],
        right_function: Callable[[TypeRight], TypeResult],
    ) -> TypeResult:
        """Right monad specific function to handle railroad orientated programming.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], None]
            Callback function for either monads of type Left
        right_function: Callable[[TypeRight], None]
            Callback function for either monads of type Right
        """
        return right_function(self._right_value)

    def is_left(self) -> bool:
        """Either monad is left function

        Returns
        -------
        result: bool
            True: if either monad is of type left, False: if either monad is of type right
        """
        return False

    def is_right(self) -> bool:
        """Either monad is right function

        Returns
        -------
        result: bool
            True: if either monad is of type right, False: if either monad is of type left
        """
        return True

    def __str__(self) -> str:
        return f"Right({self._right_value})"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Right) and str(self) == str(__o)

    def __repr__(self) -> str:
        return str(self)


class Left(Generic[TypeLeft]):
    __match_args__ = ("_left_value",)

    def __init__(self, value: TypeLeft):
        self._left_value = value

    def map(
        self, function: Callable[[Any], TypeResult]
    ) -> Either[TypeLeft, TypeResult]:
        return self

    def map_left(
        self, function: Callable[[TypeLeft], TypeResult]
    ) -> Either[TypeResult, Any]:
        return Left(function(self._left_value))

    def bind(
        self, function: Callable[[Any], Either[TypeLeft, TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        return self

    def bind_left(
        self, function: Callable[[TypeLeft], Either[TypeResult, Any]]
    ) -> Either[TypeResult, Any]:
        return function(self._left_value)

    def apply(
        self, applicative: Either[TypeLeft, Callable[..., TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        return self

    def apply2(self, applicative_value: Either[TypeLeft, Any]) -> Either[TypeLeft, Any]:
        return self

    def unwrap(self) -> Any:
        raise Exception(self._left_value)

    def unwrap_or(self, default_value: TypeRight) -> TypeRight:
        return default_value

    def unwrap_or_else(
        self, left_function: Callable[[TypeLeft], TypeRight]
    ) -> TypeRight:
        return left_function(self._left_value)

    def unwrap_left_or(self, default_value: TypeLeft) -> TypeLeft:
        return self._left_value

    def inspect(self, function: Callable[[Any], None]) -> Either[TypeLeft, Any]:
        return self

    def inspect_left(
        self, function: Callable[[TypeLeft], None]
    ) -> Either[TypeLeft, Any]:
        function(self._left_value)
        return self

    def match(
        self,
        left_function: Callable[[TypeLeft], TypeResult],
        right_function: Callable[[Any], TypeResult],
    ) -> TypeResult:
        return left_function(self._left_value)

    def is_left(self) -> bool:
        return True

    def is_right(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"Left({self._left_value})"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Left) and str(self) == str(__o)

    def __repr__(self) -> str:
        return str(self)


Either: TypeAlias = Left[TypeLeft] | Right[TypeRight]


def either_safe(function: Callable[[], TypeResult]) -> Either[Exception, TypeResult]:
    """Calls an unsafe function which might raise an Exception and returns Right with the result, otherwise Left
    containing the Exception.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        Callable function which may raise an exception

    Returns
    -------
    either: Either[Exception, TypeResult]
        Returns an Either Monad which contains either the function result or an Exception with a message added.
    """
    try:
        return Right(function())
    except Exception as e:
        return Left(e)
