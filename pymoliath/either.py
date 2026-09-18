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
    """The Either Monad represents values with two possibilities: either Left[A] or Right[B]."""

    __match_args__ = ("_right_value",)

    def __init__(self, value: TypeRight) -> None:
        """Right Monad constructor which takes a value of type TypeRight.

        Parameters
        ----------
        value: TypeRight
            Value to be stored in the Right Monad.
        """
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
            """Maps the applicative's function, curried, over this Right value."""
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
            """Maps the applicative's function, curried, over the applicative_value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def is_right_and(self, function: Callable[[TypeRight], bool]) -> bool:
        """Returns True if the Either Monad is Right and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeRight], bool]
            Predicate function applied to the Right value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._right_value)

    def is_left_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Either Monad is Right.

        Parameters
        ----------
        function: Callable[[TypeLeft], bool]
            Predicate function which would be applied to the Left value.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeRight], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Right value, or returns the default value if Left.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Either Monad is Left.
        function: Callable[[TypeRight], TypeResult]
            Function applied to the Right value.

        Returns
        -------
        result: TypeResult
            Returns the function result.
        """
        return function(self._right_value)

    def and_(self, other: Either[Any, TypeResult]) -> Either[Any, TypeResult]:
        """Returns `other` if the Either Monad is Right, otherwise Left.

        Parameters
        ----------
        other: Either[TypeLeft, TypeResult]
            Either Monad to be returned if this Either Monad is Right.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns `other`.
        """
        return other

    def or_(self, other: Either[Any, TypeRight]) -> Either[Any, TypeRight]:
        """Returns this Either Monad if it is Right, otherwise `other`.

        Parameters
        ----------
        other: Either[TypeLeft, TypeRight]
            Either Monad to be returned if this Either Monad is Left.

        Returns
        -------
        result: Either[TypeLeft, TypeRight]
            Returns this Right.
        """
        return self

    def zip(
        self, other: Either[Any, TypePure]
    ) -> Either[Any, Tuple[TypeRight, TypePure]]:
        """Combines this Either Monad with another into an Either Monad of a tuple, or Left if either is Left.

        Parameters
        ----------
        other: Either[TypeLeft, TypePure]
            Either Monad to be zipped with this Either Monad.

        Returns
        -------
        result: Either[TypeLeft, Tuple[TypeRight, TypePure]]
            Returns Right of a tuple of both values, or Left.
        """
        return other.map(lambda o: (self._right_value, o))

    @overload
    def flatten(self: Right[Right[TypeResult]]) -> Either[Any, TypeResult]: ...

    @overload
    def flatten(self: Right[Left[TypeLeft]]) -> Either[TypeLeft, Any]: ...

    def flatten(self) -> Either[Any, Any]:
        """Flattens a nested Either Monad by one level.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns the nested Either Monad.
        """
        return cast(Either[Any, Any], self._right_value)

    def right(self) -> Maybe[TypeRight]:
        """Either Monad specific function to return the Right value as a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeRight]
            Returns Just with the Right value.
        """
        from pymoliath.maybe import Just

        return Just(self._right_value)

    def left(self) -> Maybe[Any]:
        """Either Monad specific function to return the Left value as a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeLeft]
            Returns Nothing, since this Either Monad is Right.
        """
        from pymoliath.maybe import Nothing

        return Nothing()

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
        """Returns the string representation of the Right Monad."""
        return f"Right({self._right_value})"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `__o` is a Right Monad with an equal string representation."""
        return isinstance(__o, Right) and str(self) == str(__o)

    def __repr__(self) -> str:
        """Returns the string representation of the Right Monad (same as __str__)."""
        return str(self)


class Left(Generic[TypeLeft]):
    __match_args__ = ("_left_value",)

    def __init__(self, value: TypeLeft):
        """Left Monad constructor which takes a value of type TypeLeft.

        Parameters
        ----------
        value: TypeLeft
            Value to be stored in the Left Monad.
        """
        self._left_value = value

    def map(
        self, function: Callable[[Any], TypeResult]
    ) -> Either[TypeLeft, TypeResult]:
        """Returns this Left unchanged, since map only transforms the Right value.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which would be applied to the Right value if this Either Monad were Right.

        Returns
        -------
        either: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def map_left(
        self, function: Callable[[TypeLeft], TypeResult]
    ) -> Either[TypeResult, Any]:
        """Calls function on the wrapped Left value and returns a new Left with the result.

        Parameters
        ----------
        function: Callable[[TypeLeft], TypeResult]
            Function which takes a value of TypeLeft and returns a value of type TypeResult.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns a new Left with the function result.
        """
        return Left(function(self._left_value))

    def bind(
        self, function: Callable[[Any], Either[TypeLeft, TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        """Returns this Left unchanged, since bind only chains on the Right value.

        Parameters
        ----------
        function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]
            Function which would be called with the Right value if this Either Monad were Right.

        Returns
        -------
        either: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def bind_left(
        self, function: Callable[[TypeLeft], Either[TypeResult, Any]]
    ) -> Either[TypeResult, Any]:
        """Calls function with the wrapped Left value and returns its resulting Either Monad.

        Parameters
        ----------
        function: Callable[[TypeLeft], Either[TypeResult, TypeRight]]
            Function which takes a value of TypeLeft and returns a new Either Monad.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns the Either Monad from the function call.
        """
        return function(self._left_value)

    def apply(
        self, applicative: Either[TypeLeft, Callable[..., TypeResult]]
    ) -> Either[TypeLeft, TypeResult]:
        """Returns this Left unchanged, since apply only executes the applicative's function for Right.

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[[TypeRight], TypeResult]]
            Applicative Either Monad which contains a function.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def apply2(self, applicative_value: Either[TypeLeft, Any]) -> Either[TypeLeft, Any]:
        """Returns this Left unchanged, since apply2 only applies the value when this Either Monad is Right.

        Parameters
        ----------
        applicative_value: Either[TypeLeft, TypePure]
            Either Monad which contains a value.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def is_right_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Either Monad is Left.

        Parameters
        ----------
        function: Callable[[TypeRight], bool]
            Predicate function which would be applied to the Right value.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def is_left_and(self, function: Callable[[TypeLeft], bool]) -> bool:
        """Returns True if the Either Monad is Left and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeLeft], bool]
            Predicate function applied to the Left value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._left_value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        """Returns the default value, since this Either Monad is Left.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned since the Either Monad is Left.
        function: Callable[[TypeRight], TypeResult]
            Function which would be applied to the Right value if this Either Monad were Right.

        Returns
        -------
        result: TypeResult
            Returns the default value.
        """
        return default_value

    def and_(self, other: Either[TypeLeft, Any]) -> Either[TypeLeft, Any]:
        """Returns this Left, since and_ only returns `other` when this Either Monad is Right.

        Parameters
        ----------
        other: Either[TypeLeft, TypeResult]
            Either Monad which would be returned if this Either Monad were Right.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def or_(self, other: Either[Any, TypeRight]) -> Either[Any, TypeRight]:
        """Returns `other`, since this Either Monad is Left.

        Parameters
        ----------
        other: Either[TypeLeft, TypeRight]
            Either Monad to be returned since this Either Monad is Left.

        Returns
        -------
        result: Either[TypeLeft, TypeRight]
            Returns `other`.
        """
        return other

    def zip(self, other: Either[Any, Any]) -> Either[TypeLeft, Any]:
        """Returns this Left, since zip cannot combine values when this Either Monad is Left.

        Parameters
        ----------
        other: Either[TypeLeft, TypePure]
            Either Monad which would be zipped with this Either Monad if it were Right.

        Returns
        -------
        result: Either[TypeLeft, Tuple[TypeRight, TypePure]]
            Returns this Left.
        """
        return self

    def flatten(self) -> Either[TypeLeft, Any]:
        """Returns this Left unchanged, since there is nothing to flatten.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns this Left.
        """
        return self

    def right(self) -> Maybe[Any]:
        """Either Monad specific function to return the Right value as a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeRight]
            Returns Nothing, since this Either Monad is Left.
        """
        from pymoliath.maybe import Nothing

        return Nothing()

    def left(self) -> Maybe[TypeLeft]:
        """Either Monad specific function to return the Left value as a Maybe Monad.

        Returns
        -------
        maybe: Maybe[TypeLeft]
            Returns Just with the Left value.
        """
        from pymoliath.maybe import Just

        return Just(self._left_value)

    def unwrap(self) -> Any:
        """Raises an Exception containing the Left value, since this Either Monad is Left.

        Returns
        -------
        result: Any
            Never returns; always raises an Exception.
        """
        raise Exception(self._left_value)

    def unwrap_or(self, default_value: TypeRight) -> TypeRight:
        """Returns the provided default value, since this Either Monad is Left.

        Parameters
        ----------
        default_value: TypeRight
            Default value of TypeRight

        Returns
        -------
        result: TypeRight
            Returns the default value.
        """
        return default_value

    def unwrap_or_else(
        self, left_function: Callable[[TypeLeft], TypeRight]
    ) -> TypeRight:
        """Calls left_function with the Left value and returns its result, since this Either Monad is Left.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeRight]
            Called with the left value and must return a value of type TypeRight

        Returns
        -------
        result: TypeRight
            Returns the result of calling left_function with the Left value.
        """
        return left_function(self._left_value)

    def unwrap_left_or(self, default_value: TypeLeft) -> TypeLeft:
        """Returns the Left value, since this Either Monad is Left.

        Parameters
        ----------
        default_value: TypeLeft
            Default value of TypeLeft which is ignored since this Either Monad is Left.

        Returns
        -------
        result: TypeLeft
            Returns the Left value.
        """
        return self._left_value

    def inspect(self, function: Callable[[Any], None]) -> Either[TypeLeft, Any]:
        """Returns this Left unchanged, without calling function, since this Either Monad is Left.

        Parameters
        ----------
        function: Callable[[TypeRight], None]
            Inspection function which would be called with the Right value if this Either Monad were Right.

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
        """
        return self

    def inspect_left(
        self, function: Callable[[TypeLeft], None]
    ) -> Either[TypeLeft, Any]:
        """Inspect the Either value of TypeLeft

        Parameters
        ----------
        function: Callable[[TypeLeft], None]
            Inspection function which takes the left value of the Either monad

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
        """
        function(self._left_value)
        return self

    def match(
        self,
        left_function: Callable[[TypeLeft], TypeResult],
        right_function: Callable[[Any], TypeResult],
    ) -> TypeResult:
        """Left monad specific function to handle railroad orientated programming.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeResult]
            Callback function for either monads of type Left
        right_function: Callable[[TypeRight], TypeResult]
            Callback function for either monads of type Right
        """
        return left_function(self._left_value)

    def is_left(self) -> bool:
        """Either monad is left function

        Returns
        -------
        result: bool
            True: if either monad is of type left, False: if either monad is of type right
        """
        return True

    def is_right(self) -> bool:
        """Either monad is right function

        Returns
        -------
        result: bool
            True: if either monad is of type right, False: if either monad is of type left
        """
        return False

    def __str__(self) -> str:
        """Returns the string representation of the Left Monad."""
        return f"Left({self._left_value})"

    def __eq__(self, __o: object) -> bool:
        """Returns True if `__o` is a Left Monad with an equal string representation."""
        return isinstance(__o, Left) and str(self) == str(__o)

    def __repr__(self) -> str:
        """Returns the string representation of the Left Monad (same as __str__)."""
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
