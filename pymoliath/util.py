from functools import partial
from functools import reduce
from typing import Callable, Any, TypeVar, Union

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


def compose(
    *callables: Callable[[TypeSource], TypeResult],
) -> Callable[[TypeSource], TypeResult]:
    """Compose multiple functions right to left.

    Composes zero or more functions into a functional composition. The
    functions are composed right to left. A composition of zero
    functions gives back the identity function.

    Parameters
    ----------
    callables: Callable
      Multiple functions to be composed

    Returns
    -------
    result: Callable
      Composed callable function

    Example
    -------
    >>> compose()(10)
    10
    >>> compose(lambda x: x)(11)
    11
    >>> compose(lambda x: x, lambda y: y + 2)(10)
    12
    """

    def composition(source: Any) -> Any:
        """Applies `callables` right to left to `source`, falling back to unpacking on TypeError."""
        try:
            return reduce(lambda acc, f: f(acc), callables[::-1], source)
        except TypeError:
            return reduce(lambda acc, f: f(*acc), callables[::-1], source)

    return composition


def curry(function: Callable[..., Any]) -> Callable[[Any], Any]:
    """Currying function

    Parameters
    ----------
    function: Callable[..., Any]
        Any function to be curried

    Returns
    -------
    result: Callable[[Any], Any]
        Curried function which can be called until all function arguments
        are passed.
    """

    def inner(value: Any) -> Union[Any, Callable[[Any], Any]]:
        """Applies `value` to `function`, or partially applies it if `function` needs more arguments."""
        try:
            return function(value)
        except TypeError:
            return partial(function, value)

    return inner


def pipe(
    *callables: Callable[[TypeSource], TypeResult],
) -> Callable[[TypeSource], TypeResult]:
    """Compose multiple functions left to right.

    Composes zero or more functions into a functional composition. The
    functions are composed left to right (the mirror of `compose`). A
    composition of zero functions gives back the identity function.

    Parameters
    ----------
    callables: Callable
      Multiple functions to be composed

    Returns
    -------
    result: Callable
      Composed callable function

    Example
    -------
    >>> pipe()(10)
    10
    >>> pipe(lambda x: x)(11)
    11
    >>> pipe(lambda x: x, lambda y: y + 2)(10)
    12
    """
    return compose(*callables[::-1])


def identity(value: TypeSource) -> TypeSource:
    """Identity function (Haskell `id`).

    Parameters
    ----------
    value: TypeSource
        Any value

    Returns
    -------
    result: TypeSource
        Returns the passed value unchanged.

    Example
    -------
    >>> identity(10)
    10
    """
    return value


def const(value: TypeSource) -> Callable[[Any], TypeSource]:
    """Haskell `const`: returns a function which always returns `value`, ignoring its argument.

    Parameters
    ----------
    value: TypeSource
        Value to be returned by the constant function.

    Returns
    -------
    result: Callable[[Any], TypeSource]
        A function which always returns `value` regardless of the passed argument.

    Example
    -------
    >>> const(10)("ignored")
    10
    """

    def constant(_: Any) -> TypeSource:
        """Ignores its argument and returns the enclosing `value`."""
        return value

    return constant


def flip(
    function: Callable[[TypeSource, TypeResult], TypePure],
) -> Callable[[TypeResult, TypeSource], TypePure]:
    """Haskell `flip`: swaps the argument order of a two-argument function.

    Parameters
    ----------
    function: Callable[[TypeSource, TypeResult], TypePure]
        A two-argument function.

    Returns
    -------
    result: Callable[[TypeResult, TypeSource], TypePure]
        The passed function with its two arguments swapped.

    Example
    -------
    >>> flip(lambda a, b: a - b)(2, 10)
    8
    """

    def flipped(b: TypeResult, a: TypeSource) -> TypePure:
        """Calls `function` with its two arguments in the original (unswapped) order."""
        return function(a, b)

    return flipped
