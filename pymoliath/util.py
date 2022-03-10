from functools import partial
from functools import reduce
from typing import Callable, Any, TypeVar, Union

TypeSource = TypeVar('TypeSource')
TypeResult = TypeVar('TypeResult')
TypePure = TypeVar('TypePure')


def compose(*callables: Callable[[TypeSource], TypeResult]) -> Callable[[TypeSource], TypeResult]:
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
        try:
            return reduce(lambda acc, f: f(acc), callables[::-1], source)
        except TypeError:
            return reduce(lambda acc, f: f(*acc), callables[::-1], source)

    return composition


def curry(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Currying function

    Parameters
    ----------
    function: Callable[[Any], Any]
        Any function to be curried

    Returns
    -------
    result: Callable[[Any], Any]
        Curried function which can be called until all function arguments
        are passed.
    """

    def inner(value: Any) -> Union[Any, Callable[[Any], Any]]:
        try:
            return function(value)
        except TypeError:
            return partial(function, value)

    return inner
