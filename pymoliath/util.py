"""
# Utilities

Small, dependency-free functional helpers used throughout pymoliath (and useful on their own):
function composition (`compose`, `pipe`), auto-currying (`curry`), and the classic combinators
`identity`, `const`, and `flip` from Haskell's `Prelude`.

```python
pipe(str.strip, str.upper)(" hello ")  # "HELLO"
curry(lambda x, y: x + y)(1)(2)  # 3
```
"""

import inspect
from functools import partial
from typing import Any, Callable, TypeVar

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _positional_parameters(function: Callable[..., Any]) -> list[inspect.Parameter]:
    """The positional parameters of `function` (without `*args`), or none if it has no signature."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):  # e.g. some builtins
        return []
    return [parameter for parameter in parameters if parameter.kind in _POSITIONAL]


def compose(
    *callables: Callable[[TypeSource], TypeResult],
) -> Callable[[TypeSource], TypeResult]:
    """Compose multiple functions right to left.

    Composes zero or more functions into a functional composition. The
    functions are composed right to left. A composition of zero
    functions gives back the identity function.

    A tuple is unpacked into the arguments of a function that takes two or more positional
    parameters (required or with defaults, not counting `*args`); every other function receives it
    as a single argument. Each function's signature is read once, when composing.

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
    >>> compose(lambda total: total * 2, lambda a, b: a + b)((1, 2))
    6
    >>> compose(lambda a, b=10: a + b)((1, 2))
    3
    >>> compose(len)((1, 2))
    2
    """
    steps = [
        (function, len(_positional_parameters(function)) >= 2)
        for function in reversed(callables)
    ]

    def composition(value: Any) -> Any:
        """Applies `callables` right to left to `value`."""
        for function, unpack in steps:
            if unpack and isinstance(value, tuple):
                value = function(*value)
            else:
                value = function(value)
        return value

    return composition


def curry(function: Callable[..., Any]) -> Callable[[Any], Any]:
    """Curries `function`: it takes its required positional arguments one call at a time.

    `curry(f)(a)(b)(c)` is `f(a, b, c)`. A function that needs at most one argument is returned
    unchanged, so `curry(f)(a)` is `f(a)`. Functions without a signature (some builtins) are
    treated as one-argument functions.

    Parameters
    ----------
    function: Callable[..., Any]
        Any function to be curried

    Returns
    -------
    result: Callable[[Any], Any]
        Curried function which takes one argument per call until all required positional
        arguments are passed.

    Example
    -------
    >>> curry(lambda a, b: a + b)(1)(2)
    3
    >>> curry(lambda a, b, c: a + b + c)(1)(2)(3)
    6
    >>> curry(lambda a: a * 2)(3)
    6
    """
    return _curried(function, _required_positional(function))


def _required_positional(function: Callable[..., Any]) -> int:
    """How many positional arguments `function` needs (not counting defaults and `*args`)."""
    if inspect.isfunction(function):
        # Fast path for plain functions and lambdas (what `apply` usually curries): read the code
        # object directly instead of building an `inspect.Signature`, which is ~20x slower.
        code = function.__code__
        return code.co_argcount - len(function.__defaults__ or ())
    return sum(
        parameter.default is inspect.Parameter.empty
        for parameter in _positional_parameters(function)
    )


def _curried(function: Callable[..., Any], remaining: int) -> Callable[[Any], Any]:
    """Takes the `remaining` arguments one call at a time, then calls `function` with the last."""
    if remaining <= 1:
        return function
    return lambda value: _curried(partial(function, value), remaining - 1)


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
