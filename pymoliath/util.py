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
from types import FunctionType, MethodType
from typing import Any, Callable, NamedTuple, TypeVar

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class _Arity(NamedTuple):
    """How many positional arguments a callable takes (computed once, without calling it)."""

    positional: (
        int  # positional parameters, including those with defaults (excluding *args)
    )
    required: int  # positional parameters without a default
    varargs: bool  # whether it has *args
    kwonly_required: bool  # whether a keyword-only parameter has no default

    def binds(self, count: int) -> bool:
        """Whether `count` positional arguments make a complete call."""
        return (
            not self.kwonly_required
            and self.required <= count
            and (count <= self.positional or self.varargs)
        )

    def accepts(self, count: int) -> bool:
        """Whether `count` positional arguments fit, even if more are needed afterwards."""
        return count <= self.positional or self.varargs


def _code_arity(function: FunctionType, bound: int) -> _Arity:
    """Reads the arity of a plain function from its code object (fast path)."""
    code = function.__code__
    positional = code.co_argcount - bound
    defaults = len(function.__defaults__ or ())
    kwonly_defaults = len(function.__kwdefaults__ or {})
    return _Arity(
        positional=positional,
        required=max(0, code.co_argcount - defaults - bound),
        varargs=bool(code.co_flags & inspect.CO_VARARGS),
        kwonly_required=code.co_kwonlyargcount > kwonly_defaults,
    )


def _arity(function: Callable[..., Any]) -> _Arity | None:
    """The arity of `function`, or None if it can't be determined (e.g. some builtins)."""
    if isinstance(function, FunctionType):
        return _code_arity(function, 0)
    if isinstance(function, MethodType) and isinstance(function.__func__, FunctionType):
        return _code_arity(function.__func__, 1)
    if isinstance(function, partial) and not function.keywords:
        inner = _arity(function.func)
        if inner is not None:
            applied = len(function.args)
            return inner._replace(
                positional=max(0, inner.positional - applied),
                required=max(0, inner.required - applied),
            )
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    kinds = [parameter.kind for parameter in signature.parameters.values()]
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return _Arity(
        positional=len(positional),
        required=sum(
            parameter.default is inspect.Parameter.empty for parameter in positional
        ),
        varargs=inspect.Parameter.VAR_POSITIONAL in kinds,
        kwonly_required=any(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ),
    )


def compose(
    *callables: Callable[[TypeSource], TypeResult],
) -> Callable[[TypeSource], TypeResult]:
    """Compose multiple functions right to left.

    Composes zero or more functions into a functional composition. The
    functions are composed right to left. A composition of zero
    functions gives back the identity function.

    A tuple is unpacked into the arguments of a function that takes two or more positional
    parameters (required or with defaults, not counting `*args`), if the tuple fits them.
    Otherwise - for one-parameter and `*args` functions, or a tuple of the wrong length - it is
    passed as a single argument. The arity of each function is read once, when composing, and no
    function is ever retried, so exceptions raised inside the functions propagate unchanged.

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
    steps: list[tuple[Callable[..., Any], _Arity | None]] = [
        (function, _arity(function)) for function in reversed(callables)
    ]

    def composition(source: Any) -> Any:
        """Applies `callables` right to left to `source`."""
        value = source
        for function, arity in steps:
            if (
                isinstance(value, tuple)
                and arity is not None
                and arity.positional >= 2
                and arity.binds(len(value))
            ):
                value = function(*value)
            else:
                value = function(value)
        return value

    return composition


def curry(function: Callable[..., Any]) -> Callable[[Any], Any]:
    """Curries `function`: it takes its arguments one call at a time.

    `curry(f)(a)(b)(c)` is `f(a, b, c)`. A function that is complete with one argument is
    returned unchanged, so `curry(f)(a)` is `f(a)`.

    The arity is read from the function once (without calling it), so exceptions raised inside
    `function` - including TypeErrors - always propagate unchanged. Functions without a readable
    signature (some builtins) are called with one argument.

    Parameters
    ----------
    function: Callable[..., Any]
        Any function to be curried

    Returns
    -------
    result: Callable[[Any], Any]
        Curried function which takes one argument per call until all positional arguments are
        passed.

    Example
    -------
    >>> curry(lambda a, b: a + b)(1)(2)
    3
    >>> curry(lambda a, b, c: a + b + c)(1)(2)(3)
    6
    >>> curry(lambda a: a * 2)(3)
    6
    """
    if not _needs_more_than_one(function):
        return function
    return lambda value: curry(partial(function, value))


def _needs_more_than_one(function: Callable[..., Any]) -> bool:
    """Whether one positional argument fits `function` but doesn't complete the call.

    Plain functions and partials of plain functions take a fast path through the code object,
    since `curry` runs on every applicative `apply`.
    """
    applied = 0
    target = function
    if isinstance(target, partial) and not target.keywords:
        applied = len(target.args)
        target = target.func
    if isinstance(target, FunctionType):
        code = target.__code__
        required = code.co_argcount - len(target.__defaults__ or ()) - applied
        return required > 1
    arity = _arity(function)
    return arity is not None and not arity.binds(1) and arity.accepts(1)


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
