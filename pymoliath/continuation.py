"""
# Continuation Monad

The Continuation Monad represents a computation written in continuation-passing style (CPS).
Instead of returning its result directly, a CPS computation accepts a callback - the
"continuation" - and hands its result to that callback instead of returning it. This inverts
control: the *caller* decides what happens next by supplying the continuation, while the
computation itself decides *when* (or whether, or how many times) to invoke it.

* Haskell: [Control.Monad.Cont](https://hackage.haskell.org/package/mtl/docs/Control-Monad-Cont.html)

This implementation is heavily inspired by Haskell's `Cont` monad.

The `Continuation` type wraps a computation of type
`Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]`: given a callback from `TypeSource` to
`TypeReturn`, it produces a `TypeReturn`.

```python
Continuation[TypeSource, TypeReturn]
```

## Practical Examples and Benefits:

Continuation-passing style underlies callback-based APIs, generator-based coroutines, and
early-return-like control flow. Wrapping CPS computations in a monad lets `map`/`bind` compose
chains of continuations declaratively instead of manually nesting callbacks.

### Benefits:
1. Composability: Chains of continuation-passing functions can be composed with `map`/`bind`
   instead of manually nesting callbacks ("callback hell").
2. Explicit Control Flow: A `Continuation` never runs anything until `run` is called with a
   concrete callback, making the point where control actually transfers explicit in the code.
3. Foundation for Control Abstractions: Because a continuation can call its callback zero, one, or
   many times (or not at all), non-local control flow like early exit or backtracking can be built
   on top of it.

#### Example: Composing computations without nesting callbacks.

```python
# Without Continuation (nested callbacks)
def get_user(user_id, callback):
    callback(lookup_user(user_id))

def get_profile(user, callback):
    callback(lookup_profile(user))

get_user(1, lambda user: get_profile(user, lambda profile: print(profile)))

# With Continuation (flat composition)
(Continuation(lambda callback: get_user(1, callback))
    .bind(lambda user: Continuation(lambda callback: get_profile(user, callback)))
    .run(print))
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeReturn = TypeVar("TypeReturn")
TypeResult = TypeVar("TypeResult")


class Continuation(Generic[TypeSource, TypeReturn]):
    """The Continuation monad represents computations in continuation-passing style (CPS).

    In continuation-passing style a function's result is not returned but instead passed to
    another function, received as a parameter (the continuation). `Continuation` wraps such a
    computation so it can be transformed with `map`/`bind` before ever being run.
    """

    __slots__ = ("_computation",)

    _computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]

    def __init__(
        self, computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]
    ):
        """Continuation Monad constructor which takes a computation of type
        Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn].

        Parameters
        ----------
        computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]
            Callable to be stored in the Continuation Monad, invoked with a callback when `run`
            is called.

        Examples
        --------
        >>> cont: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> cont.run(str)
        '10'
        """
        self._computation = computation

    def map(
        self: Continuation[TypeSource, TypeReturn],
        function: Callable[[TypeSource], TypeResult],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation monad functor interface (>=, map).

        Composes `function` in front of the continuation's eventual callback, transforming the
        value that will be produced without running the computation yet.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Returns a new continuation monad with the composed computation and the passed
            function.

        Examples
        --------
        >>> cont: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> cont.map(lambda x: x + 1).run(str)
        '11'
        """

        def mapping(callback: Callable[[TypeResult], TypeReturn]) -> TypeReturn:
            """Runs this continuation with a callback that first applies `function`, then `callback`."""
            return self.run(lambda source: callback(function(source)))

        return Continuation(mapping)

    def bind(
        self: Continuation[TypeSource, TypeReturn],
        function: Callable[[TypeSource], Continuation[TypeResult, TypeReturn]],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation monad bind interface (>>=, bind, flatMap).

        Chains this continuation into a second continuation-producing function, running the
        second continuation's computation with the original callback once the first resolves.

        Parameters
        ----------
        function: Callable[[TypeSource], Continuation[TypeResult, TypeReturn]]
            Function which takes a value of type TypeSource and returns a new continuation of
            type TypeResult.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Returns a new continuation monad with the function bound to the continuation
            computation.

        Examples
        --------
        >>> cont: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> cont.bind(lambda x: Continuation(lambda callback: callback(x + 1))).run(str)
        '11'
        """

        return Continuation(lambda cont: self.run(lambda a: function(a).run(cont)))

    def apply(
        self: Continuation[TypeSource, TypeReturn],
        applicative: Continuation[Callable[..., TypeResult], TypeReturn],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation Monad applicative interface for continuations containing a value (<*>).

        Parameters
        ----------
        applicative: Continuation[Callable[[TypeSource], TypeResult], TypeReturn]
            Applicative continuation which contains a function and will be applied to this
            continuation.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Applies a continuation containing a value to a continuation containing a function.

        Examples
        --------
        >>> val: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> func: Continuation[Callable[[int], int], str] = Continuation(
        ...     lambda callback: callback(lambda x: x * 2)
        ... )
        >>> val.apply(func).run(str)
        '20'
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Continuation[TypeResult, TypeReturn]:
            """Maps the applicative's function, curried, over this Continuation's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Continuation[Callable[..., TypeResult], TypeReturn],
        applicative_value: Continuation[Any, TypeReturn],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation Monad applicative interface for continuations containing a function (<*>).

        Parameters
        ----------
        applicative_value: Continuation[TypeSource, TypeReturn]
            Continuation which contains a value and will be applied to this continuation
            containing a function.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Applies a continuation containing a function to a continuation containing a value.

        Examples
        --------
        >>> func: Continuation[Callable[[int], int], str] = Continuation(
        ...     lambda callback: callback(lambda x: x * 2)
        ... )
        >>> val: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> func.apply2(val).run(str)
        '20'
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Continuation[TypeResult, TypeReturn]:
            """Maps the applicative value's function, curried, over this Continuation's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def run(self, callback: Callable[[TypeSource], TypeReturn]) -> TypeReturn:
        """Run the computation of the continuation with the passed callback.

        Parameters
        ----------
        callback: Callable[[TypeSource], TypeReturn]
            Callback which receives the computation's result instead of it being returned
            directly.

        Returns
        -------
        result: TypeReturn
            Returns whatever `callback` returns when invoked by the continuation's computation.

        Examples
        --------
        >>> cont: Continuation[int, str] = Continuation(lambda callback: callback(10))
        >>> cont.run(str)
        '10'
        """
        return self._computation(callback)
