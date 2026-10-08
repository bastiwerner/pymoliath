"""
# IO Monad

The IO Monad is a container used to represent a computation that performs side effects (reading a
file, printing to the console, calling an external service) before producing a value. It wraps a
zero-argument callable rather than a plain value, so the side effect is deferred: constructing an
`IO` does nothing by itself, and the wrapped computation only actually runs once `run()` is called.

* Haskell: [System.IO](https://hackage.haskell.org/package/base-4.16.0.0/docs/System-IO.html)

This implementation is heavily inspired by Haskell's `IO` monad.

The `IO[TypeSource]` type wraps a `Callable[[], TypeSource]`: a computation which, once run,
performs some I/O before returning a value of type `TypeSource`.

```python
IO[TypeSource]
```

## Practical Examples and Benefits:

The IO Monad is useful for keeping side-effecting code explicit and composable: instead of calling
side-effecting functions eagerly and interleaving them with pure logic, you build up a description
of the computation with `map`/`bind`, and only trigger the actual effects once, at the edge of the
program, by calling `run()`.

### Benefits:
1. Explicit Effects: Wrapping a computation in `IO` marks it as side-effecting in the type system,
   separating it from pure functions.
2. Deferred Execution: Nothing runs until `run()` is called, so a chain of `map`/`bind` calls can be
   built up and passed around as a value before any side effect actually happens.
3. Composability: `map`/`bind`/`apply` let you combine IO actions the same way you would combine
   pure functions, instead of manually sequencing statements.

#### Example: Reading a value and transforming it before printing it.

```python
# Without IO (Imperative)
raw = input("Enter a number: ")
doubled = int(raw) * 2
print(doubled)

# With IO (Functional)
action = IO(lambda: input("Enter a number: ")).map(lambda raw: int(raw) * 2).map(print)
action.run()
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class IO(Generic[TypeSource]):
    """IO monad implementation

    A value of type `IO[TypeSource]` is a computation which, when performed, does some I/O before
    returning a value of type `TypeSource`. Unlike `Maybe`/`Result`, `IO` has no failure state -
    every `IO` action "succeeds" from the monad's perspective, and exceptions propagate normally.
    """

    __slots__ = ("_value",)

    _value: Callable[
        [], TypeSource
    ]  # Private io monad value of type callable which should not be modified

    def __init__(self, value: Callable[[], TypeSource]):
        """IO Monad constructor which takes a callable of type Callable[[], TypeSource].

        Parameters
        ----------
        value: Callable[[], TypeSource]
            Callable to be stored in the IO Monad, executed when `run` is called.

        Examples
        --------
        >>> action: IO[int] = IO(lambda: 10)
        >>> action.run()
        10
        """
        if not callable(value):
            raise TypeError("IO value must be of type Callable")
        self._value = value

    def map(
        self: IO[TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> IO[TypeResult]:
        """IO monad functor interface (>=, map).

        Definition: IO(f: _ -> a) >= f: a -> b => IO(f: _ -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        io: IO[TypeResult]
            Returns a new io monad containing the result of the passed function and the io call.

        Examples
        --------
        >>> action: IO[int] = IO(lambda: 10)
        >>> action.map(lambda x: x + 1).run()
        11
        """
        return IO(lambda: function(self.run()))

    def bind(
        self: IO[TypeSource], function: Callable[[TypeSource], IO[TypeResult]]
    ) -> IO[TypeResult]:
        """IO monad bind interface (>>=, bind, flatMap).

        Definition: IO(f: _ -> a) >>= f: a -> IO(f: _ -> b) => IO(f: _ -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], IO[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        io: IO[TypeResult]
            Returns an io monad with the function call result

        Examples
        --------
        >>> action: IO[int] = IO(lambda: 10)
        >>> action.bind(lambda x: IO(lambda: x + 1)).run()
        11
        """
        return function(self.run())

    def apply(
        self: IO[TypeSource], applicative: IO[Callable[..., TypeResult]]
    ) -> IO[TypeResult]:
        """IO monad applicative interface for io monads containing a function returning a value (<*>).

        Definition: IO(f: _ -> a) <*> IO(f: _ -> f: a -> b) => IO(f: _ -> b)

        Parameters
        ----------
        applicative: IO[Callable[[TypeSource], TypeResult]]
            Applicative io monad which contains a function and will be applied to the io monad containing a value.

        Returns
        -------
        io: IO[TypeResult]
            Applies an io monad containing a value of type TypeSource to an io monad containing a function
            of type Callable[[TypeSource], TypeResult].

        Examples
        --------
        >>> val: IO[int] = IO(lambda: 10)
        >>> func: IO[Callable[[int], int]] = IO(lambda: (lambda x: x * 2))
        >>> val.apply(func).run()
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> IO[TypeResult]:
            """Maps the applicative's function, curried, over this IO monad's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: IO[Callable[..., TypeResult]], applicative_value: IO[Any]
    ) -> IO[TypeResult]:
        """IO monad applicative interface for io monads containing a function (<*>).

        Definition: IO(f: _ -> f: a -> b) <*> IO(f: _ -> a) => IO(f: _ -> b)

        Parameters
        ----------
        applicative_value: IO[TypePure]
            IO monad value which will be applied to the io monad containing a function

        Returns
        -------
        io: IO[TypeResult]
            Applies an io monad containing a function of type Callable[[TypePure], TypeResult]
            to an io monad of type TypePure (value or function).

        Examples
        --------
        >>> func: IO[Callable[[int], int]] = IO(lambda: (lambda x: x * 2))
        >>> val: IO[int] = IO(lambda: 10)
        >>> func.apply2(val).run()
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> IO[TypeResult]:
            """Maps the applicative value's function, curried, over this IO monad's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def run(self: IO[TypeSource]) -> TypeSource:
        """IO monad lazy run function to start the io action.

        Definition: IO(f: _ -> a).run() => a

        Returns
        -------
        result: TypeSource
            Calls the io monad function which returns a value of type TypeSource.

        Examples
        --------
        >>> action: IO[int] = IO(lambda: 10)
        >>> action.run()
        10
        """
        return self._value()

    def __str__(self) -> str:
        """Returns the string representation of the IO Monad.

        Examples
        --------
        >>> str(IO(lambda: 10))  # doctest: +ELLIPSIS
        'IO(<function...>)'
        """
        return f"IO({self._value})"

    def __repr__(self) -> str:
        """Returns the string representation of the IO Monad (same as __str__).

        Examples
        --------
        >>> repr(IO(lambda: 10))  # doctest: +ELLIPSIS
        'IO(<function...>)'
        """
        return str(self)
