"""
# Writer Monad

The Writer Monad represents a computation that produces a value while also accumulating a log
alongside it - typically a string of diagnostic messages, but any monoid works. It wraps a tuple
of `(TypeSource, TypeMonoid)`: the computed value and the accumulated log. `bind` automatically
combines the logs of chained computations using the monoid's `+` operator, so the caller never has
to merge logs by hand.

* Haskell: [Control.Monad.Writer](https://hackage.haskell.org/package/mtl/docs/Control-Monad-Writer.html)

This implementation is heavily inspired by Haskell's `Writer` monad.

The `Writer[TypeSource, TypeMonoid]` type wraps a `Tuple[TypeSource, TypeMonoid]`: a value of type
`TypeSource` paired with an accumulated log of type `TypeMonoid`.

```python
Writer[TypeSource, TypeMonoid]
```

## Practical Examples and Benefits:

The Writer Monad is particularly useful for attaching a log, trace, or set of accumulated
diagnostics to a computation without threading a mutable log object through every function call by
hand (e.g. logging each step of a calculation, or collecting warnings produced along the way).

### Benefits:
1. Implicit Log Accumulation: `bind` combines logs from chained computations automatically via the
   monoid's `+`, so intermediate steps don't need to know about or mutate a shared log.
2. Separation of Concerns: The "what happened" (the log) is kept separate from "the result", while
   still being produced by the exact same computation.
3. Composability: `map`/`bind`/`apply` compose Writer computations the same way any other monad is
   composed, with logs accumulating for free along the way.

#### Example: Accumulating a log of steps without a mutable log variable.

```python
# Without Writer (Imperative)
log = []
value = 5
value = value + 1
log.append(f"incremented to {value}")
value = value * 2
log.append(f"doubled to {value}")
print(value, log)

# With Writer (Functional)
result = (
    Writer(5, [])
    .bind(lambda v: Writer(v + 1, [f"incremented to {v + 1}"]))
    .bind(lambda v: Writer(v * 2, [f"doubled to {v * 2}"]))
)
print(result.run())
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Protocol, Tuple, TypeVar

from pymoliath.util import curry

TSupportsAdd = TypeVar("TSupportsAdd", bound="SupportsAdd")


class SupportsAdd(Protocol):
    """Structural bound for TypeMonoid: any type implementing the monoid closure law (a + b is also in TypeMonoid)."""

    def __add__(self: TSupportsAdd, other: TSupportsAdd, /) -> TSupportsAdd:
        """Returns the result of combining self with other under the monoid's associative operation."""
        ...


TypeSource = TypeVar("TypeSource")
TypePure = TypeVar("TypePure")
TypeMonoid = TypeVar("TypeMonoid", bound=SupportsAdd)
TypeInner = TypeVar("TypeInner")
TypeResult = TypeVar("TypeResult")


class Writer(Generic[TypeSource, TypeMonoid]):
    """Writer monad implementation.

    The Writer[TypeSource, TypeMonoid] class represents a computation that produces a tuple containing a value of
    type TypeSource and one of type TypeMonoid. The value TypeMonoid represents a data type
    which must follow the monoid laws. A typical example of an TypeMonoid value could be a logging String.

    TypeSource: any type which will be used for monad computation (bind, map, apply)
    TypeMonoid: any type that behaves as a monoid which can be added together.

    The class of monoids TypeMonoid (types with an associative binary operation (e.g. + ) that has an identity).

    Monoid instances should satisfy the following laws:

    1. Closure: If 'a' and 'b' are in TypeMonoid, then 'a + b' is also in TypeMonoid.
    2. Identity: There exists an element in TypeMonoid (denoted 0) such that: a + 0 = a = 0 + a
    3. Associativity: (a + b) + c = a + (b + c)
    """

    __slots__ = ("_value",)

    _value: Tuple[
        TypeSource, TypeMonoid
    ]  # Private writer monad value which should not be modified

    def __init__(self, value: TypeSource, monoid: TypeMonoid) -> None:
        """Writer monad constructor which takes a value of type TypeSource and a monoid of type TypeMonoid.

        Parameters
        ----------
        value: TypeSource
            Generic writer monad value
        monoid: TypeMonoid
            Generic writer monad monoid (see description above)

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, [])
        >>> writer.run()
        (10, [])
        """
        self._value = (value, monoid)

    def map(
        self: Writer[TypeSource, TypeMonoid],
        function: Callable[[TypeSource], TypeResult],
    ) -> Writer[TypeResult, TypeMonoid]:
        """Writer monad functor interface (>=, map).

        Definition: M(a, m) >= f: a -> b => M(b, m)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        writer: Writer[TypeResult, TypeMonoid]
            Returns a writer monad with the function result as value and the closure of the monoids (+)

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, ["created"])
        >>> writer.map(lambda x: x + 1).run()
        (11, ['created'])
        """
        value, monoid = self.run()
        return Writer(function(value), monoid)

    def bind(
        self: Writer[TypeSource, TypeMonoid],
        function: Callable[[TypeSource], Writer[TypeResult, TypeMonoid]],
    ) -> Writer[TypeResult, TypeMonoid]:
        """Writer monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], Writer[TypeResult, TypeMonoid]]
            Function which takes a value of type TypeSource and returns a writer monad of type TypeResult, TypeMonoid.

        Returns
        -------
        writer: Writer[TypeResult, TypeMonoid]
            Returns a writer monad with the function result and the closure of the monoids (+)

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, ["created"])
        >>> writer.bind(lambda x: Writer(x + 1, ["incremented"])).run()
        (11, ['created', 'incremented'])
        """
        value, monoid = self.run()
        result, other_monoid = function(value).run()
        return Writer(result, monoid + other_monoid)

    def apply(
        self: Writer[TypeSource, TypeMonoid],
        applicative: Writer[Callable[..., TypeResult], TypeMonoid],
    ) -> Writer[TypeResult, TypeMonoid]:
        """Writer monad applicative interface for writer monads containing a value.

        Parameters
        ----------
        applicative: Writer[Callable[[TypeSource], TypeResult], TypeMonoid]
            Writer monad applicative containing a function as value.

        Returns
        -------
        writer: Writer[TypeResult, TypeMonoid]
            Applies a writer monad containing a value of type TypeSource to a writer monad containing a function
            of type Callable[[TypeSource], TypeResult].

        Examples
        --------
        >>> val: Writer[int, list] = Writer(10, ["value"])
        >>> func: Writer[Callable[[int], int], list] = Writer(lambda x: x * 2, ["func"])
        >>> val.apply(func).run()
        (20, ['value', 'func'])
        """

        value, monoid = self.run()
        function, other_monoid = applicative.run()
        # The dynamic partial-application fallback can't be typed statically:
        # partial[TypeResult] isn't TypeResult, but it's a valid TypeResult once
        # fully applied by a later apply/apply2 call.
        return Writer(curry(function)(value), monoid + other_monoid)

    def apply2(
        self: Writer[Callable[..., TypeResult], TypeMonoid],
        monad_value: Writer[Any, TypeMonoid],
    ) -> Writer[TypeResult, TypeMonoid]:
        """Writer monad applicative interface for writer monads containing a function.

        Parameters
        ----------
        monad_value: Writer[TypePure, TypeMonoid]
            Writer monad value which will be applied to the writer monad containing a function

        Returns
        -------
        writer: Writer[TypeResult, TypeMonoid]
            Applies a writer monad containing a function of type Callable[[TypeSource], TypeResult]
            to a writer monad of type TypeSource (value or function).

        Examples
        --------
        >>> func: Writer[Callable[[int], int], list] = Writer(lambda x: x * 2, ["func"])
        >>> val: Writer[int, list] = Writer(10, ["value"])
        >>> func.apply2(val).run()
        (20, ['func', 'value'])
        """
        value_function, monoid = self.run()
        value, other_monoid = monad_value.run()
        return Writer(curry(value_function)(value), monoid + other_monoid)

    def tell(
        self: Writer[TypeSource, TypeMonoid], monoid_value: TypeMonoid
    ) -> Writer[TypeSource, TypeMonoid]:
        """Writer monad specific function to add or create a writer monad with a monoid value.

        Definition: Monad(a, m) :: tell(n) -> Monad(a, m + n)  where m must be a monoid which can be empty.

        Parameters
        ----------
        monoid_value: TypeMonoid
            Monoid value of type TypeMonoid

        Returns
        -------
        writer: Writer[TypeSource, TypeMonoid]
           Returns a writer monad containing the closure (+) of the passed monoid value.

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, ["created"])
        >>> writer.tell(["logged"]).run()
        (10, ['created', 'logged'])
        """
        value, monoid = self.run()
        return Writer(value, monoid + monoid_value)

    def listen(
        self: Writer[TypeSource, TypeMonoid],
    ) -> Writer[Tuple[TypeSource, TypeMonoid], TypeMonoid]:
        """Writer monad specific function listen.

        Definition: listen :: Monad(a, m) -> Monad(a, (a, m))

        Listen is an action that executes the action in the monad and adds its output to the value of the computation.

        Returns
        -------
        writer: Writer[Tuple[TypeSource, TypeMonoid], TypeMonoid]

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, ["created"])
        >>> writer.listen().run()
        ((10, ['created']), ['created'])
        """
        return self.map(lambda _: self.run())

    def pass_(
        self: Writer[Tuple[TypeInner, Callable[[TypeMonoid], TypeMonoid]], TypeMonoid],
    ) -> Writer[TypeInner, TypeMonoid]:
        """Writer monad specific function pass_ (actually pass)

        Definition: pass :: Monad((a, f: m -> m), m) -> Monad(a, m)

        The pass function will execute the function which is contained in the tuple value of the writer monad and
        applies it to the monoid value. Returns a writer monad containing the value and the resulting monoid value.

        Returns
        -------
        writer: Writer[TypeSource, TypeMonoid]

        Examples
        --------
        >>> writer: Writer[Tuple[int, Callable[[list], list]], list] = Writer(
        ...     (10, lambda log: [entry.upper() for entry in log]), ["created"]
        ... )
        >>> writer.pass_().run()
        (10, ['CREATED'])
        """
        pass_tuple, monoid = self.run()
        value, monoid_function = pass_tuple
        return Writer(value, monoid_function(monoid))

    def run(self: Writer[TypeSource, TypeMonoid]) -> Tuple[TypeSource, TypeMonoid]:
        """Writer monad run function to return the stored value and monoid.

        Returns
        -------
        result: Tuple[TypeSource, TypeMonoid]
            Returns the tuple of the stored value and monoid.

        Examples
        --------
        >>> writer: Writer[int, list] = Writer(10, ["created"])
        >>> writer.run()
        (10, ['created'])
        """
        return self._value

    def __str__(self) -> str:
        """Returns the string representation of the Writer Monad.

        Examples
        --------
        >>> str(Writer(10, ["created"]))
        "Writer((10, ['created']))"
        """
        return f"Writer({self._value})"

    def __repr__(self) -> str:
        """Returns the string representation of the Writer Monad (same as __str__).

        Examples
        --------
        >>> repr(Writer(10, ["created"]))
        "Writer((10, ['created']))"
        """
        return str(self)
