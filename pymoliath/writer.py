from __future__ import annotations

from functools import partial
from typing import Callable, Generic, TypeVar, Tuple

TypeSource = TypeVar('TypeSource')
TypePure = TypeVar("TypePure")
TypeMonoid = TypeVar('TypeMonoid')
TypeResult = TypeVar('TypeResult')


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
    _value: Tuple[TypeSource, TypeMonoid]  # Private writer monad value which should not be modified

    def __init__(self, value: TypeSource, monoid: TypeMonoid) -> None:
        """Writer monad constructor which takes a value of type TypeSource and a monoid of type TypeMonoid.

        Parameters
        ----------
        value: TypeSource
            Generic writer monad value
        monoid: TypeMonoid
            Generic writer monad monoid (see description above)
        """
        self._value = (value, monoid)

    def map(self: Writer[TypeSource, TypeMonoid],
            function: Callable[[TypeSource], TypeResult]) -> Writer[TypeResult, TypeMonoid]:
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
        """
        value, monoid = self.run()
        return self.__class__(function(value), monoid)

    def bind(self: Writer[TypeSource, TypeMonoid],
             function: Callable[[TypeSource], Writer[TypeResult, TypeMonoid]]) -> Writer[TypeResult, TypeMonoid]:
        """Writer monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], Writer[TypeResult, TypeMonoid]]
            Function which takes a value of type TypeSource and returns a writer monad of type TypeResult, TypeMonoid.

        Returns
        -------
        writer: Writer[TypeResult, TypeMonoid]
            Returns a writer monad with the function result and the closure of the monoids (+)
        """
        value, monoid = self.run()
        result, other_monoid = function(value).run()
        return self.__class__(result, monoid + other_monoid)  # type: ignore 

    def apply(self: Writer[TypeSource, TypeMonoid],
              applicative: Writer[Callable[[TypeSource], TypeResult], TypeMonoid]) -> Writer[TypeResult, TypeMonoid]:
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
        """

        value, monoid = self.run()
        function, other_monoid = applicative.run()
        try:
            return self.__class__(function(value), monoid + other_monoid)  # type: ignore
        except TypeError:
            return self.__class__(partial(function, value), monoid + other_monoid)  # type: ignore

    def apply2(self: Writer[Callable[[TypePure], TypeResult], TypeMonoid],
               monad_value: Writer[TypePure, TypeMonoid]) -> Writer[TypeResult, TypeMonoid]:
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
        """
        value_function, monoid = self.run()
        value, other_monoid = monad_value.run()
        try:
            return self.__class__(value_function(value), monoid + other_monoid)  # type: ignore
        except TypeError:
            return self.__class__(partial(value_function, value), monoid + other_monoid)  # type: ignore

    def tell(self: Writer[TypeSource, TypeMonoid], monoid_value: TypeMonoid) -> Writer[TypeSource, TypeMonoid]:
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
        """
        value, monoid = self.run()
        return self.__class__(value, monoid + monoid_value)  # type: ignore

    def listen(self: Writer[TypeSource, TypeMonoid]) -> Writer[TypeSource, Tuple[TypeSource, TypeMonoid]]:
        """Writer monad specific function listen.

        Definition: listen :: Monad(a, m) -> Monad(a, (a, m))

        Listen is an action that executes the action in the monad and adds its output to the value of the computation.

        Returns
        -------
        writer: Writer[TypeSource, Tuple[TypeSource, TypeMonoid]]
        """
        return self.map(lambda _: self.run())  # type: ignore

    def pass_(self: Writer[Tuple[TypeSource, Callable[[TypeMonoid], TypeMonoid]], TypeMonoid]) -> Writer[
        TypeSource, TypeMonoid]:
        """Writer monad specific function pass_ (actually pass)

        Definition: pass :: Monad((a, f: m -> m), m) -> Monad(a, m)

        The pass function will execute the function which is contained in the tuple value of the writer monad and
        applies it to the monoid value. Returns a writer monad containing the value and the resulting monoid value.

        Returns
        -------
        writer: Writer[TypeSource, TypeMonoid]
        """
        pass_tuple, monoid = self.run()
        value, monoid_function = pass_tuple
        return self.__class__(value, monoid_function(monoid))  # type: ignore

    def run(self: Writer[TypeSource, TypeMonoid]) -> Tuple[TypeSource, TypeMonoid]:
        return self._value

    def __str__(self) -> str:
        return f'Writer {self._value}'

    def __repr__(self) -> str:
        return str(self)
