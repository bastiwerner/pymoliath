"""
# List Monad

The List Monad represents a computation that can produce zero, one, or many results at once. It
extends Python's built-in `list`, so any `ListMonad` is a regular list, but adds `map`/`bind`/
`apply` plus a large set of Rust-`Iterator`-flavored helpers (`fold`, `find`, `take_while`,
`dedup`, ...) that operate eagerly and return a new `ListMonad`.

* Haskell: [Data.List](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-List.html)
* Rust: [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html)

This implementation is heavily inspired by Haskell's list monad and Rust's `Iterator` trait - see
`pymoliath.lazy` for a lazily-evaluated counterpart (`Sequence`) sharing the same operation names.

```python
ListMonad[TypeSource]
```

## Practical Examples and Benefits:

The List Monad is particularly useful for representing non-deterministic computations - a step
that can produce several possible results - and for composing collection pipelines declaratively
instead of writing nested loops and mutable accumulator variables.

### Benefits:
1. Non-Determinism: `bind` naturally models computations with multiple possible outcomes: binding
   a function that returns a `ListMonad` over every element flattens all the possibilities into one
   list, the way nested `for` loops would.
2. Declarative Pipelines: `map`/`filter`/`fold` and friends chain into a single expression instead
   of an imperative loop with intermediate accumulator variables.
3. Rich Vocabulary: Common list operations (`find`, `partition`, `dedup`, `take_while`, `min_by_key`,
   ...) are available as methods, so common iteration patterns don't need to be re-implemented by
   hand each time.

#### Example: Cartesian product via bind instead of nested loops.

```python
# Without ListMonad (nested loops)
pairs = []
for x in [1, 2]:
    for y in ["a", "b"]:
        pairs.append((x, y))

# With ListMonad (bind)
pairs = ListMonad([1, 2]).bind(lambda x: ListMonad(["a", "b"]).map(lambda y: (x, y)))
```
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Hashable
from functools import reduce
from itertools import chain
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Protocol,
    Tuple,
    TypeVar,
    cast,
    overload,
)

from pymoliath.option import Nil, Option, Some
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")
TypeHashable = TypeVar("TypeHashable", bound=Hashable)


class SupportsLessThan(Protocol):
    """Structural bound for TypeOrd: any type supporting the `<` ordering comparison."""

    def __lt__(self, other: Any, /) -> bool:
        """Returns True if self is ordered before other."""
        ...


TypeOrd = TypeVar("TypeOrd", bound=SupportsLessThan)


class SupportsAdd(Protocol):
    """Structural bound for TypeMonoid: any type implementing the monoid closure law (a + b is also in TypeMonoid)."""

    def __add__(self, other: Any, /) -> Any:
        """Returns the result of adding other to self."""
        ...


TypeMonoid = TypeVar("TypeMonoid", bound=SupportsAdd)


class ListMonad(List[TypeSource]):
    """A list which additionally implements the Monad interface (map/bind/apply) and a large set
    of eager, Rust-`Iterator`-flavored helper methods, all returning a new `ListMonad`.
    """

    __slots__ = ()

    def map(
        self: ListMonad[TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> ListMonad[TypeResult]:
        """ListMonad monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        list: ListMonad[TypeResult]
            Returns a new list monad with the elements applied to the function.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).map(lambda x: x + 1)
        [2, 3, 4]
        """
        return ListMonad(map(function, self))

    def bind(
        self: ListMonad[TypeSource],
        function: Callable[[TypeSource], ListMonad[TypeResult]],
    ) -> ListMonad[TypeResult]:
        """ListMonad monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], ListMonad[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        list: ListMonad[TypeResult]
            Returns a list monad from the result of the function call.

        Examples
        --------
        >>> ListMonad([1, 2]).bind(lambda x: ListMonad([x, x * 10]))
        [1, 10, 2, 20]
        """
        return ListMonad(chain.from_iterable(map(function, self)))

    def filter(
        self: ListMonad[TypeSource], filter_function: Callable[[TypeSource], bool]
    ) -> ListMonad[TypeSource]:
        """ListMonad filter function

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            Filter function for the list

        Returns
        -------
        filtered: ListMonad[TypeSource]
            Returns a new filtered ListMonad

        Examples
        --------
        >>> ListMonad([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
        [2, 4]
        """
        return ListMonad(filter(filter_function, self))

    def take(self: ListMonad[TypeSource], amount: int) -> ListMonad[TypeSource]:
        """ListMonad take function

        Parameters
        ----------
        amount: int
            Amount of values to be taken from the list for the next operation.

        Returns
        -------
        list: ListMonad[TypeSource]
            Takes our only an specific amount of values from the list for further execution.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 4]).take(2)
        [1, 2]
        """
        return ListMonad(itertools.islice(self, amount))

    def skip(self: ListMonad[TypeSource], amount: int) -> ListMonad[TypeSource]:
        """ListMonad skip function

        Parameters
        ----------
        amount: int
            Amount of values to be skipped from the list for the next operation.

        Returns
        -------
        list: ListMonad[TypeSource]
            Skips an amount of values from the list for further execution.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 4]).skip(2)
        [3, 4]
        """
        return ListMonad(itertools.islice(self, amount, None))

    def apply(
        self: ListMonad[TypeSource],
        applicative: ListMonad[Callable[..., TypeResult]],
    ) -> ListMonad[TypeResult]:
        """ListMonad monad applicative interface for list monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: ListMonad[Callable[[TypeSource], TypeResult]]
            Applicative list monad which contains a function and will be applied to the list monad containing values.

        Returns
        -------
        list: ListMonad[TypeResult]
            Applies a list monad containing values of type TypeSource to an list monad containing a function
            of type Callable[[TypeSource], TypeResult].

        Examples
        --------
        >>> val: ListMonad[int] = ListMonad([1, 2])
        >>> func: ListMonad[Callable[[int], int]] = ListMonad([lambda x: x * 2])
        >>> val.apply(func)
        [2, 4]
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> ListMonad[TypeResult]:
            """Maps the applicative's function, curried, over this ListMonad's values."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: ListMonad[Callable[..., TypeResult]],
        applicative_value: ListMonad[Any],
    ) -> ListMonad[TypeResult]:
        """ListMonad monad applicative interface for list monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: ListMonad[TypePure]
            ListMonad monad value which will be applied to the list monad containing a function

        Returns
        -------
        list: ListMonad[TypeResult]
            Applies an list monad containing a function of type Callable[[TypePure], TypeResult]
            to a list monad of type TypePure (value or function).

        Examples
        --------
        >>> func: ListMonad[Callable[[int], int]] = ListMonad([lambda x: x * 2])
        >>> val: ListMonad[int] = ListMonad([1, 2])
        >>> func.apply2(val)
        [2, 4]
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> ListMonad[TypeResult]:
            """Maps the applicative value's values, curried, over this ListMonad's function."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def fold(
        self: ListMonad[TypeSource],
        initial: TypeResult,
        function: Callable[[TypeResult, TypeSource], TypeResult],
    ) -> TypeResult:
        """Left fold over the ListMonad with a seed value.

        Parameters
        ----------
        initial: TypeResult
            Seed value for the fold.
        function: Callable[[TypeResult, TypeSource], TypeResult]
            Function combining the accumulator with each element.

        Returns
        -------
        result: TypeResult
            Returns the final accumulator value.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).fold(0, lambda acc, x: acc + x)
        6
        """
        return reduce(function, self, initial)

    def reduce(
        self: ListMonad[TypeSource],
        function: Callable[[TypeSource, TypeSource], TypeSource],
    ) -> Option[TypeSource]:
        """Left fold over the ListMonad using its first element as the seed.

        Parameters
        ----------
        function: Callable[[TypeSource, TypeSource], TypeSource]
            Function combining the accumulator with each element.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the final accumulator value, or Nil if the ListMonad is empty.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).reduce(lambda acc, x: acc + x)
        Some(6)
        >>> ListMonad([]).reduce(lambda acc, x: acc + x)
        Nil()
        """
        if not self:
            return Nil()
        return Some(reduce(function, self))

    def enumerate(self: ListMonad[TypeSource]) -> ListMonad[Tuple[int, TypeSource]]:
        """Pairs each value with its index.

        Returns
        -------
        result: ListMonad[Tuple[int, TypeSource]]
            Returns a new ListMonad of (index, value) pairs.

        Examples
        --------
        >>> ListMonad(["a", "b"]).enumerate()
        [(0, 'a'), (1, 'b')]
        """
        return ListMonad(enumerate(self))

    def zip(
        self: ListMonad[TypeSource], other: Iterable[TypeResult]
    ) -> ListMonad[Tuple[TypeSource, TypeResult]]:
        """Pairs elements of this ListMonad with elements of another, stopping at the shorter one.

        Parameters
        ----------
        other: Iterable[TypeResult]
            Iterable to zip with this ListMonad.

        Returns
        -------
        result: ListMonad[Tuple[TypeSource, TypeResult]]
            Returns a new ListMonad of paired elements.

        Examples
        --------
        >>> ListMonad([1, 2]).zip(["a", "b"])
        [(1, 'a'), (2, 'b')]
        """
        return ListMonad(zip(self, other))

    def zip_with(
        self: ListMonad[TypeSource],
        other: Iterable[TypeResult],
        function: Callable[[TypeSource, TypeResult], TypePure],
    ) -> ListMonad[TypePure]:
        """Pairs elements of this ListMonad with another and combines them with a function.

        Parameters
        ----------
        other: Iterable[TypeResult]
            Iterable to zip with this ListMonad.
        function: Callable[[TypeSource, TypeResult], TypePure]
            Function combining each pair of elements.

        Returns
        -------
        result: ListMonad[TypePure]
            Returns a new ListMonad of combined elements.

        Examples
        --------
        >>> ListMonad([1, 2]).zip_with([10, 20], lambda a, b: a + b)
        [11, 22]
        """
        return ListMonad(function(a, b) for a, b in zip(self, other))

    def chain(
        self: ListMonad[TypeSource], other: Iterable[TypeSource]
    ) -> ListMonad[TypeSource]:
        """Concatenates this ListMonad with another iterable.

        Parameters
        ----------
        other: Iterable[TypeSource]
            Iterable to append to this ListMonad.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new ListMonad containing all elements of this ListMonad followed by `other`.

        Examples
        --------
        >>> ListMonad([1, 2]).chain([3, 4])
        [1, 2, 3, 4]
        """
        return ListMonad(chain(self, other))

    @overload
    def flatten(self: ListMonad[List[TypeResult]]) -> ListMonad[TypeResult]: ...

    @overload
    def flatten(
        self: ListMonad[ListMonad[TypeResult]],
    ) -> ListMonad[TypeResult]: ...

    def flatten(self) -> ListMonad[Any]:
        """Flattens one level of nesting.

        Returns
        -------
        result: ListMonad[TypeResult]
            Returns a new ListMonad with all nested elements concatenated.

        Examples
        --------
        >>> ListMonad([[1, 2], [3, 4]]).flatten()
        [1, 2, 3, 4]
        """
        return ListMonad(chain.from_iterable(cast(Iterable[Iterable[Any]], self)))

    def any(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> bool:
        """Returns True if any element matches the predicate.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: bool
            Returns True if any element matches, otherwise False.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).any(lambda x: x > 2)
        True
        >>> ListMonad([1, 2, 3]).any(lambda x: x > 5)
        False
        """
        return any(predicate(value) for value in self)

    def all(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> bool:
        """Returns True if all elements match the predicate.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: bool
            Returns True if all elements match, otherwise False.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).all(lambda x: x > 0)
        True
        >>> ListMonad([1, 2, 3]).all(lambda x: x > 2)
        False
        """
        return all(predicate(value) for value in self)

    def find(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Option[TypeSource]:
        """Returns the first element matching the predicate.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the first matching element, or Nil if none match.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).find(lambda x: x > 1)
        Some(2)
        >>> ListMonad([1, 2, 3]).find(lambda x: x > 5)
        Nil()
        """
        for value in self:
            if predicate(value):
                return Some(value)
        return Nil()

    def position(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Option[int]:
        """Returns the index of the first element matching the predicate.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Option[int]
            Returns Some of the index of the first matching element, or Nil if none match.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).position(lambda x: x > 1)
        Some(1)
        >>> ListMonad([1, 2, 3]).position(lambda x: x > 5)
        Nil()
        """
        for index, value in enumerate(self):
            if predicate(value):
                return Some(index)
        return Nil()

    def min(self: ListMonad[TypeOrd]) -> Option[TypeOrd]:
        """Returns the smallest element.

        Returns
        -------
        result: Option[TypeOrd]
            Returns Some of the smallest element, or Nil if the ListMonad is empty.

        Examples
        --------
        >>> ListMonad([3, 1, 2]).min()
        Some(1)
        >>> ListMonad([]).min()
        Nil()
        """
        if not self:
            return Nil()
        return Some(min(self))

    def max(self: ListMonad[TypeOrd]) -> Option[TypeOrd]:
        """Returns the largest element.

        Returns
        -------
        result: Option[TypeOrd]
            Returns Some of the largest element, or Nil if the ListMonad is empty.

        Examples
        --------
        >>> ListMonad([3, 1, 2]).max()
        Some(3)
        >>> ListMonad([]).max()
        Nil()
        """
        if not self:
            return Nil()
        return Some(max(self))

    def min_by_key(
        self: ListMonad[TypeSource], key_function: Callable[[TypeSource], TypeOrd]
    ) -> Option[TypeSource]:
        """Returns the element with the smallest key.

        Parameters
        ----------
        key_function: Callable[[TypeSource], TypeOrd]
            Function returning the value elements are compared by.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the element with the smallest key, or Nil if the ListMonad is empty.

        Examples
        --------
        >>> ListMonad(["aaa", "a", "aa"]).min_by_key(len)
        Some('a')
        >>> ListMonad([]).min_by_key(len)
        Nil()
        """
        if not self:
            return Nil()
        return Some(min(self, key=key_function))

    def max_by_key(
        self: ListMonad[TypeSource], key_function: Callable[[TypeSource], TypeOrd]
    ) -> Option[TypeSource]:
        """Returns the element with the largest key.

        Parameters
        ----------
        key_function: Callable[[TypeSource], TypeOrd]
            Function returning the value elements are compared by.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the element with the largest key, or Nil if the ListMonad is empty.

        Examples
        --------
        >>> ListMonad(["aaa", "a", "aa"]).max_by_key(len)
        Some('aaa')
        >>> ListMonad([]).max_by_key(len)
        Nil()
        """
        if not self:
            return Nil()
        return Some(max(self, key=key_function))

    def partition(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Tuple[ListMonad[TypeSource], ListMonad[TypeSource]]:
        """Splits the ListMonad into elements matching and not matching the predicate.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Tuple[ListMonad[TypeSource], ListMonad[TypeSource]]
            Returns a tuple of (matching, non_matching) ListMonads.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 4]).partition(lambda x: x % 2 == 0)
        ([2, 4], [1, 3])
        """
        matches: ListMonad[TypeSource] = ListMonad()
        non_matches: ListMonad[TypeSource] = ListMonad()
        for value in self:
            (matches if predicate(value) else non_matches).append(value)
        return matches, non_matches

    def take_while(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> ListMonad[TypeSource]:
        """Takes elements while the predicate holds, stopping at the first non-matching element.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new ListMonad of the leading matching elements.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 1]).take_while(lambda x: x < 3)
        [1, 2]
        """
        return ListMonad(itertools.takewhile(predicate, self))

    def skip_while(
        self: ListMonad[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> ListMonad[TypeSource]:
        """Skips elements while the predicate holds, keeping the remainder from the first non-matching element.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new ListMonad without the leading matching elements.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 1]).skip_while(lambda x: x < 3)
        [3, 1]
        """
        return ListMonad(itertools.dropwhile(predicate, self))

    def step_by(self: ListMonad[TypeSource], step: int) -> ListMonad[TypeSource]:
        """Returns every `step`-th element, starting with the first.

        Parameters
        ----------
        step: int
            Step size, must be at least 1.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new ListMonad of every `step`-th element.

        Examples
        --------
        >>> ListMonad([1, 2, 3, 4, 5]).step_by(2)
        [1, 3, 5]
        """
        if step < 1:
            raise ValueError("step must be at least 1")
        return ListMonad(itertools.islice(self, 0, None, step))

    def rev(self: ListMonad[TypeSource]) -> ListMonad[TypeSource]:
        """Returns a new ListMonad with the elements in reverse order.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new reversed ListMonad.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).rev()
        [3, 2, 1]
        """
        return ListMonad(reversed(self))

    def sum(self: ListMonad[TypeMonoid], initial: TypeMonoid) -> TypeMonoid:
        """Sums all elements, starting from an initial value.

        Parameters
        ----------
        initial: TypeMonoid
            Initial value to start the summation from.

        Returns
        -------
        result: TypeMonoid
            Returns the sum of `initial` and all elements.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).sum(0)
        6
        """
        return reduce(operator.add, self, initial)

    def dedup(self: ListMonad[TypeSource]) -> ListMonad[TypeSource]:
        """Removes consecutive duplicate elements, keeping the first of each run.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new ListMonad without consecutive duplicates.

        Examples
        --------
        >>> ListMonad([1, 1, 2, 2, 1]).dedup()
        [1, 2, 1]
        """
        return ListMonad(key for key, _ in itertools.groupby(self))

    def distinct(self: ListMonad[TypeHashable]) -> ListMonad[TypeHashable]:
        """Removes all duplicate elements, preserving first-occurrence order.

        Requires elements to be hashable.

        Returns
        -------
        result: ListMonad[TypeHashable]
            Returns a new ListMonad without any duplicates.

        Examples
        --------
        >>> ListMonad([1, 2, 1, 3, 2]).distinct()
        [1, 2, 3]
        """
        seen: set[TypeHashable] = set()
        result: ListMonad[TypeHashable] = ListMonad()
        for value in self:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def sorted(self: ListMonad[TypeOrd], reverse: bool = False) -> ListMonad[TypeOrd]:
        """Returns a new sorted ListMonad.

        Parameters
        ----------
        reverse: bool
            Sort in descending order if True.

        Returns
        -------
        result: ListMonad[TypeOrd]
            Returns a new sorted ListMonad.

        Examples
        --------
        >>> ListMonad([3, 1, 2]).sorted()
        [1, 2, 3]
        >>> ListMonad([3, 1, 2]).sorted(reverse=True)
        [3, 2, 1]
        """
        return ListMonad(sorted(self, reverse=reverse))

    def sort_by(
        self: ListMonad[TypeSource],
        key_function: Callable[[TypeSource], TypeOrd],
        reverse: bool = False,
    ) -> ListMonad[TypeSource]:
        """Returns a new ListMonad sorted by a key function.

        Parameters
        ----------
        key_function: Callable[[TypeSource], TypeOrd]
            Function returning the value elements are sorted by.
        reverse: bool
            Sort in descending order if True.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new sorted ListMonad.

        Examples
        --------
        >>> ListMonad(["aaa", "a", "aa"]).sort_by(len)
        ['a', 'aa', 'aaa']
        """
        return ListMonad(sorted(self, key=key_function, reverse=reverse))

    def to_list(self: ListMonad[TypeSource]) -> List[TypeSource]:
        """Converts the ListMonad into a plain built-in list.

        Returns
        -------
        result: List[TypeSource]
            Returns a plain `list` containing the same elements.

        Examples
        --------
        >>> ListMonad([1, 2, 3]).to_list()
        [1, 2, 3]
        """
        return list(self)

    def __str__(self: ListMonad[TypeSource]) -> str:
        """Returns the string representation of the ListMonad.

        Examples
        --------
        >>> str(ListMonad([1, 2, 3]))
        'ListMonad([1, 2, 3])'
        """
        return f"ListMonad({super().__str__()})"
