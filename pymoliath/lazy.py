from __future__ import annotations

import itertools
import operator
from functools import reduce
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from pymoliath.list import ListMonad, TypeHashable, TypeMonoid, TypeOrd
from pymoliath.option import Nil, Option, Some
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class LazyMonad(Generic[TypeSource]):
    __slots__ = ("_computation",)

    _computation: Callable[[], TypeSource]

    def __init__(self, value: Union[TypeSource, Callable[[], TypeSource]]):
        """Lazy monad constructor which takes either a value of type TypeSource or a callable which
        must return a value of type TypeSource.

        Parameters
        ----------
        value
        """
        if isinstance(value, Callable):
            self._computation = value
        else:
            self._computation = lambda: value

    def map(
        self: LazyMonad[TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> LazyMonad[TypeResult]:
        """Lazy monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        io: LazyMonad[TypeResult]
            Returns a new lazy monad containing the result of the passed function and the io call.
        """
        return LazyMonad(lambda: function(self.run()))

    def bind(
        self: LazyMonad[TypeSource],
        function: Callable[[TypeSource], LazyMonad[TypeResult]],
    ) -> LazyMonad[TypeResult]:
        """Lazy monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], LazyMonad[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Returns an lazy monad with the function call result
        """
        return function(self.run())

    def apply(
        self: LazyMonad[TypeSource],
        applicative: LazyMonad[Callable[..., TypeResult]],
    ) -> LazyMonad[TypeResult]:
        """LazyMonad monad applicative interface for lazy monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: LazyMonad[Callable[[TypeSource], TypeResult]]
            Applicative io monad which contains a function and will be applied to the io monad containing a value.

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Applies an lazy monad containing a value of type TypeSource to an lazy monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> LazyMonad[TypeResult]:
            """Maps the applicative's function, curried, over this LazyMonad's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: LazyMonad[Callable[..., TypeResult]],
        applicative_value: LazyMonad[Any],
    ) -> LazyMonad[TypeResult]:
        """LazyMonad monad applicative interface for lazy monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: LazyMonad[TypePure]
            LazyMonad monad value which will be applied to the lazy monad containing a function

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Applies an lazy monad containing a function of type Callable[[TypePure], TypeResult]
            to an lazy monad of type TypePure (value or function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> LazyMonad[TypeResult]:
            """Maps the applicative value's value, curried, over this LazyMonad's function."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def run(self) -> TypeSource:
        """Lazy monad lazy run function to start the computation.

        Returns
        -------
        result: TypeSource
            Calls the lazy monad function which returns a value of type TypeSource.
        """
        return self._computation()

    def __str__(self) -> str:
        """Returns the string representation of the LazyMonad."""
        return f"LazyMonad({self._computation})"

    def __repr__(self) -> str:
        """Returns the string representation of the LazyMonad."""
        return str(self)


class Sequence(Generic[TypeSource]):
    """Lazy sequence evaluation.

    Every intermediate operation (map/bind/filter/take/skip/enumerate/zip/chain/flatten/take_while/
    skip_while/step_by/dedup/distinct) builds a new pipeline without consuming the source. The
    pipeline is only ever pulled from element by element once a terminal operation
    (run/fold/reduce/any/all/find/position/min/max/sum/count/partition/rev/sorted/sort_by, or
    iterating the Sequence directly) is called - so predicate based terminal operations
    short-circuit and infinite sources are supported as long as a short-circuiting terminal
    operation is used.

    Note: constructing a Sequence directly from a raw generator/iterator (rather than from a list
    or a callable returning a fresh iterable) makes the Sequence single-use, since the underlying
    generator is exhausted the first time it is consumed - a second terminal call would then see
    an empty source.

    Note: consecutive .map()/.filter() calls are deliberately left as nested built-in map/filter
    iterators rather than fused into one composed function. Benchmarked: CPython's C-level
    map/filter iterator step is cheaper than the Python-level closure calls a fused composition
    would add, so fusing is consistently slower (and increasingly so with chain depth) - not
    just unnecessary complexity.
    """

    __slots__ = ("_pipeline",)

    def __init__(
        self, value: Union[Iterable[TypeSource], Callable[[], Iterable[TypeSource]]]
    ):
        """Lazy Sequence Monad constructor/unit function

        Parameters
        ----------
        value: Union[Iterable[TypeSource], Callable[[], Iterable[TypeSource]]]
            Single value or type TypeSource or callable returning a value of type TypeSource
        """
        if isinstance(value, Callable):
            self._pipeline: Callable[[], Iterator[TypeSource]] = lambda: iter(value())
        else:
            self._pipeline = lambda: iter(value)

    def __iter__(self: Sequence[TypeSource]) -> Iterator[TypeSource]:
        """Returns a fresh iterator over the Sequence's pipeline, pulling from the source."""
        return self._pipeline()

    def map(
        self: Sequence[TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> Sequence[TypeResult]:
        """Sequence Monad map function

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function to be evaluated lazy

        Returns
        -------
        sequence: Sequence[TypeResult]
            Returns a new sequence monad containing the resulting value
        """
        return Sequence(lambda: map(function, self._pipeline()))

    def bind(
        self: Sequence[TypeSource],
        function: Callable[[TypeSource], Sequence[TypeResult]],
    ) -> Sequence[TypeResult]:
        """Sequence Monad bind function

        Parameters
        ----------
        function: Callable[[TypeSource], Sequence[TypeResult]]
            Function to be evaluated lazy

        Returns
        -------
        sequence: Sequence[TypeResult]
            Returns the new sequence monad from the bind function
        """

        def generator() -> Iterator[TypeResult]:
            """Flat-maps each value from the source pipeline through `function`."""
            for value in self._pipeline():
                yield from function(value)

        return Sequence(generator)

    def filter(
        self: Sequence[TypeSource], filter_function: Callable[[TypeSource], bool]
    ) -> Sequence[TypeSource]:
        """Sequence filter function

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            Filter function for the list

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a filtered sequence monad
        """
        return Sequence(lambda: filter(filter_function, self._pipeline()))

    def take(self: Sequence[TypeSource], amount: int) -> Sequence[TypeSource]:
        """Sequence monad take function

        Parameters
        ----------
        amount: int
            Amount of values to be taken from the list for the next operation.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Takes our only an specific amount of values from the list for further execution.
        """
        return Sequence(lambda: itertools.islice(self._pipeline(), amount))

    def skip(self: Sequence[TypeSource], amount: int) -> Sequence[TypeSource]:
        """Sequence monad skip function

        Parameters
        ----------
        amount: int
            Amount of values to be skipped from the list for the next operation.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Skips an amount of values from the list for further execution.
        """
        return Sequence(lambda: itertools.islice(self._pipeline(), amount, None))

    def enumerate(self: Sequence[TypeSource]) -> Sequence[Tuple[int, TypeSource]]:
        """Pairs each value with its index, lazily.

        Returns
        -------
        sequence: Sequence[Tuple[int, TypeSource]]
            Returns a new Sequence of (index, value) pairs.
        """
        return Sequence(lambda: enumerate(self._pipeline()))

    def zip(
        self: Sequence[TypeSource], other: Iterable[TypeResult]
    ) -> Sequence[Tuple[TypeSource, TypeResult]]:
        """Pairs elements of this Sequence with elements of another, lazily, stopping at the shorter one.

        Parameters
        ----------
        other: Iterable[TypeResult]
            Iterable to zip with this Sequence.

        Returns
        -------
        sequence: Sequence[Tuple[TypeSource, TypeResult]]
            Returns a new Sequence of paired elements.
        """
        return Sequence(lambda: zip(self._pipeline(), other))

    def zip_with(
        self: Sequence[TypeSource],
        other: Iterable[TypeResult],
        function: Callable[[TypeSource, TypeResult], TypePure],
    ) -> Sequence[TypePure]:
        """Pairs elements of this Sequence with another, lazily, and combines them with a function.

        Parameters
        ----------
        other: Iterable[TypeResult]
            Iterable to zip with this Sequence.
        function: Callable[[TypeSource, TypeResult], TypePure]
            Function combining each pair of elements.

        Returns
        -------
        sequence: Sequence[TypePure]
            Returns a new Sequence of combined elements.
        """
        return Sequence(
            lambda: (function(a, b) for a, b in zip(self._pipeline(), other))
        )

    def chain(
        self: Sequence[TypeSource], other: Iterable[TypeSource]
    ) -> Sequence[TypeSource]:
        """Concatenates this Sequence with another iterable, lazily.

        Parameters
        ----------
        other: Iterable[TypeSource]
            Iterable to append to this Sequence.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a new Sequence containing all elements of this Sequence followed by `other`.
        """
        return Sequence(lambda: itertools.chain(self._pipeline(), other))

    @overload
    def flatten(self: Sequence[List[TypeResult]]) -> Sequence[TypeResult]: ...

    @overload
    def flatten(self: Sequence[Sequence[TypeResult]]) -> Sequence[TypeResult]: ...

    def flatten(self) -> Sequence[Any]:
        """Flattens one level of nesting, lazily.

        Returns
        -------
        sequence: Sequence[TypeResult]
            Returns a new Sequence with all nested elements concatenated.
        """
        return Sequence(
            lambda: itertools.chain.from_iterable(
                cast(Iterable[Iterable[Any]], self._pipeline())
            )
        )

    def take_while(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Sequence[TypeSource]:
        """Takes elements while the predicate holds, lazily, stopping at the first non-matching element.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a new Sequence of the leading matching elements.
        """
        return Sequence(lambda: itertools.takewhile(predicate, self._pipeline()))

    def skip_while(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Sequence[TypeSource]:
        """Skips elements while the predicate holds, lazily, keeping the remainder.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a new Sequence without the leading matching elements.
        """
        return Sequence(lambda: itertools.dropwhile(predicate, self._pipeline()))

    def step_by(self: Sequence[TypeSource], step: int) -> Sequence[TypeSource]:
        """Returns every `step`-th element, lazily, starting with the first.

        Parameters
        ----------
        step: int
            Step size, must be at least 1.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a new Sequence of every `step`-th element.
        """
        if step < 1:
            raise ValueError("step must be at least 1")
        return Sequence(lambda: itertools.islice(self._pipeline(), 0, None, step))

    def dedup(self: Sequence[TypeSource]) -> Sequence[TypeSource]:
        """Removes consecutive duplicate elements, lazily, keeping the first of each run.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a new Sequence without consecutive duplicates.
        """

        def generator() -> Iterator[TypeSource]:
            """Yields the first element of each run of consecutive duplicates."""
            for key, _ in itertools.groupby(self._pipeline()):
                yield key

        return Sequence(generator)

    def distinct(self: Sequence[TypeHashable]) -> Sequence[TypeHashable]:
        """Removes all duplicate elements, lazily, preserving first-occurrence order.

        Requires elements to be hashable.

        Returns
        -------
        sequence: Sequence[TypeHashable]
            Returns a new Sequence without any duplicates.
        """

        def generator() -> Iterator[TypeHashable]:
            """Yields each value from the source pipeline only the first time it is seen."""
            seen: set[TypeHashable] = set()
            for value in self._pipeline():
                if value not in seen:
                    seen.add(value)
                    yield value

        return Sequence(generator)

    def apply(
        self: Sequence[TypeSource],
        applicative: Sequence[Callable[..., TypeResult]],
    ) -> Sequence[TypeResult]:
        """Sequence monad applicative interface for sequence monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: Sequence[Callable[[TypeSource], TypeResult]]
            Applicative list monad which contains a function and will be applied to the list monad containing values.

        Returns
        -------
        sequence: Sequence[TypeResult]
            Applies a sequnece monad containing values of type TypeSource to an sequence monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Sequence[TypeResult]:
            """Maps the applicative's function, curried, over this Sequence's values."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Sequence[Callable[..., TypeResult]],
        applicative_value: Sequence[Any],
    ) -> Sequence[TypeResult]:
        """Sequence monad applicative interface for sequence monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Sequence[TypePure]
            Sequence monad value which will be applied to the sequence monad containing a function

        Returns
        -------
        sequence: Sequence[TypeResult]
            Applies an sequence monad containing a function of type Callable[[TypePure], TypeResult]
            to a sequence monad of type TypePure (value or function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Sequence[TypeResult]:
            """Maps the applicative value's values, curried, over this Sequence's function."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def fold(
        self: Sequence[TypeSource],
        initial: TypeResult,
        function: Callable[[TypeResult, TypeSource], TypeResult],
    ) -> TypeResult:
        """Left fold over the Sequence with a seed value. Terminal operation.

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
        """
        return reduce(function, self._pipeline(), initial)

    def reduce(
        self: Sequence[TypeSource],
        function: Callable[[TypeSource, TypeSource], TypeSource],
    ) -> Option[TypeSource]:
        """Left fold over the Sequence using its first element as the seed. Terminal operation.

        Parameters
        ----------
        function: Callable[[TypeSource, TypeSource], TypeSource]
            Function combining the accumulator with each element.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the final accumulator value, or Nil if the Sequence is empty.
        """
        pipeline = self._pipeline()
        try:
            first = next(pipeline)
        except StopIteration:
            return Nil()
        return Some(reduce(function, pipeline, first))

    def any(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> bool:
        """Returns True if any element matches the predicate. Terminal, short-circuiting operation.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: bool
            Returns True if any element matches, otherwise False.
        """
        return any(predicate(value) for value in self._pipeline())

    def all(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> bool:
        """Returns True if all elements match the predicate. Terminal, short-circuiting operation.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: bool
            Returns True if all elements match, otherwise False.
        """
        return all(predicate(value) for value in self._pipeline())

    def find(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Option[TypeSource]:
        """Returns the first element matching the predicate. Terminal, short-circuiting operation.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the first matching element, or Nil if none match.
        """
        for value in self._pipeline():
            if predicate(value):
                return Some(value)
        return Nil()

    def position(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Option[int]:
        """Returns the index of the first element matching the predicate. Terminal, short-circuiting operation.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Option[int]
            Returns Some of the index of the first matching element, or Nil if none match.
        """
        for index, value in enumerate(self._pipeline()):
            if predicate(value):
                return Some(index)
        return Nil()

    def min(self: Sequence[TypeOrd]) -> Option[TypeOrd]:
        """Returns the smallest element. Terminal operation.

        Returns
        -------
        result: Option[TypeOrd]
            Returns Some of the smallest element, or Nil if the Sequence is empty.
        """
        values = list(self._pipeline())
        if not values:
            return Nil()
        return Some(min(values))

    def max(self: Sequence[TypeOrd]) -> Option[TypeOrd]:
        """Returns the largest element. Terminal operation.

        Returns
        -------
        result: Option[TypeOrd]
            Returns Some of the largest element, or Nil if the Sequence is empty.
        """
        values = list(self._pipeline())
        if not values:
            return Nil()
        return Some(max(values))

    def min_by_key(
        self: Sequence[TypeSource], key_function: Callable[[TypeSource], TypeOrd]
    ) -> Option[TypeSource]:
        """Returns the element with the smallest key. Terminal operation.

        Parameters
        ----------
        key_function: Callable[[TypeSource], TypeOrd]
            Function returning the value elements are compared by.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the element with the smallest key, or Nil if the Sequence is empty.
        """
        values = list(self._pipeline())
        if not values:
            return Nil()
        return Some(min(values, key=key_function))

    def max_by_key(
        self: Sequence[TypeSource], key_function: Callable[[TypeSource], TypeOrd]
    ) -> Option[TypeSource]:
        """Returns the element with the largest key. Terminal operation.

        Parameters
        ----------
        key_function: Callable[[TypeSource], TypeOrd]
            Function returning the value elements are compared by.

        Returns
        -------
        result: Option[TypeSource]
            Returns Some of the element with the largest key, or Nil if the Sequence is empty.
        """
        values = list(self._pipeline())
        if not values:
            return Nil()
        return Some(max(values, key=key_function))

    def sum(self: Sequence[TypeMonoid], initial: TypeMonoid) -> TypeMonoid:
        """Sums all elements, starting from an initial value. Terminal operation.

        Parameters
        ----------
        initial: TypeMonoid
            Initial value to start the summation from.

        Returns
        -------
        result: TypeMonoid
            Returns the sum of `initial` and all elements.
        """
        return reduce(operator.add, self._pipeline(), initial)

    def count(self: Sequence[TypeSource]) -> int:
        """Counts the number of elements. Terminal operation.

        Returns
        -------
        result: int
            Returns the number of elements in the Sequence.
        """
        return sum(1 for _ in self._pipeline())

    def partition(
        self: Sequence[TypeSource], predicate: Callable[[TypeSource], bool]
    ) -> Tuple[ListMonad[TypeSource], ListMonad[TypeSource]]:
        """Splits the Sequence into elements matching and not matching the predicate. Terminal operation.

        Parameters
        ----------
        predicate: Callable[[TypeSource], bool]
            Predicate function.

        Returns
        -------
        result: Tuple[ListMonad[TypeSource], ListMonad[TypeSource]]
            Returns a tuple of (matching, non_matching) ListMonads.
        """
        matches: ListMonad[TypeSource] = ListMonad()
        non_matches: ListMonad[TypeSource] = ListMonad()
        for value in self._pipeline():
            (matches if predicate(value) else non_matches).append(value)
        return matches, non_matches

    def rev(self: Sequence[TypeSource]) -> ListMonad[TypeSource]:
        """Materializes the Sequence and returns it in reverse order. Terminal operation.

        Note: this must fully consume the Sequence first, so it will hang on an infinite source.

        Returns
        -------
        result: ListMonad[TypeSource]
            Returns a new reversed ListMonad.
        """
        return ListMonad(reversed(list(self._pipeline())))

    def sorted(self: Sequence[TypeOrd], reverse: bool = False) -> ListMonad[TypeOrd]:
        """Materializes the Sequence and returns it sorted. Terminal operation.

        Note: this must fully consume the Sequence first, so it will hang on an infinite source.

        Parameters
        ----------
        reverse: bool
            Sort in descending order if True.

        Returns
        -------
        result: ListMonad[TypeOrd]
            Returns a new sorted ListMonad.
        """
        return ListMonad(sorted(self._pipeline(), reverse=reverse))

    def sort_by(
        self: Sequence[TypeSource],
        key_function: Callable[[TypeSource], TypeOrd],
        reverse: bool = False,
    ) -> ListMonad[TypeSource]:
        """Materializes the Sequence and returns it sorted by a key function. Terminal operation.

        Note: this must fully consume the Sequence first, so it will hang on an infinite source.

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
        """
        return ListMonad(sorted(self._pipeline(), key=key_function, reverse=reverse))

    def run(self) -> List[TypeSource]:
        """Sequence monad lazy evaluation function. Terminal operation.

        Returns
        -------
        result: List[TypeSource]
            Returns a list of the lazy evaluated result
        """
        return list(self._pipeline())

    def __str__(self) -> str:
        """Returns the string representation of the Sequence."""
        return f"Sequence({self._pipeline})"

    def __repr__(self) -> str:
        """Returns the string representation of the Sequence."""
        return str(self)
