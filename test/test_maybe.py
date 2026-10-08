import copy
import math
import pickle
import unittest
from decimal import Decimal
from typing import Any, Callable
from unittest.mock import MagicMock, Mock

from pymoliath.either import Either, Left, Right
from pymoliath.errors import UnwrapError
from pymoliath.maybe import (
    Just,
    Maybe,
    Nothing,
)
from pymoliath.util import compose


class TestMaybe(unittest.TestCase):
    """
    Monad operations:
    ≡       Identical to
    >>=     bind, flatMap
    (.)     Function composition
    <*>     Applicative functor: (<$>) :: (Functor f) => (a -> b) -> f a -> f b
    <$>     Applicative functor: (<*>) :: f (a -> b) -> f a -> f b
    """

    def test_monad_left_identity_law(self):
        """Left identity law: return a >>= f ≡ f a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Left identity: The first monad law states that if we take a value,
        put it in a default context with return and then feed it to a function by using >>=,
        it’s the same as just taking the value and applying the function to it.
        """

        def just_function(_):
            return Just(10)

        def nothing_function(_):
            return Nothing()

        self.assertEqual(just_function(10), Just(10).bind(just_function))
        self.assertEqual(nothing_function(10), Nothing().bind(nothing_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        just_value = Just("Hello")
        nothing_value = Nothing()

        self.assertEqual(just_value, just_value.bind(lambda x: Just(x)))
        self.assertEqual(nothing_value, nothing_value.bind(lambda _: Nothing()))

    def test_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        just_value = Just(42)

        def f(x: int) -> Maybe[int]:
            return Just(x + 1000)

        def g(y: int) -> Maybe[int]:
            return Just(y * 42)

        self.assertEqual(
            just_value.bind(f).bind(g), just_value.bind(lambda x: f(x).bind(g))
        )

        nothing_value = Nothing()

        def h(_):
            return Nothing()

        def i(_):
            return Nothing()

        self.assertEqual(
            nothing_value.bind(h).bind(i), nothing_value.bind(lambda x: h(x).bind(i))
        )

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Just(10).map(lambda x: x), Just(10))
        self.assertEqual(Nothing().map(lambda x: x), Nothing())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        just_value = Just(42)
        nothing_value = Nothing()

        def f(x: int) -> int:
            return x + 1000

        def g(y: int) -> int:
            return y * 42

        self.assertEqual(just_value.map(compose(f, g)), just_value.map(g).map(f))
        self.assertEqual(nothing_value.map(compose(f, g)), nothing_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        just_value = Just(42)
        nothing_value = Nothing()

        self.assertEqual(just_value.apply(Just(lambda x: x)), just_value)
        self.assertEqual(nothing_value.apply(Just(lambda x: x)), nothing_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42

        def f(x: int) -> int:
            return x * 42

        self.assertEqual(Just(x).apply(Just(f)), Just(f(x)))
        self.assertEqual(Nothing().apply(Just(f)), Nothing())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """

        def composition(
            f: Callable[[int], int],
        ) -> Callable[[Callable[[int], int]], Callable[[int], int]]:
            return lambda g: lambda x: f(g(x))

        w: Maybe[int] = Just(42)
        u: Maybe[Callable[[int], int]] = Just(lambda x: x + 42)
        v: Maybe[Callable[[int], int]] = Just(lambda x: x * 42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Just(composition)))), w.apply(v).apply(u)
        )

        u = Nothing()
        v = Nothing()
        self.assertEqual(
            w.apply(v.apply(u.apply(Just(composition)))), w.apply(v).apply(u)
        )

    def test_str(self):
        just = Just("a")
        nothing = Nothing()

        self.assertEqual(str(just), "Just(a)")
        self.assertEqual(str(nothing), "Nothing()")

    def test_variants_are_instances(self):
        self.assertTrue(isinstance(Just("a"), (Just, Nothing)))  # pyright: ignore[reportUnnecessaryIsInstance]
        self.assertTrue(isinstance(Nothing(), (Just, Nothing)))  # pyright: ignore[reportUnnecessaryIsInstance]

    def test_nothing_is_singleton(self):
        self.assertIs(Nothing(), Nothing())
        empty: Maybe[int] = Nothing()
        self.assertIs(empty, Nothing())

    def test_from_and_to_optional(self):
        maybe_dict: Maybe[dict[Any, Any]] = Just.from_optional({})
        maybe_string: Maybe[str] = Just.from_optional("")
        maybe_none: Maybe[str] = Just.from_optional(None)

        self.assertEqual({}, maybe_dict.to_optional())
        self.assertEqual("", maybe_string.to_optional())
        self.assertEqual(None, maybe_none.to_optional())

    def test_is_nothing_and_is_just(self):
        just = Just(10)
        nothing = Nothing()
        maybe_none = Just.from_optional(None)
        maybe_value = Just.from_optional(10)

        self.assertTrue(just.is_just() and not just.is_nothing())
        self.assertTrue(nothing.is_nothing() and not nothing.is_just())
        self.assertTrue(maybe_value.is_just() and not maybe_value.is_nothing())
        self.assertTrue(maybe_none.is_nothing() and not maybe_none.is_just())

    def test_unwrap_or(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(10, just_value.unwrap_or(20))
        self.assertEqual(20, nothing.unwrap_or(20))

    def test_filter(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(Just(10), just_value.filter(lambda v: v > 5))
        self.assertEqual(Nothing(), just_value.filter(lambda v: v > 10))
        self.assertEqual(Nothing(), nothing.filter(lambda v: v < 10))

    def test_is_just_and(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertTrue(just_value.is_just_and(lambda v: v > 5))
        self.assertFalse(just_value.is_just_and(lambda v: v > 10))
        self.assertFalse(nothing.is_just_and(lambda v: v > 5))

    def test_map_or(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(11, just_value.map_or(0, lambda v: v + 1))
        self.assertEqual(0, nothing.map_or(0, lambda v: v + 1))

    def test_and_or(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(Just(20), just_value.and_(Just(20)))
        self.assertEqual(Nothing(), nothing.and_(Just(20)))
        self.assertEqual(just_value, just_value.or_(Just(20)))
        self.assertEqual(Just(20), nothing.or_(Just(20)))

    def test_zip(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(Just((10, "a")), just_value.zip(Just("a")))
        self.assertEqual(Nothing(), just_value.zip(Nothing()))
        self.assertEqual(Nothing(), nothing.zip(Just("a")))

    def test_flatten(self):
        self.assertEqual(Just(10), Just(Just(10)).flatten())
        self.assertEqual(Nothing(), Just(Nothing()).flatten())
        self.assertEqual(Nothing(), Nothing().flatten())

    def test_right_or(self):
        just_value = Just(10)
        nothing_value = Nothing()

        self.assertEqual(Right(10), just_value.right_or("error"))
        self.assertEqual(Left("error"), nothing_value.right_or("error"))
        self.assertEqual(Right(10), just_value.right_or_else(lambda: "error"))
        self.assertEqual(Left("error"), nothing_value.right_or_else(lambda: "error"))

    def test_safe(self):
        exception_function = MagicMock(side_effect=Exception("error"))
        maybe_unsafe = Just.safe(lambda: exception_function())
        maybe_safe = Just.safe(lambda: 10)

        self.assertEqual(Nothing(), maybe_unsafe)
        self.assertEqual(Just(10), maybe_safe)
        self.assertEqual(
            Nothing(), Just.safe(lambda: int("x"), exceptions=(ValueError,))
        )
        with self.assertRaises(KeyError):
            Just.safe(lambda: {}["missing"], exceptions=(ValueError,))

    def test_unwrap(self):
        just = Just("a")
        nothing = Nothing()

        with self.assertRaises(UnwrapError):
            nothing.unwrap()

        self.assertEqual("a", just.unwrap())
        self.assertEqual("a", just.unwrap_or("b"))
        self.assertEqual(10, nothing.unwrap_or(10))
        self.assertEqual("a", just.unwrap_or_else(lambda: "b"))
        self.assertEqual(10, nothing.unwrap_or_else(lambda: 10))

    def test_inspect(self):
        just = Just("a")
        nothing = Nothing()
        print_mock = Mock()

        self.assertEqual(just, just.inspect(print_mock))
        self.assertEqual(nothing, nothing.inspect(print_mock))
        print_mock.assert_called_once_with("a")

    def test_match(self):
        just = Just("a")
        nothing = Nothing()

        self.assertEqual("a", just.match(just=lambda x: x, nothing=lambda: "default"))
        self.assertEqual(
            "default",
            just.bind(lambda x: Nothing()).match(
                just=lambda x: x, nothing=lambda: "default"
            ),
        )
        self.assertEqual(
            "default", nothing.match(just=lambda x: x, nothing=lambda: "default")
        )
        self.assertEqual(
            "default",
            nothing.bind(lambda x: Just(x)).match(
                just=lambda x: x, nothing=lambda: "default"
            ),
        )

    def test_supports_structural_pattern_matching(self):
        def describe(value: Maybe[int]) -> str:
            # No `case _:` fallback: Maybe is a closed union (Just[T] | Nothing),
            # so this is statically exhaustive without one.
            match value:
                case Just(x):
                    return f"just {x}"
                case Nothing():
                    return "nothing"

        self.assertEqual("just 10", describe(Just(10)))
        self.assertEqual("nothing", describe(Nothing()))

    def test_apply2(self):
        def increment(x: int) -> int:
            return x + 1

        function: Maybe[Callable[[int], int]] = Just(increment)
        failed: Maybe[Callable[[int], int]] = Nothing()
        value: Maybe[int] = Just(1)
        error: Maybe[int] = Nothing()

        self.assertEqual(Just(2), function.apply2(value))
        self.assertEqual(Nothing(), function.apply2(error))
        self.assertEqual(Nothing(), failed.apply2(error))
        self.assertEqual(value.apply(function), function.apply2(value))

        add: Maybe[Callable[[int], Callable[[int], int]]] = Just(
            lambda a: lambda b: a + b
        )
        self.assertEqual(Just(3), add.apply2(Just(1)).apply2(Just(2)))


class TestMaybeValueSemantics(unittest.TestCase):
    def test_equality_and_hash(self):
        self.assertEqual(Just({"a": 1, "b": 2}), Just({"b": 2, "a": 1}))
        self.assertNotEqual(Just(1), Just("1"))
        self.assertNotEqual(Just(1), Nothing())
        self.assertEqual(1, len({Just(1), Just(1)}))
        self.assertEqual(1, len({Nothing(), Nothing()}))
        with self.assertRaises(TypeError):
            hash(Just([]))

    def test_repr(self):
        self.assertEqual("Just('1')", repr(Just("1")))
        self.assertEqual("Just(1)", str(Just(1)))
        self.assertEqual("Nothing()", repr(Nothing()))


class TestMaybeFeatures(unittest.TestCase):
    def test_map_or_else(self):
        empty: Maybe[int] = Nothing()
        self.assertEqual(0, empty.map_or_else(lambda: 0, lambda x: x * 2))
        self.assertEqual(4, Just(2).map_or_else(lambda: 0, lambda x: x * 2))

    def test_xor(self):
        one: Maybe[int] = Just(1)
        empty: Maybe[int] = Nothing()
        self.assertEqual(Just(1), one.xor(empty))
        self.assertEqual(Just(1), empty.xor(one))
        self.assertEqual(Nothing(), one.xor(Just(2)))

    def test_aliases(self):
        empty: Maybe[int] = Nothing()
        self.assertEqual(Just(2), Just(1).and_then(lambda x: Just(x + 1)))
        self.assertEqual(Just(1), empty.or_else(lambda: Just(1)))

    def test_transpose(self):
        self.assertEqual(Right(Just(1)), Just(Right(1)).transpose())
        self.assertEqual(Right(Nothing()), Nothing().transpose())
        error: Maybe[Either[str, int]] = Just(Left("e"))
        self.assertEqual(Left("e"), error.transpose())


class TestMaybeCopyAndPickle(unittest.TestCase):
    def test_just_round_trips(self):
        just = Just([1, 2])
        for clone in (
            copy.copy(just),
            copy.deepcopy(just),
            pickle.loads(pickle.dumps(just)),
        ):
            self.assertEqual(just, clone)
        self.assertIsNot(just.value, copy.deepcopy(just).value)

    def test_nothing_stays_a_singleton(self):
        for clone in (
            copy.copy(Nothing()),
            copy.deepcopy(Nothing()),
            pickle.loads(pickle.dumps(Nothing())),
        ):
            self.assertIs(Nothing(), clone)


class TestMaybeMoreValueSemantics(unittest.TestCase):
    def test_nan_and_decimal_equality(self):
        self.assertNotEqual(Just(float("nan")), Just(float("nan")))
        self.assertTrue(math.isnan(Just(float("nan")).unwrap()))
        self.assertEqual(Just(Decimal("1.0")), Just(Decimal("1.00")))

    def test_variants_are_final(self):
        self.assertTrue(getattr(Just, "__final__", False))
        self.assertTrue(getattr(Nothing, "__final__", False))

    def test_from_optional(self):
        self.assertIs(Nothing(), Just.from_optional(None))
        self.assertEqual(Just(0), Just.from_optional(0))

    def test_keyword_arguments(self):
        empty: Maybe[int] = Nothing()
        self.assertEqual(Just(2), Just(2).filter(predicate=lambda x: x > 1))
        self.assertEqual(0, empty.unwrap_or_else(function=lambda: 0))
        self.assertEqual(Left("missing"), empty.right_or(left_value="missing"))
        self.assertEqual(
            Left("missing"), empty.right_or_else(function=lambda: "missing")
        )
