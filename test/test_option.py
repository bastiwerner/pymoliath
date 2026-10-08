import unittest
from typing import Any, Callable
from unittest.mock import MagicMock, Mock

from pymoliath import option
from pymoliath.errors import UnwrapError
from pymoliath.option import (
    OPTION_TYPES,
    Nil,
    Option,
    Some,
    from_optional,
    is_nil,
    is_option,
    is_some,
    map2,
    map3,
    safe,
)
from pymoliath.result import Err, Ok, Result
from pymoliath.util import compose, flow


class TestOption(unittest.TestCase):
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
        it’s the same as some taking the value and applying the function to it.
        """

        def some_function(_):
            return Some(10)

        def nothing_function(_):
            return Nil()

        self.assertEqual(some_function(10), Some(10).bind(some_function))
        self.assertEqual(nothing_function(10), Nil().bind(nothing_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        some_value = Some("Hello")
        nothing_value = Nil()

        self.assertEqual(some_value, some_value.bind(lambda x: Some(x)))
        self.assertEqual(nothing_value, nothing_value.bind(lambda _: Nil()))

    def test_some_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        some_value = Some(42)

        def f(x: int) -> Option[int]:
            return Some(x + 1000)

        def g(y: int) -> Option[int]:
            return Some(y * 42)

        self.assertEqual(
            some_value.bind(f).bind(g), some_value.bind(lambda x: f(x).bind(g))
        )

        nothing_value = Nil()

        def h(_):
            return Nil()

        def i(_):
            return Nil()

        self.assertEqual(
            nothing_value.bind(h).bind(i), nothing_value.bind(lambda x: h(x).bind(i))
        )

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Some(10).map(lambda x: x), Some(10))
        self.assertEqual(Nil().map(lambda x: x), Nil())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        some_value = Some(42)
        nothing_value = Nil()

        def f(x: int) -> int:
            return x + 1000

        def g(y: int) -> int:
            return y * 42

        self.assertEqual(some_value.map(compose(f, g)), some_value.map(g).map(f))
        self.assertEqual(nothing_value.map(compose(f, g)), nothing_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        some_value = Some(42)
        nothing_value = Nil()

        self.assertEqual(some_value.apply(Some(lambda x: x)), some_value)
        self.assertEqual(nothing_value.apply(Some(lambda x: x)), nothing_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42

        def f(x: int) -> int:
            return x * 42

        self.assertEqual(Some(x).apply(Some(f)), Some(f(x)))
        self.assertEqual(Nil().apply(Some(f)), Nil())

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

        w: Option[int] = Some(42)
        u: Option[Callable[[int], int]] = Some(lambda x: x + 42)
        v: Option[Callable[[int], int]] = Some(lambda x: x * 42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Some(composition)))), w.apply(v).apply(u)
        )

        u = Nil()
        v = Nil()
        self.assertEqual(
            w.apply(v.apply(u.apply(Some(composition)))), w.apply(v).apply(u)
        )

    def test_map2_and_map3(self):
        self.assertEqual(Some(3), map2(Some(1), Some(2), lambda a, b: a + b))
        self.assertEqual(Nil(), map2(Some(1), Nil(), lambda a, b: a))
        self.assertEqual(
            Some(6), map3(Some(1), Some(2), Some(3), lambda a, b, c: a + b + c)
        )
        self.assertEqual(Nil(), map3(Some(1), Some(2), Nil(), lambda a, b, c: a))

    def test_maybe_monad_representation(self):
        some = Some("a")
        nothing = Nil()

        self.assertEqual(str(some), "Some(a)")
        self.assertEqual(str(nothing), "Nil()")

    def test_maybe_optional_instances(self):
        self.assertTrue(isinstance(Some("a"), (Some, Nil)))  # pyright: ignore[reportUnnecessaryIsInstance]
        self.assertTrue(isinstance(Nil(), (Some, Nil)))  # pyright: ignore[reportUnnecessaryIsInstance]

    def test_nil_is_singleton(self):
        self.assertIs(Nil(), Nil())
        empty: Option[int] = Nil()
        self.assertIs(empty, Nil())

    def test_maybe_from_and_to_optional(self):
        maybe_dict: Option[dict[Any, Any]] = from_optional({})
        maybe_string: Option[str] = from_optional("")
        maybe_none: Option[str] = from_optional(None)

        self.assertEqual({}, maybe_dict.to_optional())
        self.assertEqual("", maybe_string.to_optional())
        self.assertEqual(None, maybe_none.to_optional())

    def test_maybe_is_nothing_is_some(self):
        some = Some(10)
        nothing = Nil()
        maybe_none = from_optional(None)
        maybe_value = from_optional(10)

        self.assertTrue(some.is_some() and not some.is_nothing())
        self.assertTrue(nothing.is_nothing() and not nothing.is_some())
        self.assertTrue(maybe_value.is_some() and not maybe_value.is_nothing())
        self.assertTrue(maybe_none.is_nothing() and not maybe_none.is_some())

    def test_maybe_monad_unwrap(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(10, some_value.unwrap_or(20))
        self.assertEqual(20, nothing.unwrap_or(20))

    def test_maybe_monad_filter(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(Some(10), some_value.filter(lambda v: v > 5))
        self.assertEqual(Nil(), some_value.filter(lambda v: v > 10))
        self.assertEqual(Nil(), nothing.filter(lambda v: v < 10))

    def test_option_monad_is_some_and(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertTrue(some_value.is_some_and(lambda v: v > 5))
        self.assertFalse(some_value.is_some_and(lambda v: v > 10))
        self.assertFalse(nothing.is_some_and(lambda v: v > 5))

    def test_option_monad_map_or(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(11, some_value.map_or(0, lambda v: v + 1))
        self.assertEqual(0, nothing.map_or(0, lambda v: v + 1))

    def test_option_monad_and_or(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(Some(20), some_value.and_(Some(20)))
        self.assertEqual(Nil(), nothing.and_(Some(20)))
        self.assertEqual(some_value, some_value.or_(Some(20)))
        self.assertEqual(Some(20), nothing.or_(Some(20)))

    def test_option_monad_zip(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(Some((10, "a")), some_value.zip(Some("a")))
        self.assertEqual(Nil(), some_value.zip(Nil()))
        self.assertEqual(Nil(), nothing.zip(Some("a")))

    def test_option_monad_flatten(self):
        self.assertEqual(Some(10), Some(Some(10)).flatten())
        self.assertEqual(Nil(), Some(Nil()).flatten())
        self.assertEqual(Nil(), Nil().flatten())

    def test_option_monad_ok_or(self):
        some_value = Some(10)
        nothing = Nil()

        self.assertEqual(Ok(10), some_value.ok_or("error"))
        self.assertEqual(Err("error"), nothing.ok_or("error"))
        self.assertEqual(Ok(10), some_value.ok_or_else(lambda: "error"))
        self.assertEqual(Err("error"), nothing.ok_or_else(lambda: "error"))

    def test_safe_function(self):
        exception_function = MagicMock(side_effect=Exception("error"))
        maybe_unsafe = safe(lambda: exception_function())
        maybe_safe = safe(lambda: 10)

        self.assertEqual(Nil(), maybe_unsafe)
        self.assertEqual(Some(10), maybe_safe)
        self.assertEqual(Nil(), safe(lambda: int("x"), exceptions=(ValueError,)))
        with self.assertRaises(KeyError):
            safe(lambda: {}["missing"], exceptions=(ValueError,))

    def test_maybe_unwrap(self):
        some = Some("a")
        nothing = Nil()

        with self.assertRaises(UnwrapError):
            nothing.unwrap()

        self.assertEqual("a", some.unwrap())
        self.assertEqual("a", some.unwrap_or("b"))
        self.assertEqual(10, nothing.unwrap_or(10))
        self.assertEqual("a", some.unwrap_or_else(lambda: "b"))
        self.assertEqual(10, nothing.unwrap_or_else(lambda: 10))

    def test_maybe_inspect(self):
        some = Some("a")
        nothing = Nil()
        print_mock = Mock()

        self.assertEqual(some, some.inspect(print_mock))
        self.assertEqual(nothing, nothing.inspect(print_mock))
        print_mock.assert_called_once_with("a")

    def test_maybe_functions(self):
        some = Some("a")
        nothing = Nil()

        self.assertEqual("a", some.match(some=lambda x: x, nil=lambda: "default"))
        self.assertEqual(
            "default",
            some.bind(lambda x: Nil()).match(some=lambda x: x, nil=lambda: "default"),
        )
        self.assertEqual(
            "default", nothing.match(some=lambda x: x, nil=lambda: "default")
        )
        self.assertEqual(
            "default",
            nothing.bind(lambda x: Some(x)).match(
                some=lambda x: x, nil=lambda: "default"
            ),
        )

    def test_option_supports_structural_pattern_matching(self):
        def describe(value: Option[int]) -> str:
            # No `case _:` fallback: Option is a closed union (Some[T] | Nil),
            # so this is statically exhaustive without one.
            match value:
                case Some(x):
                    return f"some {x}"
                case Nil():
                    return "nil"

        self.assertEqual("some 10", describe(Some(10)))
        self.assertEqual("nil", describe(Nil()))


class TestOptionValueSemantics(unittest.TestCase):
    def test_equality_and_hash(self):
        self.assertEqual(Some({"a": 1, "b": 2}), Some({"b": 2, "a": 1}))
        self.assertNotEqual(Some(1), Some("1"))
        self.assertNotEqual(Some(1), Nil())
        self.assertEqual(1, len({Some(1), Some(1)}))
        self.assertEqual(1, len({Nil(), Nil()}))
        with self.assertRaises(TypeError):
            hash(Some([]))

    def test_repr(self):
        self.assertEqual("Some('1')", repr(Some("1")))
        self.assertEqual("Some(1)", str(Some(1)))
        self.assertEqual("Nil()", repr(Nil()))

    def test_runtime_checks(self):
        value: Option[int] = Some(1)
        self.assertTrue(is_some(value))
        self.assertFalse(is_nil(value))
        self.assertTrue(is_nil(Nil()))
        self.assertTrue(is_option(Nil()))
        self.assertFalse(is_option(None))
        self.assertIsInstance(Some(1), OPTION_TYPES)


class TestOptionFeatures(unittest.TestCase):
    def test_map_or_else(self):
        empty: Option[int] = Nil()
        self.assertEqual(0, empty.map_or_else(lambda: 0, lambda x: x * 2))
        self.assertEqual(4, Some(2).map_or_else(lambda: 0, lambda x: x * 2))

    def test_xor(self):
        one: Option[int] = Some(1)
        empty: Option[int] = Nil()
        self.assertEqual(Some(1), one.xor(empty))
        self.assertEqual(Some(1), empty.xor(one))
        self.assertEqual(Nil(), one.xor(Some(2)))
        self.assertEqual(Nil(), empty.xor(Nil()))

    def test_aliases(self):
        empty: Option[int] = Nil()
        self.assertEqual(Some(2), Some(1).and_then(lambda x: Some(x + 1)))
        self.assertEqual(Some(1), empty.or_else(lambda: Some(1)))
        self.assertEqual(Some(3), Some(3).or_else(lambda: Some(1)))

    def test_transpose(self):
        self.assertEqual(Ok(Some(1)), Some(Ok(1)).transpose())
        self.assertEqual(Ok(Nil()), Nil().transpose())
        error: Option[Result[int, str]] = Some(Err("e"))
        self.assertEqual(Err("e"), error.transpose())

    def test_curried_functions_with_flow(self):
        # Lambdas passed to curried functions are not inferred (the input type is not known
        # yet), so pipelines use annotated functions.
        def increment(x: int) -> int:
            return x + 1

        def positive(x: int) -> bool:
            return x > 0

        def wrap(x: int) -> Option[int]:
            return Some(x)

        seen: list[int] = []
        start: Option[int] = Some(2)
        self.assertEqual(
            3,
            flow(
                start,
                option.map(increment),
                option.filter(positive),
                option.bind(wrap),
                option.inspect(seen.append),
                option.unwrap_or(0),
            ),
        )
        self.assertEqual([3], seen)
        self.assertEqual(0, option.unwrap_or_else(lambda: 0)(Nil()))
