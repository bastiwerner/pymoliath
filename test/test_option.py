import unittest
from typing import Any, Callable
from unittest.mock import MagicMock, Mock

from pymoliath.option import Nil, Option, Some, from_optional, safe
from pymoliath.result import Err, Ok
from pymoliath.util import compose


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

        self.assertEqual(Some(lambda x: x).apply2(some_value), some_value)
        self.assertEqual(Some(lambda x: x).apply2(nothing_value), nothing_value)

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

        self.assertEqual(Some(f).apply2(Some(x)), Some(f(x)))
        self.assertEqual(Some(f).apply2(Nil()), Nil())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Some(42)
        u = Some(lambda x: x + 42)
        v = Some(lambda x: x * 42)

        def composition(
            f: Callable[[Any], Any], g: Callable[[Any], Any]
        ) -> Callable[[Any], Any]:
            return compose(f, g)

        self.assertEqual(
            w.apply(v.apply(u.apply(Some(composition)))), w.apply(v).apply(u)
        )
        self.assertEqual(
            Some(composition).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w))
        )

        w = Some(42)
        u = Nil()
        v = Nil()

        self.assertEqual(
            w.apply(v.apply(u.apply(Some(composition)))), w.apply(v).apply(u)
        )
        self.assertEqual(
            Some(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w),
            u.apply2(v.apply2(w)),
        )

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
        self.assertIs(Nil[int](), Nil[str]())

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

    def maybe_safe_function(self):
        exception_function = MagicMock(side_effect=Exception("error"))
        maybe_unsafe = safe(lambda: exception_function())
        maybe_safe = safe(lambda: 10)

        self.assertEqual(Nil(), maybe_unsafe)
        self.assertEqual(Some(10), maybe_safe)

    def test_maybe_unwrap(self):
        some = Some("a")
        nothing = Nil()

        with self.assertRaises(Exception):
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

        self.assertEqual("a", some.match(lambda x: x, lambda: "default"))
        self.assertEqual(
            "default",
            some.bind(lambda x: Nil()).match(lambda x: x, lambda: "default"),
        )
        self.assertEqual("default", nothing.match(lambda x: x, lambda: "default"))
        self.assertEqual(
            "default",
            nothing.bind(lambda x: Some(x)).match(lambda x: x, lambda: "default"),
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
