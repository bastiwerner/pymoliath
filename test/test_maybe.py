import unittest
from unittest.mock import Mock
from unittest.mock import MagicMock

from pymoliath.maybe import Just, Nothing, Maybe
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
        just_value = Just('Hello')
        nothing_value = Nothing()

        self.assertEqual(just_value, just_value.bind(lambda x: Just(x)))
        self.assertEqual(nothing_value, nothing_value.bind(lambda _: Nothing()))

    def test_just_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        just_value = Just(42)

        def f(x):
            return Just(x + 1000)

        def g(y):
            return Just(y * 42)

        self.assertEqual(just_value.bind(f).bind(g), just_value.bind(lambda x: f(x).bind(g)))

        nothing_value = Nothing()

        def h(_):
            return Nothing()

        def i(_):
            return Nothing()

        self.assertEqual(nothing_value.bind(h).bind(i), nothing_value.bind(lambda x: h(x).bind(i)))

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

        def f(x):
            return x + 1000

        def g(y):
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

        self.assertEqual(Just(lambda x: x).apply2(just_value), just_value)
        self.assertEqual(Just(lambda x: x).apply2(nothing_value), nothing_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42

        def f(x):
            return x * 42

        self.assertEqual(Just(x).apply(Just(f)), Just(f(x)))
        self.assertEqual(Nothing().apply(Just(f)), Nothing())

        self.assertEqual(Just(f).apply2(Just(x)), Just(f(x)))
        self.assertEqual(Just(f).apply2(Nothing()), Nothing())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Just(42)
        u = Just(lambda x: x + 42)
        v = Just(lambda x: x * 42)

        def composition(f, g):
            return compose(f, g)

        self.assertEqual(w.apply(v.apply(u.apply(Just(composition)))), w.apply(v).apply(u))
        self.assertEqual(Just(composition).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

        w = Just(42)
        u = Nothing()
        v = Nothing()

        self.assertEqual(w.apply(v.apply(u.apply(Just(composition)))), w.apply(v).apply(u))
        self.assertEqual(Just(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_maybe_abstract_class_will_raise_exception(self):
        with self.assertRaises(TypeError):
            maybe = Maybe()

    def test_maybe_monad_representation(self):
        just = Just('a')
        nothing = Nothing()

        self.assertEqual(str(just), 'Just(a)')
        self.assertEqual(str(nothing), 'Nothing()')

    def test_maybe_optional_instances(self):
        self.assertTrue(isinstance(Just('a'), Maybe))
        self.assertTrue(isinstance(Nothing(), Maybe))

    def test_maybe_from_and_to_optional(self):
        maybe_dict: Maybe[dict] = Maybe.from_optional({})
        maybe_string: Maybe[str] = Maybe.from_optional('')
        maybe_none: Maybe[str] = Maybe.from_optional(None)

        self.assertEqual({}, maybe_dict.to_optional())
        self.assertEqual('', maybe_string.to_optional())
        self.assertEqual(None, maybe_none.to_optional())

    def test_maybe_is_nothing_is_just(self):
        just = Just(10)
        nothing = Nothing()
        maybe_none = Maybe.from_optional(None)
        maybe_value = Maybe.from_optional(10)

        self.assertTrue(just.is_just() and not just.is_nothing())
        self.assertTrue(nothing.is_nothing() and not nothing.is_just())
        self.assertTrue(maybe_value.is_just() and not maybe_value.is_nothing())
        self.assertTrue(maybe_none.is_nothing() and not maybe_none.is_just())

    def test_maybe_monad_unwrap(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(10, just_value.unwrap_or(20))
        self.assertEqual(20, nothing.unwrap_or(20))

    def test_maybe_monad_filter(self):
        just_value = Just(10)
        nothing = Nothing()

        self.assertEqual(Just(10), just_value.filter(lambda v: v > 10))
        self.assertEqual(Nothing(), just_value.filter(lambda v: v <= 10))
        self.assertEqual(Nothing(), nothing.filter(lambda v: v < 10))

    def maybe_safe_function(self):
        exception_function = MagicMock(side_effect=Exception("error"))
        maybe_unsafe = Maybe.safe(lambda: exception_function())
        maybe_safe = Maybe.safe(lambda: 10)

        self.assertEqual(Nothing(), maybe_unsafe)
        self.assertEqual(Just(10), maybe_safe)

    def test_maybe_unwrap(self):
        just = Just('a')
        nothing = Nothing()

        with self.assertRaises(Exception):
            nothing.unwrap()

        self.assertEqual('a', just.unwrap())
        self.assertEqual('a', just.unwrap_or('b'))
        self.assertEqual(10, nothing.unwrap_or(10))
        self.assertEqual('a', just.unwrap_or_else(lambda: 10))
        self.assertEqual(10, nothing.unwrap_or_else(lambda: 10))

    def test_maybe_inspect(self):
        just = Just('a')
        nothing = Nothing()
        print_mock = Mock()

        self.assertEqual(just, just.inspect(print_mock))
        self.assertEqual(nothing, nothing.inspect(print_mock))
        print_mock.assert_called_once_with('a')

    def test_maybe_functions(self):
        just = Just('a')
        nothing = Nothing()

        self.assertEqual('a', just.match(lambda x: x, lambda: 'default'))
        self.assertEqual('default', just.bind(lambda x: Nothing()).match(lambda x: x, lambda: 'default'))
        self.assertEqual('default', nothing.match(lambda x: x, lambda: 'default'))
        self.assertEqual('default', nothing.bind(lambda x: Just(x)).match(lambda x: x, lambda: 'default'))
