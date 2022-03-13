import unittest
from typing import Any

from pymoliath.reader import Reader
from pymoliath.util import compose


class TestReader(unittest.TestCase):
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
        reader_function = lambda x: Reader(lambda env: x + env)

        self.assertEqual(reader_function(10).run(1), Reader(lambda env: 10).bind(reader_function).run(1))
        self.assertEqual(11, reader_function(10).run(1))
        self.assertEqual(11, Reader(lambda env: 10).bind(reader_function).run(1))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        reader_value = Reader(lambda env: f'Hello {env}')

        self.assertEqual(reader_value.run('world'), reader_value.bind(lambda x: Reader(lambda env: x)).run('world'))
        self.assertEqual('Hello world', reader_value.run('world'))
        self.assertEqual('Hello world', reader_value.bind(lambda x: Reader(lambda env: x)).run('world'))

    def test_just_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        reader_value = Reader(lambda env: 42)
        f = lambda x: Reader(lambda env: x + 1000)
        g = lambda y: Reader(lambda env: y * 42)

        self.assertEqual(reader_value.bind(f).bind(g).run(1), reader_value.bind(lambda x: f(x).bind(g)).run(1))
        self.assertEqual(43764, reader_value.bind(f).bind(g).run(1))
        self.assertEqual(43764, reader_value.bind(lambda x: f(x).bind(g)).run(1))

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Reader(lambda env: env + 10).map(lambda x: x).run(1), Reader(lambda env: env + 10).run(1))
        self.assertEqual(11, Reader(lambda env: env + 10).map(lambda x: x).run(1))
        self.assertEqual(11, Reader(lambda env: env + 10).run(1))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        reader_value = Reader(lambda env: 42)

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(reader_value.map(compose(f, g)).run(1), reader_value.map(g).map(f).run(1))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        reader_value = Reader(lambda env: 42)

        self.assertEqual(reader_value.apply(Reader(lambda env: lambda x: x)).run(1), reader_value.run(1))
        self.assertEqual(Reader(lambda env: lambda x: x).apply2(reader_value).run(1), reader_value.run(1))

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(Reader(lambda env: x).apply(Reader(lambda env: f)).run(1), Reader(lambda env: f(x)).run(1))
        self.assertEqual(Reader(lambda env: f).apply2(Reader(lambda env: x)).run(1), Reader(lambda env: f(x)).run(1))

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Reader(lambda env: 42)
        u = Reader(lambda env: lambda x: x + 42)
        v = Reader(lambda env: lambda x: x * 42)
        composed_reader = Reader(lambda env: lambda f, g: compose(f, g))

        self.assertEqual(w.apply(v.apply(u.apply(composed_reader))).run(1), w.apply(v).apply(u).run(1))
        self.assertEqual(composed_reader.apply2(u).apply2(v).apply2(w).run(1), u.apply2(v.apply2(w)).run(1))

    def test_reader_monad_representation(self):
        def test(env: Any):
            return 'a'

        reader = Reader(test)

        self.assertEqual(str(reader), f'Reader({str(test)})')

        calculate_length = Reader.ask().map(lambda v: len(v))
        calculate_modified_length = calculate_length.local(lambda env: f'Prefix {env}')

        self.assertEqual(5, calculate_length.run('12345'))
        self.assertEqual(12, calculate_modified_length.run('12345'))
