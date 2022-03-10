import unittest
from unittest.mock import patch, Mock, call

from pymoliath.io import IO
from pymoliath.util import compose


class TestIOMonads(unittest.TestCase):
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
        io_function = lambda x: IO(lambda: x + 1)

        self.assertEqual(io_function(10).run(), IO(lambda: 10).bind(io_function).run())

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        io_value = IO(lambda: 'Hello')

        self.assertEqual(io_value.run(), io_value.bind(lambda x: IO(lambda: x)).run())

    def test_io_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        io_value = IO(lambda: 42)
        f = lambda x: IO(lambda: x + 1000)
        g = lambda y: IO(lambda: y * 42)

        self.assertEqual(io_value.bind(f).bind(g).run(), io_value.bind(lambda x: f(x).bind(g)).run())

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(IO(lambda: 10).map(lambda x: x).run(), IO(lambda: 10).run())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        io_value = IO(lambda: 42)

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(io_value.map(compose(f, g)).run(), io_value.map(g).map(f).run())

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        io_value = IO(lambda: 42)

        self.assertEqual(io_value.apply(IO(lambda: lambda x: x)).run(), io_value.run())
        self.assertEqual(IO(lambda: lambda x: x).apply2(io_value).run(), io_value.run())

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(IO(lambda: x).apply(IO(lambda: f)).run(), IO(lambda: f(x)).run())
        self.assertEqual(IO(lambda: f).apply2(IO(lambda: x)).run(), IO(lambda: f(x)).run())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = IO(lambda: 42)
        u = IO(lambda: lambda x: x + 42)
        v = IO(lambda: lambda x: x * 42)
        composition = lambda f, g: compose(f, g)

        self.assertEqual(w.apply(v.apply(u.apply(IO(lambda: composition)))).run(), w.apply(v).apply(u).run())
        self.assertEqual(IO(lambda: composition).apply2(u).apply2(v).apply2(w).run(), u.apply2(v.apply2(w)).run())

    def test_io_monad_non_callable_value_raises_error(self):
        with self.assertRaises(TypeError):
            IO(42)

    def test_io_monad_representation(self):
        def test():
            return 'a'

        io = IO(test)

        self.assertEqual(str(io), f'IO {str(test)}')

    @patch('builtins.print')
    def test_io_result_function(self, mock_print: Mock):
        io = IO(lambda: 'a')

        self.assertEqual('a', io.run())

        io_function = IO(lambda: 'hello')

        self.assertEqual('hello world', io_function.map(lambda x: f'{x} world').run())

        read = lambda id: IO(lambda: id)
        write = lambda id: lambda value: IO(lambda: print(f'{id}: {value}'))
        to_upper = lambda text: str(text).upper()

        change_to_upper_io = (read("value")
                              .map(to_upper)
                              .bind(write('WRITE'))
                              .bind(lambda _: read('yes'))
                              .map(to_upper)
                              .bind(write("WRITE"))
                              )

        change_to_upper_io.run()

        mock_print.assert_has_calls([call('WRITE: VALUE'), call('WRITE: YES')])
