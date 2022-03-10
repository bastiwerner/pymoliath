import unittest

from pymoliath.util import compose
from pymoliath.writer import Writer


class TestWriterMonad(unittest.TestCase):
    """
    Monad operations:
    ≡       Identical to
    >>=     bind, flatMap
    >=      map
    (.)     Function composition
    <*>     Applicative functor: (<$>) :: (Functor f) => (a -> b) -> f a -> f b
    <$>     Applicative functor: (<*>) :: f (a -> b) -> f a -> f b
    """

    def test_monad_left_identity_law(self):
        """Left identity law: return a >>= f ≡ f a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        > The `return or pure` function means the instantiation of a Writer monad with an empty (identity) monoid.

        Left identity: The first monad law states that if we take a value,
        put it in a default context with return and then feed it to a function by using >>=,
        it’s the same as just taking the value and applying the function to it.
        """
        writer_function = lambda v: Writer(v, 'monoid type string')

        self.assertEqual(writer_function(10).run(), Writer(10, '').bind(writer_function).run())

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        > The `return or pure` function means the instantiation of a Writer monad with an empty (identity) monoid.

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        writer_value = Writer(5, 'monoid type string')

        self.assertEqual(writer_value.run(), writer_value.bind(lambda v: Writer(v, '')).run())

    def test_io_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        writer_value = Writer(5, 'monoid type string')
        f = lambda v: Writer(v + 1000, 'add operation')
        g = lambda v: Writer(v * 42, 'multiply operation')

        self.assertEqual(writer_value.bind(f).bind(g).run(), writer_value.bind(lambda v: f(v).bind(g)).run())

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Writer(10, 'hi').map(lambda value: value).run(), Writer(10, 'hi').run())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        io_value = Writer(42, 'hi')

        f = lambda value: value + 1000
        g = lambda value: value * 42

        self.assertEqual(io_value.map(compose(f, g)).run(), io_value.map(g).map(f).run())

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://en.wikibooks.org/wiki/Haskell/Applicative_functors

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        writer_value = Writer(10, 'initial')
        applicative = Writer(lambda x: x, '')

        self.assertEqual(writer_value.apply(applicative).run(), writer_value.run())
        self.assertEqual(applicative.apply2(writer_value).run(), writer_value.run())

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://en.wikibooks.org/wiki/Haskell/Applicative_functors

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        f = lambda v: v * 42

        self.assertEqual(Writer(f, '').apply2(Writer(10, 'hi')).run(), Writer(f(10), 'hi').run())
        self.assertEqual(Writer(10, 'hi').apply(Writer(f, '')).run(), Writer(f(10), 'hi').run())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://en.wikibooks.org/wiki/Haskell/Applicative_functors

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Writer(42, 'hi')
        u = Writer(lambda v: v + 42, ' add operation')
        v = Writer(lambda v: v * 42, ' multiply operation')
        pure = Writer(lambda f, g: compose(f, g), '')

        self.assertEqual(w.apply(v.apply(u.apply(pure))).run(), w.apply(v).apply(u).run())
        self.assertEqual(pure.apply2(u).apply2(v).apply2(w).run(), u.apply2(v.apply2(w)).run())

    def test_writer_monad_representation(self):
        writer = Writer(10, 'hi')

        self.assertEqual(str(writer), f"Writer (10, 'hi')")

    def test_writer_examples(self):
        map_example = Writer(1, "map Example").map(lambda v: v + 1)
        self.assertEqual((2, "map Example"), map_example.run())

        apply_example = Writer(10, "apply value")
        apply_example_function = Writer(lambda v: v % 7, "apply function ")
        self.assertEqual((3, "apply function apply value"), apply_example_function.apply2(apply_example).run())

        bind_value = Writer(5, "bind value ")
        bind_function = lambda v: Writer(v * v, "bind function")
        self.assertEqual((25, "bind value bind function"), bind_value.bind(bind_function).run())

        tell_example = Writer(1, "tell example").tell(" monoid append")
        self.assertEqual((1, "tell example monoid append"), tell_example.run())

        listen_example = Writer(10, "listen log").listen()
        self.assertEqual(((10, "listen log"), "listen log"), listen_example.run())

        pass_example = Writer((10, lambda w: w + " passing"), "monoid").pass_()
        self.assertEqual((10, "monoid passing"), pass_example.run())

        self.assertEqual((6.0, 'initial value add 1 multiply by 2 divide by 2'),
                         (Writer(5.0, '').tell("initial value ")
                          .bind(lambda x: Writer(x + 1, "add 1 "))
                          .apply(Writer(lambda i: i * 2, "multiply by 2 "))
                          .map(lambda x: x / 2).tell("divide by 2")
                          .run()))

        self.assertEqual((6.0, 'initial value add 1 multiply by 2 divide by 2'),
                         (Writer(5.0, '')
                          .tell("initial value ")
                          .bind(lambda x: Writer(x + 1, "add 1 "))
                          .map(lambda v: v * 2)
                          .tell('multiply by 2 ')
                          .bind(lambda v: Writer(lambda x: v / x, f"divide by "))
                          .apply2(Writer(2, "2"))
                          .run()))
