import unittest
from typing import Tuple

from pymoliath.lazy import LazyMonad, Sequence
from pymoliath.util import compose


class TestLazyMonad(unittest.TestCase):
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
        lazy_function = lambda x: LazyMonad(x + 1)
        lazy_function2 = lambda x: LazyMonad(lambda: x + 1)

        self.assertEqual(lazy_function(10).run(), LazyMonad(10).bind(lazy_function).run())
        self.assertEqual(lazy_function2(10).run(), LazyMonad(10).bind(lazy_function2).run())

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        lazy_value = LazyMonad('Hello')

        self.assertEqual(lazy_value.run(), lazy_value.bind(lambda x: LazyMonad(lambda: x)).run())

    def test_io_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        lazy_value = LazyMonad(42)
        f = lambda x: LazyMonad(x + 1000)
        g = lambda y: LazyMonad(y * 42)

        self.assertEqual(lazy_value.bind(f).bind(g).run(), lazy_value.bind(lambda x: f(x).bind(g)).run())

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(LazyMonad(10).map(lambda x: x).run(), LazyMonad(10).run())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        lazy_value = LazyMonad(42)

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(lazy_value.map(compose(f, g)).run(), lazy_value.map(g).map(f).run())

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        lazy_value = LazyMonad(42)  # is equal to LazyMonad(lambda: 42)
        applicative = LazyMonad(lambda: lambda x: x)  # LazyMonad( void -> f (x) -> x )

        self.assertEqual(lazy_value.apply(applicative).run(), lazy_value.run())
        self.assertEqual(applicative.apply2(lazy_value).run(), lazy_value.run())

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42
        lazy_value = LazyMonad(x)  # LazyMonad( void -> x ) contains a value
        applicative = LazyMonad(lambda: f)  # LazyMonad(void -> f(x) -> x ) contains a function

        self.assertEqual(lazy_value.apply(applicative).run(), LazyMonad(f(x)).run())
        self.assertEqual(applicative.apply2(lazy_value).run(), LazyMonad(f(x)).run())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = LazyMonad(42)  # LazyMonad( void -> 42 ) contains a value
        u = LazyMonad(lambda: lambda x: x + 42)  # LazyMonad( void -> f(x) -> x ) contains a function
        v = LazyMonad(lambda: lambda x: x * 42)  # LazyMonad( void -> f(x) -> x ) contains a function
        composition = lambda f, g: compose(f, g)  # h(x) = f(g(x))
        applicative = LazyMonad(lambda: composition)  # LazyMonad( void -> h(x) -> x ) contains a function composition

        self.assertEqual(w.apply(v.apply(u.apply(applicative))).run(), w.apply(v).apply(u).run())
        self.assertEqual(applicative.apply2(u).apply2(v).apply2(w).run(), u.apply2(v.apply2(w)).run())

    def test_io_monad_representation(self):
        def test():
            return 'a'

        lazy = LazyMonad(test)

        self.assertEqual(str(lazy), f'LazyMonad({str(test)})')

    def test_io_result_function(self):
        lazy = LazyMonad(42)

        self.assertEqual(42, lazy.run())

        result: LazyMonad[str] = (lazy
                                  .map(lambda x: x ** 2)
                                  .map(lambda x: x + 5)
                                  .bind(lambda x: LazyMonad(str(x))))
        self.assertEqual("1769", result.run())


class TestSequence(unittest.TestCase):
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
        it’s the same as list taking the value and applying the function to it.
        """
        list_function = lambda x: Sequence([x + 1, x + 2])

        self.assertEqual(list_function(10).run(), Sequence([10]).bind(list_function).run())

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        list_value = Sequence(['Hello', 'world'])

        self.assertEqual(list_value.run(), list_value.bind(lambda x: Sequence([x])).run())

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Sequence([10]).map(lambda x: x).run(), Sequence([10]).run())

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        list_value = Sequence([42])

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(list_value.map(compose(f, g)).run(), list_value.map(g).map(f).run())

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        list_value = Sequence([42, 10])

        self.assertEqual(list_value.apply(Sequence([lambda x: x])).run(), list_value.run())
        self.assertEqual(Sequence([lambda x: x]).apply2(list_value).run(), list_value.run())

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(Sequence([x]).apply(Sequence([f])).run(), Sequence([f(x)]).run())
        self.assertEqual(Sequence([f]).apply2(Sequence([x])).run(), Sequence([f(x)]).run())

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Sequence([42])
        u = Sequence([lambda x: x + 42])
        v = Sequence([lambda x: x * 42])
        composition = lambda f, g: compose(f, g)

        self.assertEqual(w.apply(v.apply(u.apply(Sequence([composition])))).run(), w.apply(v).apply(u).run())
        self.assertEqual(Sequence([composition]).apply2(u).apply2(v).apply2(w).run(),
                         u.apply2(v.apply2(w)).run())

    def test_list_monad_with_nested_list_and_enumeration_returns_correct_result(self):
        result: Sequence[Tuple[int, str]] = (Sequence([['hello', 'world']])
                                             .bind(lambda d: Sequence(enumerate(d)))
                                             .map(lambda tuple_result: (tuple_result[0], tuple_result[1])))

        self.assertEqual([(0, 'hello'), (1, 'world')], result.run())

    def test_list_monad_to_list_returns_correct_python_list(self):
        result = Sequence([1, 2, 3, 4])

        self.assertEqual([1, 2, 3, 4], result.run())
        self.assertEqual('[1, 2, 3, 4]', str(result.run()))

    def test_lazy_sequence_monad_with_filter_and_take_returns_corrected_filtered_list(self):
        result = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9], result.take(20).run())
        self.assertEqual([], result.skip(10).run())
        self.assertEqual([1, 2, 3, 4], result.take(4).run())
        self.assertEqual([6, 7, 8, 9], result.skip(5).run())
        self.assertEqual([3, 4], result.take(4).filter(lambda x: x > 2).run())

    def test_monad_list_examples(self):
        # test list monad from generator
        def list_generator():
            for x in range(5):
                yield x

        list_monad = Sequence(list_generator()).map(lambda x: x + 1)
        self.assertEqual([1, 2, 3, 4, 5], list_monad.run())

        self.assertEqual(['$2.00', '$100.00', '$5.00'], (Sequence([1, 99, 4])
                                                         .bind(lambda val: Sequence([val + 1]))
                                                         .bind(lambda val: Sequence([f"${val}.00"]))).run())

        self.assertEqual(['$2.00', '$100.00', '$5.00'], (Sequence([1, 99, 4])
                                                         .map(lambda val: val + 1)
                                                         .bind(lambda val: Sequence([f"${val}.00"]))).run())
