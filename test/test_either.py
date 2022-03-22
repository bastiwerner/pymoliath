import unittest

from pymoliath.either import Left, Right, Either
from pymoliath.util import compose


class TestEitherResultMonad(unittest.TestCase):
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
        right_function = lambda x: Right(x + 1)
        left_function = lambda x: Left(f'{x}')

        self.assertEqual(right_function(10), Right(10).bind(right_function))
        self.assertEqual(left_function(10), Left(10).bind(left_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        right_value = Right(10)
        left_value = Left('err')

        self.assertEqual(right_value, right_value.bind(lambda x: Right(x)))
        self.assertEqual(left_value, left_value.bind(lambda x: Left(x)))

    def test_either_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        right_value = Right(42)
        f = lambda x: Right(x + 1000)
        g = lambda y: Right(y * 42)

        self.assertEqual(right_value.bind(f).bind(g), right_value.bind(lambda x: f(x).bind(g)))

        left_value = Left(42)
        f = lambda x: Left(x + 1000)
        g = lambda y: Left(y * 42)

        self.assertEqual(left_value.bind(f).bind(g), left_value.bind(lambda x: f(x).bind(g)))

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Right(10).map(lambda x: x), Right(10))
        self.assertEqual(Left(10).map(lambda x: x), Left(10))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        right_value = Right(42)
        left_value = Left(42)

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(right_value.map(compose(f, g)), right_value.map(g).map(f))
        self.assertEqual(left_value.map(compose(f, g)), left_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        right_value = Right(42)
        left_value = Left(42)

        self.assertEqual(right_value.apply(Right(lambda x: x)), right_value)
        self.assertEqual(left_value.apply(Right(lambda x: x)), left_value)
        self.assertEqual(Right(lambda x: x).apply2(right_value), right_value)
        self.assertEqual(Right(lambda x: x).apply2(left_value), left_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(Right(x).apply(Right(f)), Right(f(x)))
        self.assertEqual(Left(x).apply(Right(f)), Left(x))

        self.assertEqual(Right(f).apply2(Right(x)), Right(f(x)))
        self.assertEqual(Right(f).apply2(Left(x)), Left(x))

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Right(42)
        u = Right(lambda x: x + 42)
        v = Right(lambda x: x * 42)
        composition = lambda f, g: compose(f, g)

        self.assertEqual(w.apply(v.apply(u.apply(Right(composition)))), w.apply(v).apply(u))
        self.assertEqual(Right(composition).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

        w = Left(42)

        self.assertEqual(w.apply(v.apply(u.apply(Right(composition)))), w.apply(v).apply(u))
        self.assertEqual(Right(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_either_monad_representation(self):
        right_value = Right('a')
        left_value = Left('b')

        self.assertEqual(str(right_value), 'Right(a)')
        self.assertEqual(str(left_value), 'Left(b)')

    def test_either_abstract_class_raise_error(self):
        with self.assertRaises(TypeError):
            Either()

    def test_either_is_left_is_right(self):
        right = Right(10)
        left = Left("error")

        self.assertTrue(right.is_right() and not right.is_left())
        self.assertTrue(left.is_left() and not left.is_right())

    def test_safe_function_for_error_handling_with_either_monad_returns_correct_either_monad(self):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_either_result: Either[Exception, int] = Either.safe(unsafe_function, 'Failed during operation: ')
        safe_either_result: Either[Exception, int] = Either.safe(safe_function)

        self.assertEqual(Left(Exception(f'Failed during operation: error')), unsafe_either_result)
        self.assertEqual(Right(10), safe_either_result)

    def test_either_monad_or_else(self):
        right_value = Right('right')
        left_value = Left('left')

        self.assertEqual('right', right_value.right_or_else('default'))
        self.assertEqual('left', left_value.left_or_else('default'))
        self.assertEqual('default', right_value.left_or_else('default'))
        self.assertEqual('default', left_value.right_or_else('default'))

    def test_either_monad_result_function(self):
        right_value = Right('right')
        left_value = Left('left')

        (right_value
         .either(lambda _: 'error', lambda x: self.assertEqual(x, 'right')))

        (right_value
         .bind(lambda x: left_value)
         .either(lambda x: self.assertEqual(x, 'left'), lambda x: x))

        (left_value
         .either(lambda x: self.assertEqual(x, 'left'), lambda x: x))

        (left_value
         .bind(lambda x: Left('other'))
         .either(lambda x: self.assertEqual('left', x), lambda x: x))

        (left_value
         .bind(lambda x: Left('other'))
         .either(lambda x: self.assertEqual('left', x), lambda x: x))
