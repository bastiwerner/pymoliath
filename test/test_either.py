import unittest

from pymoliath.either import Left, Right, Success, Failure, Try, Either
from pymoliath.util import compose


class TestEitherMonad(unittest.TestCase):
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
        success_function = lambda x: Success(x + 1)
        failure_function = lambda x: Failure(TypeError('error'))

        f = lambda a, b: a + b

        self.assertEqual(right_function(10), Right(10).bind(right_function))
        self.assertEqual(left_function(10), Left(10).bind(left_function))
        self.assertEqual(success_function(10), Success(10).bind(success_function))
        self.assertEqual(failure_function(10), Failure(TypeError('error')).bind(failure_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        right_value = Right(10)
        left_value = Left('err')
        success_value = Success(10)
        failure = Failure(TypeError('error'))

        self.assertEqual(right_value, right_value.bind(lambda x: Right(x)))
        self.assertEqual(left_value, left_value.bind(lambda x: Left(x)))
        self.assertEqual(success_value, success_value.bind(lambda x: Success(x)))
        self.assertEqual(failure, failure.bind(lambda x: Failure(x)))

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

        success_value = Success(42)
        f = lambda x: Success(x + 1000)
        g = lambda y: Success(y * 42)

        self.assertEqual(success_value.bind(f).bind(g), success_value.bind(lambda x: f(x).bind(g)))

        failure_value = Failure(ValueError('error'))
        f = lambda x: Failure(ValueError(f'error {x}'))
        g = lambda y: Failure(ValueError(f'error {y}'))

        self.assertEqual(failure_value.bind(f).bind(g), failure_value.bind(lambda x: f(x).bind(g)))

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Right(10).map(lambda x: x), Right(10))
        self.assertEqual(Left(10).map(lambda x: x), Left(10))
        self.assertEqual(Success(10).map(lambda x: x), Success(10))
        self.assertEqual(Failure(TypeError('error')).map(lambda x: x), Failure(TypeError('error')))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        right_value = Right(42)
        left_value = Left(42)
        success_value = Success(42)
        failure_value = Failure(ValueError('error'))

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(right_value.map(compose(f, g)), right_value.map(g).map(f))
        self.assertEqual(left_value.map(compose(f, g)), left_value.map(g).map(f))
        self.assertEqual(success_value.map(compose(f, g)), success_value.map(g).map(f))
        self.assertEqual(failure_value.map(compose(f, g)), failure_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        right_value = Right(42)
        left_value = Left(42)
        success_value = Success(42)
        failure_value = Failure(TypeError('error'))

        self.assertEqual(right_value.apply(Right(lambda x: x)), right_value)
        self.assertEqual(left_value.apply(Right(lambda x: x)), left_value)
        self.assertEqual(Right(lambda x: x).apply2(right_value), right_value)
        self.assertEqual(Right(lambda x: x).apply2(left_value), left_value)

        self.assertEqual(success_value.apply(Success(lambda x: x)), success_value)
        self.assertEqual(failure_value.apply(Success(lambda x: x)), failure_value)
        self.assertEqual(Success(lambda x: x).apply2(success_value), success_value)
        self.assertEqual(Success(lambda x: x).apply2(failure_value), failure_value)

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
        self.assertEqual(Success(x).apply(Success(f)), Success(f(x)))
        self.assertEqual(Failure(TypeError(x)).apply(Success(f)), Failure(TypeError(x)))

        self.assertEqual(Right(f).apply2(Right(x)), Right(f(x)))
        self.assertEqual(Right(f).apply2(Left(x)), Left(x))
        self.assertEqual(Success(f).apply2(Success(x)), Success(f(x)))
        self.assertEqual(Success(f).apply2(Failure(TypeError(x))), Failure(TypeError(x)))

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

        w = Success(42)
        u = Success(lambda x: x + 42)
        v = Success(lambda x: x * 42)

        self.assertEqual(w.apply(v.apply(u.apply(Success(composition)))), w.apply(v).apply(u))
        self.assertEqual(Success(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

        w = Failure(TypeError(42))
        self.assertEqual(w.apply(v.apply(u.apply(Failure(TypeError(42))))), w.apply(v).apply(u))
        self.assertEqual(Success(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_either_monad_representation(self):
        right_value = Right('a')
        left_value = Left('b')
        success_value = Success('a')
        failure_value = Failure(TypeError('b'))

        self.assertEqual(str(right_value), 'Right a')
        self.assertEqual(str(left_value), 'Left b')
        self.assertEqual(str(success_value), 'Success a')
        self.assertEqual(str(failure_value), f'Failure {TypeError("b")}')

    def test_either_try_abstract_class_raise_error(self):
        with self.assertRaises(TypeError):
            Either()

        with self.assertRaises(TypeError):
            Try()

    def test_either_try_is_left_is_right_is_success_if_failure(self):
        right = Right(10)
        left = Left("error")
        success = Success(10)
        failure = Failure(TypeError("error"))

        self.assertTrue(right.is_right() and not right.is_left())
        self.assertTrue(left.is_left() and not left.is_right())
        self.assertTrue(success.is_success() and not success.is_failure())
        self.assertTrue(failure.is_failure() and not failure.is_success())

    def test_either_monad_or_else(self):
        right_value = Right('right')
        left_value = Left('left')
        success_value = Success(10)
        failure = Failure(TypeError("error"))

        self.assertEqual('right', right_value.right_or_else('default'))
        self.assertEqual('left', left_value.left_or_else('default'))
        self.assertEqual('default', right_value.left_or_else('default'))
        self.assertEqual('default', left_value.right_or_else('default'))
        self.assertEqual(10, success_value.success_or_else(10))
        self.assertEqual(str(TypeError("error")), str(failure.failure_or_else(TypeError('other error'))))
        self.assertEqual(10, failure.success_or_else(10))
        self.assertEqual(str(TypeError('other error')), str(success_value.failure_or_else(TypeError('other error'))))

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

    def test_try_either_monad(self):
        def divide(dividen: int, divisor: int):
            return dividen / divisor

        def try_divide(dividen: int, divisor: int) -> Try[int]:
            return (Success(divide)
                    .apply2(Success(dividen))
                    .apply2(Success(divisor)))

        def try_divide_map(dividen: int, divisor: int) -> Try[int]:
            return Success(divide).map(lambda f: f(dividen, divisor))

        result: Try[int] = try_divide(10, 0)
        result2: Try[int] = Success(0).apply(Success(10).apply(Success(divide)))
        result3: Try[int] = try_divide_map(10, 0)

        (result
         .either(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                 lambda _: _))

        (result2
         .either(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                 lambda r: r + 1))

        (result3
         .either(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                 lambda r: r + 1))
