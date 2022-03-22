import unittest

from pymoliath.either import Right, Left
from pymoliath.result import Ok, Err, Result
from pymoliath.util import compose


class TestResultMonad(unittest.TestCase):
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
        ok_function = lambda x: Ok(x + 1)
        err_function = lambda x: Err(TypeError('error'))

        self.assertEqual(ok_function(10), Ok(10).bind(ok_function))
        self.assertEqual(err_function(10), Err(TypeError('error')).bind(err_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        ok_value = Ok(10)
        failure = Err(TypeError('error'))

        self.assertEqual(ok_value, ok_value.bind(lambda x: Ok(x)))
        self.assertEqual(failure, failure.bind(lambda x: Err(x)))

    def test_either_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        ok_value = Ok(42)
        f = lambda x: Ok(x + 1000)
        g = lambda y: Ok(y * 42)

        self.assertEqual(ok_value.bind(f).bind(g), ok_value.bind(lambda x: f(x).bind(g)))

        err_value = Err(ValueError('error'))
        f = lambda x: Err(ValueError(f'error {x}'))
        g = lambda y: Err(ValueError(f'error {y}'))

        self.assertEqual(err_value.bind(f).bind(g), err_value.bind(lambda x: f(x).bind(g)))

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Ok(10).map(lambda x: x), Ok(10))
        self.assertEqual(Err(TypeError('error')).map(lambda x: x), Err(TypeError('error')))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        ok_value = Ok(42)
        err_value = Err(ValueError('error'))

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(ok_value.map(compose(f, g)), ok_value.map(g).map(f))
        self.assertEqual(err_value.map(compose(f, g)), err_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        ok_value = Ok(42)
        err_value = Err(TypeError('error'))

        self.assertEqual(ok_value.apply(Ok(lambda x: x)), ok_value)
        self.assertEqual(err_value.apply(Ok(lambda x: x)), err_value)
        self.assertEqual(Ok(lambda x: x).apply2(ok_value), ok_value)
        self.assertEqual(Ok(lambda x: x).apply2(err_value), err_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(Ok(x).apply(Ok(f)), Ok(f(x)))
        self.assertEqual(Err(TypeError(x)).apply(Ok(f)), Err(TypeError(x)))

        self.assertEqual(Ok(f).apply2(Ok(x)), Ok(f(x)))
        self.assertEqual(Ok(f).apply2(Err(TypeError(x))), Err(TypeError(x)))

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Ok(42)
        u = Ok(lambda x: x + 42)
        v = Ok(lambda x: x * 42)
        composition = lambda f, g: compose(f, g)

        self.assertEqual(w.apply(v.apply(u.apply(Ok(composition)))), w.apply(v).apply(u))
        self.assertEqual(Ok(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

        w = Err(TypeError(42))
        self.assertEqual(w.apply(v.apply(u.apply(Err(TypeError(42))))), w.apply(v).apply(u))
        self.assertEqual(Ok(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_either_monad_representation(self):
        ok_value = Ok('a')
        err_value = Err(TypeError('b'))

        self.assertEqual(str(ok_value), 'Ok(a)')
        self.assertEqual(str(err_value), f'Err({TypeError("b")})')

    def test_result_abstract_class_raise_error(self):
        with self.assertRaises(TypeError):
            Result()

    def test_result_is_ok_is_err(self):
        ok_value = Ok(10)
        err_value = Err(TypeError("error"))

        self.assertTrue(ok_value.is_ok() and not ok_value.is_err())
        self.assertTrue(err_value.is_err() and not err_value.is_ok())

    def test_safe_function_for_error_handling_with_result_monad_returns_correct_result(self):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_result: Result[int] = Result.safe(unsafe_function, 'Failed during operation: ')
        safe_result: Result[int] = Result.safe(safe_function)

        self.assertEqual(Err(Exception(f'Failed during operation: error')), unsafe_result)
        self.assertEqual(Ok(10), safe_result)

    def test_either_monad_or_else(self):
        ok_value = Ok(10)
        failure = Err(TypeError("error"))

        self.assertEqual(10, ok_value.ok_or_else(10))
        self.assertEqual(str(TypeError("error")), str(failure.err_or_else(TypeError('other error'))))
        self.assertEqual(10, failure.ok_or_else(10))
        self.assertEqual(str(TypeError('other error')), str(ok_value.err_or_else(TypeError('other error'))))

    def test_result_monad_to_either_monad(self):
        ok_value = Ok(10)
        err_value = Err(Exception("error"))

        self.assertEqual(Right(10), ok_value.to_either())
        self.assertEqual(Left(Exception("error")), err_value.to_either())

    def test_result_monad_except(self):
        err_value = Err(TypeError("world"))

        self.assertEqual(Err(TypeError("hello world")), err_value.expect('hello '))

    def test_result_monad_examples(self):
        def divide(dividen: int, divisor: int):
            return dividen / divisor

        def try_divide(dividen: int, divisor: int) -> Result[int]:
            return (Ok(divide)
                    .apply2(Ok(dividen))
                    .apply2(Ok(divisor)))

        def try_divide_map(dividen: int, divisor: int) -> Result[int]:
            return Ok(divide).map(lambda f: f(dividen, divisor))

        result: Result[int] = try_divide(10, 0)
        result2: Result[int] = Ok(0).apply(Ok(10).apply(Ok(divide)))
        result3: Result[int] = try_divide_map(10, 0)

        (result
         .match(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                lambda _: _))

        (result2
         .match(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                lambda r: r + 1))

        (result3
         .match(lambda e: self.assertEqual(str(ZeroDivisionError('division by zero')), str(e)),
                lambda r: r + 1))
