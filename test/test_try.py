import unittest
from unittest.mock import Mock

from pymoliath import Right, Left, Ok, Err
from pymoliath.exception import Try, Success, Failure
from pymoliath.util import compose


class TestTryMonad(unittest.TestCase):
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
        success_function = lambda x: Success(x + 1)
        failure_function = lambda x: Failure(TypeError('error'))

        self.assertEqual(success_function(10), Success(10).bind(success_function))
        self.assertEqual(failure_function(10), Failure(TypeError('error')).bind(failure_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        success_value = Success(10)
        failure = Failure(TypeError('error'))

        self.assertEqual(success_value, success_value.bind(lambda x: Success(x)))
        self.assertEqual(failure, failure.bind(lambda x: Failure(x)))

    def test_either_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
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
        self.assertEqual(Success(10).map(lambda x: x), Success(10))
        self.assertEqual(Failure(TypeError('error')).map(lambda x: x), Failure(TypeError('error')))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        success_value = Success(42)
        failure_value = Failure(ValueError('error'))

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(success_value.map(compose(f, g)), success_value.map(g).map(f))
        self.assertEqual(failure_value.map(compose(f, g)), failure_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        success_value = Success(42)
        failure_value = Failure(TypeError('error'))

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

        self.assertEqual(Success(x).apply(Success(f)), Success(f(x)))
        self.assertEqual(Failure(TypeError(x)).apply(Success(f)), Failure(TypeError(x)))

        self.assertEqual(Success(f).apply2(Success(x)), Success(f(x)))
        self.assertEqual(Success(f).apply2(Failure(TypeError(x))), Failure(TypeError(x)))

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        composition = lambda f, g: compose(f, g)
        w = Success(42)
        u = Success(lambda x: x + 42)
        v = Success(lambda x: x * 42)

        self.assertEqual(w.apply(v.apply(u.apply(Success(composition)))), w.apply(v).apply(u))
        self.assertEqual(Success(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

        w = Failure(TypeError(42))
        self.assertEqual(w.apply(v.apply(u.apply(Failure(TypeError(42))))), w.apply(v).apply(u))
        self.assertEqual(Success(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_either_monad_representation(self):
        success_value = Success('a')
        failure_value = Failure(TypeError('b'))

        self.assertEqual(str(success_value), 'Success(a)')
        self.assertEqual(str(failure_value), f'Failure({TypeError("b")})')

    def test_either_abstract_class_raise_error(self):
        with self.assertRaises(TypeError):
            Try()

    def test_either_try_is_left_is_right_is_success_if_failure(self):
        success = Success(10)
        failure = Failure(TypeError("error"))

        self.assertTrue(success.is_success() and not success.is_failure())
        self.assertTrue(failure.is_failure() and not failure.is_success())

    def test_safe_function_for_error_handling_with_either_monad_returns_correct_either_monad(self):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_try_result: Try[int] = Try.safe(unsafe_function)
        safe_try_result: Try[int] = Try.safe(safe_function)

        self.assertEqual(Failure(Exception("error")), unsafe_try_result)
        self.assertEqual(Success(10), safe_try_result)

    def test_try_monad_unwrap(self):
        success_value = Success(10)
        failure = Failure(TypeError("error"))

        with self.assertRaises(Exception):
            failure.unwrap()

        self.assertEqual(10, success_value.unwrap())
        self.assertEqual(10, success_value.unwrap_or(20))
        self.assertEqual(10, failure.unwrap_or(10))
        self.assertEqual(str(TypeError("error")), str(failure.unwrap_failure_or(TypeError('other error'))))
        self.assertEqual(str(TypeError('other error')), str(success_value.unwrap_failure_or(TypeError('other error'))))
        self.assertEqual(2, Success(2).unwrap_or_else(lambda x: len(x)))
        self.assertEqual(3, Failure(Exception("foo")).unwrap_or_else(lambda x: len(str(x))))

    def test_exception_inspect(self):
        success = Success(10)
        exception = TypeError("error")
        failure = Failure(exception)
        print_mock = Mock()

        self.assertEqual(success, success.inspect(print_mock).inspect_failure(print_mock))
        print_mock.assert_called_with(10)
        self.assertEqual(failure, failure.inspect(print_mock).inspect_failure(print_mock))
        print_mock.assert_called_with(exception)

    def test_try_monad_match_function(self):
        right_value = Success('success')
        left_value = Failure(Exception('error'))

        self.assertTrue(right_value.match(lambda e: str(e) == str(Exception('error')),
                                          lambda v: v == 'success'))
        self.assertTrue(left_value.match(lambda e: str(e) == str(Exception('error')),
                                         lambda v: v == 'success'))

    def test_try_monad_to_either_monad(self):
        success = Success(10)
        failure = Failure(Exception("error"))

        self.assertEqual(Right(10), success.to_either())
        self.assertEqual(Left(Exception("error")), failure.to_either())

    def test_try_monad_to_result_monad(self):
        success = Success(10)
        failure = Failure(Exception("error"))

        self.assertEqual(Ok(10), success.to_result())
        self.assertEqual(Err(Exception("error")), failure.to_result())

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

        self.assertEqual(str(ZeroDivisionError('division by zero')), str(result.unwrap_failure_or(Exception("error"))))
        self.assertEqual(str(ZeroDivisionError('division by zero')), str(result2.unwrap_failure_or(Exception("error"))))
        self.assertEqual(str(ZeroDivisionError('division by zero')), str(result3.unwrap_failure_or(Exception("error"))))
