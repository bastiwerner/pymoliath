import unittest
from unittest.mock import Mock

from pymoliath.either import Right, Left, Ok, Err, Result
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
        err_function = lambda x: Err("error")

        self.assertEqual(ok_function(10), Ok(10).bind(ok_function))
        self.assertEqual(err_function(10), Err("error").bind(ok_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        ok_value = Ok(10)
        failure = Err("error")

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

        err_value = Err("error")
        f = lambda x: Err(f'error {x}')
        g = lambda y: Err(f'error {y}')

        self.assertEqual(err_value.bind(f).bind(g), err_value.bind(lambda x: f(x).bind(g)))

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Ok(10).map(lambda x: x), Ok(10))
        self.assertEqual(Err("error").map(lambda x: x), Err("error"))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        ok_value = Ok(42)
        err_value = Err("error")

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
        err_value = Err("error")

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
        self.assertEqual(Err(x).apply(Ok(f)), Err(x))

        self.assertEqual(Ok(f).apply2(Ok(x)), Ok(f(x)))
        self.assertEqual(Ok(f).apply2(Err(x)), Err(x))

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

        w = Err(42)
        self.assertEqual(w.apply(v.apply(u.apply(Err(42)))), w.apply(v).apply(u))
        self.assertEqual(Ok(lambda f, g: compose(f, g)).apply2(u).apply2(v).apply2(w), u.apply2(v.apply2(w)))

    def test_either_monad_representation(self):
        ok_value = Ok("a")
        err_value = Err("b")

        self.assertEqual(str(ok_value), 'Ok(a)')
        self.assertEqual(str(err_value), f'Err({TypeError("b")})')

    def test_result_is_ok_is_err(self):
        ok_value = Ok(10)
        err_value = Err("error")

        self.assertTrue(ok_value.is_ok() and not ok_value.is_err())
        self.assertTrue(err_value.is_err() and not err_value.is_ok())

    def test_either_monad_unwrap(self):
        ok_value = Ok(10)
        failure = Err("error")

        with self.assertRaises(Exception):
            failure.unwrap()

        self.assertEqual(10, ok_value.unwrap())
        self.assertEqual(10, ok_value.unwrap_or(10))
        self.assertEqual("error", failure.unwrap_err_or("other error"))
        self.assertEqual(10, failure.unwrap_or(10))
        self.assertEqual("other error", ok_value.unwrap_err_or("other error"))
        self.assertEqual(10, Ok(10).unwrap_or_else(lambda x: len(x)))
        self.assertEqual(3, Err("foo").unwrap_or_else(lambda x: len(x)))

    def test_result_inspect(self):
        ok_value = Ok(10)
        error_value = Err("error")
        print_mock = Mock()

        self.assertEqual(ok_value, ok_value.inspect(print_mock).inspect_err(print_mock))
        print_mock.assert_called_with(10)
        self.assertEqual(error_value, error_value.inspect(print_mock).inspect_err(print_mock))
        print_mock.assert_called_with("error")

    def test_safe_function_returns_correct_result(self):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_result: Result[int] = Result.safe(unsafe_function)
        safe_result: Result[int] = Result.safe(safe_function)

        self.assertEqual(Err(Exception('error')), unsafe_result)
        self.assertEqual(Ok(10), safe_result)

    def test_result_monad_to_either_monad(self):
        ok_value = Ok(10)
        err_value = Err(Exception("error"))

        self.assertEqual(Right(10), ok_value.to_either())
        self.assertEqual(Left(Exception("error")), err_value.to_either())

    def test_result_monad_map_and_bind(self):
        ok_value = Ok('hello')

        self.assertEqual(Err('hello world sucks'), (ok_value
                                                    .bind(lambda s: Err(s))
                                                    .map_err(lambda s: f'{s} world')
                                                    .bind_err(lambda e: Err(f'{e} sucks'))))

    def test_result_abstract_class_raise_error(self):
        with self.assertRaises(TypeError):
            Result()

    def test_result_monad_match_function(self):
        right_value = Right('right')
        left_value = Left('left')

        self.assertTrue(right_value.match(lambda e: e == 'left',
                                          lambda v: v == 'right'))
        self.assertTrue(left_value.match(lambda e: e == 'left',
                                         lambda v: v == 'right'))
