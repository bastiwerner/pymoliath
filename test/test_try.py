import unittest
from collections.abc import Callable
from unittest.mock import Mock

from pymoliath import Right, Left, Ok, Err
from pymoliath.exception import (
    TRY_TYPES,
    Try,
    Success,
    Failure,
    is_failure,
    is_success,
    is_try,
    map2,
    map3,
    safe,
)
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
        failure_function = lambda x: Failure(TypeError("error"))

        self.assertEqual(success_function(10), Success(10).bind(success_function))
        self.assertEqual(
            failure_function(10), Failure(TypeError("error")).bind(failure_function)
        )

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        success_value = Success(10)
        failure = Failure(TypeError("error"))

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

        self.assertEqual(
            success_value.bind(f).bind(g), success_value.bind(lambda x: f(x).bind(g))
        )

        failure_value = Failure(ValueError("error"))
        f = lambda x: Failure(ValueError(f"error {x}"))
        g = lambda y: Failure(ValueError(f"error {y}"))

        self.assertEqual(
            failure_value.bind(f).bind(g), failure_value.bind(lambda x: f(x).bind(g))
        )

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(Success(10).map(lambda x: x), Success(10))
        self.assertEqual(
            Failure(TypeError("error")).map(lambda x: x), Failure(TypeError("error"))
        )

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        success_value = Success(42)
        failure_value = Failure(ValueError("error"))

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
        failure_value = Failure(TypeError("error"))

        self.assertEqual(success_value.apply(Success(lambda x: x)), success_value)
        self.assertEqual(failure_value.apply(Success(lambda x: x)), failure_value)

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

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """

        def composition(
            f: Callable[[int], int],
        ) -> Callable[[Callable[[int], int]], Callable[[int], int]]:
            return lambda g: lambda x: f(g(x))

        w: Try[int] = Success(42)
        u: Try[Callable[[int], int]] = Success(lambda x: x + 42)
        v: Try[Callable[[int], int]] = Success(lambda x: x * 42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Success(composition)))), w.apply(v).apply(u)
        )

        w = Failure(TypeError(42))
        self.assertEqual(
            w.apply(v.apply(u.apply(Success(composition)))), w.apply(v).apply(u)
        )

    def test_map2_and_map3(self):
        self.assertEqual(Success(3), map2(Success(1), Success(2), lambda a, b: a + b))
        self.assertEqual(
            Failure(ValueError("first")),
            map2(Failure(ValueError("first")), Failure(ValueError("second")), max),
        )
        self.assertEqual(
            Success(6),
            map3(Success(1), Success(2), Success(3), lambda a, b, c: a + b + c),
        )

    def test_either_monad_representation(self):
        success_value = Success("a")
        failure_value = Failure(TypeError("b"))

        self.assertEqual(str(success_value), "Success(a)")
        self.assertEqual(str(failure_value), f"Failure({TypeError('b')})")

    def test_either_try_is_left_is_right_is_success_if_failure(self):
        success = Success(10)
        failure = Failure(TypeError("error"))

        self.assertTrue(success.is_success() and not success.is_failure())
        self.assertTrue(failure.is_failure() and not failure.is_success())

    def test_safe_function_for_error_handling_with_either_monad_returns_correct_either_monad(
        self,
    ):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_try_result: Try[int] = safe(unsafe_function)
        safe_try_result: Try[int] = safe(safe_function)

        self.assertEqual(Failure(Exception("error")), unsafe_try_result)
        self.assertEqual(Success(10), safe_try_result)

    def test_try_monad_unwrap(self):
        success_value = Success(10)
        failure: Try[int] = Failure(TypeError("error"))

        with self.assertRaises(TypeError):
            failure.unwrap()

        self.assertEqual(10, success_value.unwrap())
        self.assertEqual(10, success_value.unwrap_or(20))
        self.assertEqual(10, failure.unwrap_or(10))
        self.assertEqual(
            str(TypeError("error")),
            str(failure.unwrap_failure_or(TypeError("other error"))),
        )
        self.assertEqual(
            str(TypeError("other error")),
            str(success_value.unwrap_failure_or(TypeError("other error"))),
        )
        self.assertEqual(2, Success(2).unwrap_or_else(lambda x: len(str(x))))
        foo: Try[int] = Failure(Exception("foo"))
        self.assertEqual(3, foo.unwrap_or_else(lambda x: len(str(x))))

    def test_exception_inspect(self):
        success = Success(10)
        exception = TypeError("error")
        failure = Failure(exception)
        print_mock = Mock()

        self.assertEqual(
            success, success.inspect(print_mock).inspect_failure(print_mock)
        )
        print_mock.assert_called_with(10)
        self.assertEqual(
            failure, failure.inspect(print_mock).inspect_failure(print_mock)
        )
        print_mock.assert_called_with(exception)

    def test_try_monad_match_function(self):
        right_value = Success("success")
        left_value = Failure(Exception("error"))

        self.assertTrue(
            right_value.match(
                success=lambda v: v == "success",
                failure=lambda e: str(e) == str(Exception("error")),
            )
        )
        self.assertTrue(
            left_value.match(
                success=lambda v: v == "success",
                failure=lambda e: str(e) == str(Exception("error")),
            )
        )

    def test_try_supports_structural_pattern_matching(self):
        def describe(value: Try[int]) -> str:
            # No `case _:` fallback: Try is a closed union (Success[T] | Failure),
            # so this is statically exhaustive without one.
            match value:
                case Success(x):
                    return f"success {x}"
                case Failure(e):
                    return f"failure {e}"

        self.assertEqual("success 10", describe(Success(10)))
        self.assertEqual("failure boom", describe(Failure(Exception("boom"))))

    def test_try_monad_to_either_monad(self):
        success = Success(10)
        failure = Failure(Exception("error"))

        self.assertEqual(Right(10), success.to_either())
        # Left/Err compare exceptions by identity, so compare the messages.
        self.assertEqual(Left("error"), failure.to_either().map_left(str))

    def test_try_monad_to_result_monad(self):
        success = Success(10)
        failure = Failure(Exception("error"))

        self.assertEqual(Ok(10), success.to_result())
        self.assertEqual(Err("error"), failure.to_result().map_err(str))

    def test_try_monad_is_success_and_is_failure_and(self):
        success_value = Success(10)
        failure_value = Failure(Exception("error"))

        self.assertTrue(success_value.is_success_and(lambda v: v > 5))
        self.assertFalse(success_value.is_success_and(lambda v: v > 10))
        self.assertFalse(success_value.is_failure_and(lambda e: str(e) == "error"))
        self.assertTrue(failure_value.is_failure_and(lambda e: str(e) == "error"))
        self.assertFalse(failure_value.is_success_and(lambda v: v > 5))

    def test_try_monad_map_or(self):
        success_value = Success(10)
        failure_value = Failure(Exception("error"))

        self.assertEqual(11, success_value.map_or(0, lambda v: v + 1))
        self.assertEqual(0, failure_value.map_or(0, lambda v: v + 1))

    def test_try_monad_and_or(self):
        success_value = Success(10)
        failure_value: Try[int] = Failure(Exception("error"))

        self.assertEqual(Success(20), success_value.and_(Success(20)))
        self.assertEqual(failure_value, failure_value.and_(Success(20)))
        self.assertEqual(success_value, success_value.or_(Success(20)))
        self.assertEqual(Success(20), failure_value.or_(Success(20)))

    def test_try_monad_zip(self):
        success_value = Success(10)
        failure_value = Failure(Exception("error"))

        self.assertEqual(Success((10, "a")), success_value.zip(Success("a")))
        self.assertEqual(failure_value, success_value.zip(failure_value))
        self.assertEqual(failure_value, failure_value.zip(Success("a")))

    def test_try_monad_flatten(self):
        self.assertEqual(Success(10), Success(Success(10)).flatten())
        self.assertEqual(
            Failure(Exception("error")),
            Success(Failure(Exception("error"))).flatten(),
        )
        self.assertEqual(
            Failure(Exception("error")), Failure(Exception("error")).flatten()
        )

    def test_try_either_monad(self):
        def divide(dividen: int, divisor: int):
            return dividen / divisor

        def try_divide(dividen: int, divisor: int) -> Try[float]:
            return map2(Success(dividen), Success(divisor), divide)

        def try_divide_map(dividen: int, divisor: int) -> Try[float]:
            return Success(divide).map(lambda f: f(dividen, divisor))

        def curried_divide(dividen: int) -> Callable[[int], float]:
            return lambda divisor: divide(dividen, divisor)

        result: Try[float] = try_divide(10, 0)
        result2: Try[float] = Success(0).apply(Success(10).map(curried_divide))
        result3: Try[float] = try_divide_map(10, 0)

        self.assertEqual(
            str(ZeroDivisionError("division by zero")),
            str(result.unwrap_failure_or(Exception("error"))),
        )
        self.assertEqual(
            str(ZeroDivisionError("division by zero")),
            str(result2.unwrap_failure_or(Exception("error"))),
        )
        self.assertEqual(
            str(ZeroDivisionError("division by zero")),
            str(result3.unwrap_failure_or(Exception("error"))),
        )

    def test_apply2(self):
        def increment(x: int) -> int:
            return x + 1

        function: Try[Callable[[int], int]] = Success(increment)
        failed: Try[Callable[[int], int]] = Failure(ValueError("function"))
        value: Try[int] = Success(1)
        error: Try[int] = Failure(ValueError("value"))

        self.assertEqual(Success(2), function.apply2(value))
        self.assertEqual(Failure(ValueError("value")), function.apply2(error))
        self.assertEqual(Failure(ValueError("function")), failed.apply2(error))
        self.assertEqual(value.apply(function), function.apply2(value))

        add: Try[Callable[[int], Callable[[int], int]]] = Success(
            lambda a: lambda b: a + b
        )
        self.assertEqual(Success(3), add.apply2(Success(1)).apply2(Success(2)))


class TestTryValueSemantics(unittest.TestCase):
    def test_failure_equality_compares_exception_type_and_args(self):
        self.assertEqual(Failure(ValueError("x")), Failure(ValueError("x")))
        self.assertNotEqual(Failure(ValueError("x")), Failure(ValueError("y")))
        self.assertNotEqual(Failure(ValueError("x")), Failure(TypeError("x")))
        self.assertEqual(1, len({Failure(ValueError("x")), Failure(ValueError("x"))}))

    def test_success_equality_and_hash(self):
        self.assertEqual(Success({"a": 1, "b": 2}), Success({"b": 2, "a": 1}))
        self.assertNotEqual(Success(1), Success("1"))
        self.assertEqual(1, len({Success(1), Success(1)}))
        with self.assertRaises(TypeError):
            hash(Success([]))

    def test_repr(self):
        self.assertEqual("Success('1')", repr(Success("1")))
        self.assertEqual(
            "Failure(ValueError('boom'))", repr(Failure(ValueError("boom")))
        )
        self.assertEqual("Failure(boom)", str(Failure(ValueError("boom"))))

    def test_failure_requires_an_exception(self):
        with self.assertRaises(TypeError):
            Failure("not an exception")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_unwrap_reraises_the_original_exception(self):
        with self.assertRaises(ValueError):
            Failure(ValueError("boom")).unwrap()

    def test_runtime_checks(self):
        value: Try[int] = Success(1)
        self.assertTrue(is_success(value))
        self.assertFalse(is_failure(value))
        self.assertTrue(is_try(Failure(ValueError())))
        self.assertFalse(is_try(1))
        self.assertIsInstance(Success(1), TRY_TYPES)

    def test_safe_narrows_the_caught_exceptions(self):
        self.assertTrue(safe(lambda: int("x"), exceptions=(ValueError,)).is_failure())
        with self.assertRaises(KeyError):
            safe(lambda: {}["missing"], exceptions=(ValueError,))


class TestTryFeatures(unittest.TestCase):
    def test_map_or_else(self):
        failure: Try[int] = Failure(ValueError("abc"))
        self.assertEqual(3, failure.map_or_else(lambda e: len(str(e)), lambda x: x * 2))
        self.assertEqual(4, Success(2).map_or_else(lambda e: 0, lambda x: x * 2))

    def test_and_then(self):
        self.assertEqual(Success(2), Success(1).and_then(lambda x: Success(x + 1)))
