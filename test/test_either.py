import unittest
from collections.abc import Callable
from unittest.mock import Mock

from pymoliath import either
from pymoliath.either import (
    EITHER_TYPES,
    Either,
    Left,
    Right,
    either_safe,
    is_either,
    is_left,
    is_right,
    map2,
    map3,
)
from pymoliath.errors import UnwrapError
from pymoliath.maybe import Just, Nothing
from pymoliath.util import compose, flow


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
        left_function = lambda x: Left(f"{x}")

        self.assertEqual(right_function(10), Right(10).bind(right_function))
        self.assertEqual(left_function(10), Left(10).bind_left(left_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        right_value = Right(10)
        left_value = Left("err")

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

        self.assertEqual(
            right_value.bind(f).bind(g), right_value.bind(lambda x: f(x).bind(g))
        )

        left_value = Left(42)
        f = lambda x: Left(x + 1000)
        g = lambda y: Left(y * 42)

        self.assertEqual(
            left_value.bind(f).bind(g), left_value.bind(lambda x: f(x).bind(g))
        )

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

        w: Either[int, int] = Right(42)
        u: Either[int, Callable[[int], int]] = Right(lambda x: x + 42)
        v: Either[int, Callable[[int], int]] = Right(lambda x: x * 42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Right(composition)))), w.apply(v).apply(u)
        )

        w = Left(42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Right(composition)))), w.apply(v).apply(u)
        )

    def test_apply_left_precedence(self):
        value: Either[str, int] = Left("value")
        function: Either[str, Callable[[int], int]] = Left("function")
        self.assertEqual(Left("function"), value.apply(function))

    def test_map2_and_map3(self):
        one: Either[str, int] = Right(1)
        first: Either[str, int] = Left("first")
        second: Either[str, int] = Left("second")
        self.assertEqual(Right(3), map2(one, Right(2), lambda a, b: a + b))
        self.assertEqual(Left("first"), map2(first, second, lambda a, b: a))
        self.assertEqual(
            Right(6), map3(one, Right(2), Right(3), lambda a, b, c: a + b + c)
        )

    def test_either_monad_representation(self):
        right_value = Right("a")
        left_value = Left("b")

        self.assertEqual(str(right_value), "Right(a)")
        self.assertEqual(str(left_value), "Left(b)")

    def test_either_is_left_is_right(self):
        right = Right(10)
        left = Left("error")

        self.assertTrue(right.is_right() and not right.is_left())
        self.assertTrue(left.is_left() and not left.is_right())

    def test_safe_function_for_error_handling_with_either_monad_returns_correct_either_monad(
        self,
    ):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_either_result: Either[Exception, int] = either_safe(unsafe_function)
        safe_either_result: Either[Exception, int] = either_safe(safe_function)

        self.assertTrue(unsafe_either_result.is_left_and(lambda e: str(e) == "error"))
        self.assertEqual(Right(10), safe_either_result)
        self.assertTrue(
            either_safe(lambda: int("x"), exceptions=(ValueError,)).is_left()
        )
        with self.assertRaises(KeyError):
            either_safe(lambda: {}["missing"], exceptions=(ValueError,))

    def test_either_monad_unwrap(self):
        right_value: Either[str, str] = Right("right")
        left_value: Either[str, str] = Left("left")

        with self.assertRaises(UnwrapError):
            left_value.unwrap()

        self.assertEqual("right", right_value.unwrap())
        self.assertEqual("right", right_value.unwrap_or("default"))
        self.assertEqual("left", left_value.unwrap_left_or("default"))
        self.assertEqual("default", right_value.unwrap_left_or("default"))
        self.assertEqual("default", left_value.unwrap_or("default"))
        self.assertEqual(2, Right(2).unwrap_or_else(lambda x: len(x)))
        foo: Either[str, int] = Left("foo")
        self.assertEqual(3, foo.unwrap_or_else(lambda x: len(x)))

    def test_either_inspect(self):
        right_value = Right("right")
        left_value = Left("left")
        print_mock = Mock()

        self.assertEqual(
            right_value, right_value.inspect(print_mock).inspect_left(print_mock)
        )
        print_mock.assert_called_with("right")
        self.assertEqual(
            left_value, left_value.inspect(print_mock).inspect_left(print_mock)
        )
        print_mock.assert_called_with("left")

    def test_either_monad_map_and_bind(self):
        right: Either[str, str] = Right("hello")

        self.assertEqual(
            Left("hello world sucks"),
            (
                right.bind(lambda s: Left(s))
                .map_left(lambda s: f"{s} world")
                .bind_left(lambda s: Left(f"{s} sucks"))
            ),
        )

    def test_either_monad_match_function(self):
        right_value = Right("right")
        left_value = Left("left")

        self.assertTrue(
            right_value.match(left=lambda e: e == "left", right=lambda v: v == "right")
        )
        self.assertTrue(
            left_value.match(left=lambda e: e == "left", right=lambda v: v == "right")
        )

    def test_either_monad_is_right_and_is_left_and(self):
        right_value = Right(10)
        left_value = Left("error")

        self.assertTrue(right_value.is_right_and(lambda v: v > 5))
        self.assertFalse(right_value.is_right_and(lambda v: v > 10))
        self.assertFalse(right_value.is_left_and(lambda e: e == "error"))
        self.assertTrue(left_value.is_left_and(lambda e: e == "error"))
        self.assertFalse(left_value.is_right_and(lambda v: v > 5))

    def test_either_monad_map_or(self):
        right_value = Right(10)
        left_value = Left("error")

        self.assertEqual(11, right_value.map_or(0, lambda v: v + 1))
        self.assertEqual(0, left_value.map_or(0, lambda v: v + 1))

    def test_either_monad_and_or(self):
        right_value = Right(10)
        left_value: Either[str, int] = Left("error")

        self.assertEqual(Right(20), right_value.and_(Right(20)))
        self.assertEqual(left_value, left_value.and_(Right(20)))
        self.assertEqual(right_value, right_value.or_(Right(20)))
        self.assertEqual(Right(20), left_value.or_(Right(20)))

    def test_either_monad_zip(self):
        right_value: Either[str, int] = Right(10)
        left_value: Either[str, int] = Left("error")

        self.assertEqual(Right((10, "a")), right_value.zip(Right("a")))
        self.assertEqual(Left("error"), right_value.zip(Left("error")))
        self.assertEqual(left_value, left_value.zip(Right("a")))

    def test_either_monad_flatten(self):
        self.assertEqual(Right(10), Right(Right(10)).flatten())
        self.assertEqual(Left("error"), Right(Left("error")).flatten())
        self.assertEqual(Left("error"), Left("error").flatten())

    def test_either_monad_right_and_left(self):
        right_value = Right(10)
        left_value = Left("error")

        self.assertEqual(Just(10), right_value.right())
        self.assertEqual(Nothing(), right_value.left())
        self.assertEqual(Nothing(), left_value.right())
        self.assertEqual(Just("error"), left_value.left())

    def test_either_supports_structural_pattern_matching(self):
        def describe(value: Either[str, int]) -> str:
            # No `case _:` fallback: Either is a closed union (Left[L] | Right[R]),
            # so this is statically exhaustive without one.
            match value:
                case Left(e):
                    return f"left {e}"
                case Right(v):
                    return f"right {v}"

        self.assertEqual("right 10", describe(Right(10)))
        self.assertEqual("left error", describe(Left("error")))


class TestEitherValueSemantics(unittest.TestCase):
    def test_equality_and_hash(self):
        self.assertEqual(Right({"a": 1, "b": 2}), Right({"b": 2, "a": 1}))
        self.assertNotEqual(Right(1), Right("1"))
        self.assertNotEqual(Right(1), Left(1))
        self.assertEqual(1, len({Left("e"), Left("e")}))
        with self.assertRaises(TypeError):
            hash(Right([]))

    def test_repr(self):
        self.assertEqual("Right('1')", repr(Right("1")))
        self.assertEqual("Left('e')", repr(Left("e")))
        self.assertEqual("Left(e)", str(Left("e")))

    def test_unwrap_chains_the_original_exception(self):
        error = ValueError("boom")
        failure: Either[ValueError, int] = Left(error)
        with self.assertRaises(UnwrapError) as raised:
            failure.unwrap()
        self.assertIs(error, raised.exception.__cause__)

    def test_runtime_checks(self):
        value: Either[str, int] = Right(1)
        self.assertTrue(is_right(value))
        self.assertFalse(is_left(value))
        self.assertTrue(is_either(Left(1)))
        self.assertFalse(is_either(1))
        self.assertIsInstance(Right(1), EITHER_TYPES)


class TestEitherFeatures(unittest.TestCase):
    def test_swap_and_merge(self):
        self.assertEqual(Left(1), Right(1).swap())
        self.assertEqual(Right("e"), Left("e").swap())
        self.assertEqual(1, Right(1).merge())
        self.assertEqual(2, Left(2).merge())

    def test_map_or_else(self):
        value: Either[str, int] = Left("abc")
        self.assertEqual(3, value.map_or_else(len, lambda x: x * 2))
        self.assertEqual(4, Right(2).map_or_else(len, lambda x: x * 2))

    def test_filter(self):
        value: Either[str, int] = Right(-1)
        self.assertEqual(Left("negative"), value.filter(lambda x: x > 0, "negative"))

    def test_aliases(self):
        value: Either[str, int] = Right(1)
        self.assertEqual(Right(2), value.and_then(lambda x: Right(x + 1)))
        error: Either[str, int] = Left("abc")
        self.assertEqual(Right(3), error.or_else(lambda e: Right(len(e))))

    def test_curried_functions_with_flow(self):
        def double(x: int) -> int:
            return x * 2

        def check(x: int) -> Either[str, int]:
            return Right(x) if x < 10 else Left("too big")

        start: Either[str, int] = Right(2)
        self.assertEqual(
            4, flow(start, either.map(double), either.bind(check), either.unwrap_or(0))
        )
        error: Either[str, int] = Left("abc")
        self.assertEqual(
            3, flow(error, either.map_left(str.upper), either.unwrap_or_else(len))
        )
