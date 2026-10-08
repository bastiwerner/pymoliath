import math
import unittest
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

from pymoliath import result
from pymoliath.errors import UnwrapError
from pymoliath.option import Nil, Option, Some
from pymoliath.result import (
    RESULT_TYPES,
    Err,
    Ok,
    Result,
    from_option,
    is_err,
    is_ok,
    is_result,
    map2,
    map3,
    result_safe,
)
from pymoliath.util import compose, flow


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

        def ok_function(x: int) -> Result[int, str]:
            return Ok(x + 1)

        def err_function(x: int) -> Result[int, str]:
            return Err("error")

        ok_value: Result[int, str] = Ok(10)
        self.assertEqual(ok_function(10), ok_value.bind(ok_function))
        err_value: Result[int, str] = Err("error")
        self.assertEqual(err_function(10), err_value.bind(ok_function))

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
        ok_value: Result[int, str] = Ok(42)

        def f(x: int) -> Result[int, str]:
            return Ok(x + 1000)

        def g(y: int) -> Result[int, str]:
            return Ok(y * 42)

        self.assertEqual(
            ok_value.bind(f).bind(g), ok_value.bind(lambda x: f(x).bind(g))
        )

        err_value: Result[int, str] = Err("error")

        def f_err(x: int) -> Result[int, str]:
            return Err(f"error {x}")

        def g_err(y: int) -> Result[int, str]:
            return Err(f"error {y}")

        self.assertEqual(
            err_value.bind(f_err).bind(g_err),
            err_value.bind(lambda x: f_err(x).bind(g_err)),
        )

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
        ok_value: Result[int, str] = Ok(42)
        err_value: Result[int, str] = Err("error")

        self.assertEqual(ok_value.apply(Ok(lambda x: x)), ok_value)
        self.assertEqual(err_value.apply(Ok(lambda x: x)), err_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f: Callable[[int], int] = lambda x: x * 42

        self.assertEqual(Ok(x).apply(Ok(f)), Ok(f(x)))
        err_value: Result[int, int] = Err(x)
        self.assertEqual(err_value.apply(Ok(f)), Err(x))

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

        w: Result[int, int] = Ok(42)
        u: Result[Callable[[int], int], int] = Ok(lambda x: x + 42)
        v: Result[Callable[[int], int], int] = Ok(lambda x: x * 42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Ok(composition)))), w.apply(v).apply(u)
        )

        w = Err(42)
        self.assertEqual(
            w.apply(v.apply(u.apply(Ok(composition)))), w.apply(v).apply(u)
        )

    def test_apply_error_precedence(self):
        value: Result[int, str] = Err("value")
        function: Result[Callable[[int], int], str] = Err("function")
        self.assertEqual(Err("function"), value.apply(function))
        self.assertEqual(Err("value"), value.apply(Ok(lambda x: x)))

    def test_map2_and_map3(self):
        one: Result[int, str] = Ok(1)
        two: Result[int, str] = Ok(2)
        first: Result[int, str] = Err("first")
        second: Result[int, str] = Err("second")

        self.assertEqual(Ok(3), map2(one, two, lambda a, b: a + b))
        self.assertEqual(Err("first"), map2(first, second, lambda a, b: a + b))
        self.assertEqual(Err("second"), map2(one, second, lambda a, b: a + b))
        self.assertEqual(Ok(6), map3(one, two, Ok(3), lambda a, b, c: a + b + c))
        self.assertEqual(Err("second"), map3(one, second, first, lambda a, b, c: a))

    def test_either_monad_representation(self):
        ok_value = Ok("a")
        err_value = Err("b")

        self.assertEqual(str(ok_value), "Ok(a)")
        self.assertEqual(str(err_value), f"Err({TypeError('b')})")

    def test_result_is_ok_is_err(self):
        ok_value = Ok(10)
        err_value = Err("error")

        self.assertTrue(ok_value.is_ok() and not ok_value.is_err())
        self.assertTrue(err_value.is_err() and not err_value.is_ok())

    def test_either_monad_unwrap(self):
        ok_value: Result[int, str] = Ok(10)
        failure: Result[int, str] = Err("error")

        with self.assertRaises(UnwrapError) as raised:
            failure.unwrap()
        self.assertEqual(failure, raised.exception.container)

        self.assertEqual(10, ok_value.unwrap())
        self.assertEqual(10, ok_value.unwrap_or(10))
        self.assertEqual("error", failure.unwrap_err_or("other error"))
        self.assertEqual(10, failure.unwrap_or(10))
        self.assertEqual("other error", ok_value.unwrap_err_or("other error"))
        self.assertEqual(10, Ok(10).unwrap_or_else(lambda x: len(x)))
        foo: Result[int, str] = Err("foo")
        self.assertEqual(3, foo.unwrap_or_else(lambda x: len(x)))

    def test_result_inspect(self):
        ok_value = Ok(10)
        error_value = Err("error")
        print_mock = Mock()

        self.assertEqual(ok_value, ok_value.inspect(print_mock).inspect_err(print_mock))
        print_mock.assert_called_with(10)
        self.assertEqual(
            error_value, error_value.inspect(print_mock).inspect_err(print_mock)
        )
        print_mock.assert_called_with("error")

    def test_safe_function_returns_correct_result(self):
        def unsafe_function():
            raise Exception("error")

        def safe_function():
            return 10

        unsafe_result: Result[int, Exception] = result_safe(unsafe_function)
        safe_result: Result[int, Exception] = result_safe(safe_function)

        self.assertTrue(unsafe_result.is_err_and(lambda e: str(e) == "error"))
        self.assertEqual(Ok(10), safe_result)

    def test_safe_narrows_the_caught_exceptions(self):
        caught = result_safe(lambda: 1 // 0, exceptions=(ZeroDivisionError,))
        self.assertTrue(caught.is_err_and(lambda e: type(e) is ZeroDivisionError))

        with self.assertRaises(KeyError):
            result_safe(lambda: {}["missing"], exceptions=(ZeroDivisionError,))

    def test_result_monad_map_and_bind(self):
        ok_value: Result[str, str] = Ok("hello")

        self.assertEqual(
            Err("hello world sucks"),
            (
                ok_value.bind(lambda s: Err(s))
                .map_err(lambda s: f"{s} world")
                .bind_err(lambda e: Err(f"{e} sucks"))
            ),
        )

    def test_result_monad_is_ok_and_is_err_and(self):
        ok_value = Ok(10)
        err_value = Err("error")

        self.assertTrue(ok_value.is_ok_and(lambda v: v > 5))
        self.assertFalse(ok_value.is_ok_and(lambda v: v > 10))
        self.assertFalse(ok_value.is_err_and(lambda e: e == "error"))
        self.assertTrue(err_value.is_err_and(lambda e: e == "error"))
        self.assertFalse(err_value.is_ok_and(lambda v: v > 5))

    def test_result_monad_map_or(self):
        ok_value = Ok(10)
        err_value = Err("error")

        self.assertEqual(11, ok_value.map_or(0, lambda v: v + 1))
        self.assertEqual(0, err_value.map_or(0, lambda v: v + 1))

    def test_result_monad_and_or(self):
        ok_value: Result[int, str] = Ok(10)
        err_value: Result[int, str] = Err("error")

        self.assertEqual(Ok(20), ok_value.and_(Ok(20)))
        self.assertEqual(err_value, err_value.and_(Ok(20)))
        self.assertEqual(ok_value, ok_value.or_(Ok(20)))
        self.assertEqual(Ok(20), err_value.or_(Ok(20)))

    def test_result_monad_zip(self):
        ok_value: Result[int, str] = Ok(10)
        err_value: Result[int, str] = Err("error")

        self.assertEqual(Ok((10, "a")), ok_value.zip(Ok("a")))
        self.assertEqual(Err("error"), ok_value.zip(Err("error")))
        self.assertEqual(err_value, err_value.zip(Ok("a")))

    def test_result_monad_flatten(self):
        self.assertEqual(Ok(10), Ok(Ok(10)).flatten())
        self.assertEqual(Err("error"), Ok(Err("error")).flatten())
        self.assertEqual(Err("error"), Err("error").flatten())

    def test_result_monad_ok_and_err(self):
        ok_value = Ok(10)
        err_value = Err("error")

        self.assertEqual(Some(10), ok_value.ok())
        self.assertEqual(Nil(), ok_value.err())
        self.assertEqual(Nil(), err_value.ok())
        self.assertEqual(Some("error"), err_value.err())

    def test_result_supports_structural_pattern_matching(self):
        def describe(value: Result[int, str]) -> str:
            # No `case _:` fallback: Result is a closed union (Ok[T] | Err[E]),
            # so this is statically exhaustive without one.
            match value:
                case Ok(x):
                    return f"ok {x}"
                case Err(e):
                    return f"err {e}"

        self.assertEqual("ok 10", describe(Ok(10)))
        self.assertEqual("err error", describe(Err("error")))


class TestResultValueSemantics(unittest.TestCase):
    def test_equality_compares_values_not_strings(self):
        self.assertEqual(Ok(Decimal("1.0")), Ok(Decimal("1.00")))
        self.assertNotEqual(Ok(1), Ok("1"))
        self.assertNotEqual(Ok(1), Err(1))
        self.assertEqual({"a": 1, "b": 2}, {"b": 2, "a": 1})
        self.assertEqual(Ok({"a": 1, "b": 2}), Ok({"b": 2, "a": 1}))

        class SameString:
            def __init__(self, value: int) -> None:
                self.value = value

            def __str__(self) -> str:
                return "same"

        self.assertNotEqual(Ok(SameString(1)), Ok(SameString(2)))

    def test_nan_follows_ieee_semantics(self):
        self.assertNotEqual(Ok(float("nan")), Ok(float("nan")))
        self.assertTrue(math.isnan(Ok(float("nan")).unwrap()))

    def test_hashable(self):
        self.assertEqual(1, len({Ok(1), Ok(1)}))
        self.assertEqual(2, len({Ok(1), Err(1)}))
        self.assertEqual({Err("e"): 1}[Err("e")], 1)
        with self.assertRaises(TypeError):
            hash(Ok([]))

    def test_repr_distinguishes_payload_types(self):
        self.assertEqual("Ok('1')", repr(Ok("1")))
        self.assertEqual("Ok(1)", repr(Ok(1)))
        self.assertEqual("Err('e')", repr(Err("e")))
        self.assertEqual("Err(e)", str(Err("e")))

    def test_unwrap_chains_the_original_exception(self):
        error = ValueError("boom")
        failure: Result[int, ValueError] = Err(error)

        with self.assertRaises(UnwrapError) as raised:
            failure.unwrap()
        self.assertIs(error, raised.exception.__cause__)
        self.assertNotIsInstance(raised.exception, ValueError)

    def test_cannot_subclass_at_type_level(self):
        # Sealing is static (@final); the runtime hook was removed.
        self.assertTrue(getattr(Ok, "__final__", False))
        self.assertTrue(getattr(Err, "__final__", False))

    def test_runtime_checks(self):
        self.assertTrue(is_result(Ok(1)))
        self.assertTrue(is_result(Err(1)))
        self.assertFalse(is_result(Some(1)))
        self.assertIsInstance(Ok(1), RESULT_TYPES)

        value: Result[int, str] = Ok(1)
        self.assertTrue(is_ok(value))
        self.assertFalse(is_err(value))


class TestResultFeatures(unittest.TestCase):
    def test_match_is_keyword_only(self):
        value: Result[int, str] = Ok(1)
        self.assertEqual("ok 1", value.match(ok=lambda x: f"ok {x}", err=lambda e: e))
        error: Result[int, str] = Err("e")
        self.assertEqual("e", error.match(ok=lambda x: f"ok {x}", err=lambda e: e))
        with self.assertRaises(TypeError):
            value.match(lambda x: x, lambda e: e)  # type: ignore[misc]  # pyright: ignore[reportCallIssue]

    def test_swap(self):
        self.assertEqual(Err(1), Ok(1).swap())
        self.assertEqual(Ok("e"), Err("e").swap())

    def test_map_or_else(self):
        ok_value: Result[int, str] = Ok(2)
        err_value: Result[int, str] = Err("abc")
        self.assertEqual(4, ok_value.map_or_else(len, lambda x: x * 2))
        self.assertEqual(3, err_value.map_or_else(len, lambda x: x * 2))

    def test_filter(self):
        value: Result[int, str] = Ok(-1)
        self.assertEqual(Err("negative"), value.filter(lambda x: x > 0, "negative"))
        positive: Result[int, str] = Ok(1)
        self.assertEqual(Ok(1), positive.filter(lambda x: x > 0, "negative"))
        self.assertEqual(Err("e"), Err("e").filter(lambda x: x > 0, "negative"))

    def test_merge(self):
        self.assertEqual(1, Ok(1).merge())
        self.assertEqual(2, Err(2).merge())

    def test_transpose(self):
        self.assertEqual(Some(Ok(1)), Ok(Some(1)).transpose())
        self.assertEqual(Nil(), Ok(Nil()).transpose())
        error: Result[Option[int], str] = Err("e")
        self.assertEqual(Some(Err("e")), error.transpose())

    def test_aliases(self):
        value: Result[int, str] = Ok(1)
        self.assertEqual(
            value.bind(lambda x: Ok(x + 1)), value.and_then(lambda x: Ok(x + 1))
        )
        error: Result[int, str] = Err("abc")
        self.assertEqual(Ok(3), error.or_else(lambda e: Ok(len(e))))

    def test_from_option(self):
        self.assertEqual(Ok(1), from_option(Some(1), "missing"))
        self.assertEqual(Err("missing"), from_option(Nil(), "missing"))

    def test_curried_functions_with_flow(self):
        def double(x: int) -> int:
            return x * 2

        def validate(x: int) -> Result[int, str]:
            return Ok(x) if x < 10 else Err("too big")

        start: Result[int, str] = Ok(2)
        self.assertEqual(
            4,
            flow(start, result.map(double), result.bind(validate), result.unwrap_or(0)),
        )
        too_big: Result[int, str] = Ok(5)
        self.assertEqual(
            0,
            flow(
                too_big, result.map(double), result.bind(validate), result.unwrap_or(0)
            ),
        )
        seen: list[Any] = []
        error: Result[int, str] = Err("e")
        self.assertEqual(
            "E",
            flow(
                error,
                result.inspect(seen.append),
                result.inspect_err(seen.append),
                result.map_err(str.upper),
                result.unwrap_or_else(lambda e: e),
            ),
        )
        self.assertEqual(["e"], seen)

        # Lambdas passed to curried functions are not inferred (the input type is not known
        # yet), so pipelines use annotated functions.
        def recover(e: str) -> Result[int, str]:
            return Ok(len(e))

        self.assertEqual(Ok(1), result.bind_err(recover)(error))
