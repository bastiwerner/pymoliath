import unittest
from functools import partial
from typing import Any

from pymoliath.util import compose, const, curry, flip, identity, pipe


class TestUtil(unittest.TestCase):
    def test_compose(self):
        self.assertEqual(10, compose()(10))
        self.assertEqual(11, compose(lambda x: x)(11))
        self.assertEqual(12, compose(lambda x: x, lambda y: y + 2)(10))

        f = lambda x: x + 1000
        g = lambda y: y * 42
        self.assertEqual(f(g(10)), compose(f, g)(10))

    def test_curry(self):
        add = lambda a, b: a + b

        self.assertEqual(3, curry(add)(1)(2))
        self.assertEqual(3, curry(lambda a: a)(3))
        self.assertEqual(6, curry(lambda a, b, c: a + b + c)(1)(2)(3))
        self.assertEqual(11, curry(lambda a, b=10: a + b)(1))

    def test_curry_with_methods_partials_and_variadic_functions(self):
        class Adder:
            def add(self, a: int, b: int) -> int:
                return a + b

        self.assertEqual(3, curry(Adder().add)(1)(2))
        self.assertEqual(24, curry(partial(lambda a, b, c: a * b * c, 2))(3)(4))
        self.assertEqual((1,), curry(lambda *args: args)(1))
        self.assertEqual(2, curry(lambda a, *, k=1: a + k)(1))

    def test_curry_returns_complete_functions_unchanged(self):
        def increment(x: int) -> int:
            return x + 1

        self.assertIs(increment, curry(increment))

    def test_compose_unpacks_a_tuple_only_when_the_signature_needs_it(self):
        # compose is typed for one-argument functions; unpacking is a runtime convenience.
        add: Any = lambda a, b: a + b
        self.assertEqual(3, compose(add)((1, 2)))
        self.assertEqual((1, 2), compose(lambda pair: pair)((1, 2)))
        self.assertEqual(6, compose(lambda x: x * 2, add)((1, 2)))

    def test_compose_unpacks_into_parameters_with_defaults(self):
        with_default: Any = lambda a, b=10: a + b
        self.assertEqual(3, compose(with_default)((1, 2)))

    def test_compose_passes_the_tuple_to_one_parameter_and_varargs_functions(self):
        self.assertEqual(2, compose(len)((1, 2)))
        variadic: Any = lambda *args: args
        self.assertEqual(((1, 2),), compose(variadic)((1, 2)))

    def test_compose_propagates_type_errors_from_the_functions(self):
        calls: list[Any] = []

        def broken(value: Any) -> Any:
            calls.append(value)
            raise TypeError("inner bug")

        with self.assertRaisesRegex(TypeError, "inner bug"):
            compose(broken)((1, 2))
        self.assertEqual([(1, 2)], calls)

        two_calls: list[Any] = []

        def broken_pair(a: Any, b: Any) -> Any:
            two_calls.append((a, b))
            raise TypeError("inner bug")

        pair_function: Any = broken_pair
        with self.assertRaisesRegex(TypeError, "inner bug"):
            compose(pair_function)((1, 2))
        self.assertEqual([(1, 2)], two_calls)

    def test_curry_propagates_type_errors_from_the_function(self):
        calls: list[Any] = []

        def broken(value: Any) -> Any:
            calls.append(value)
            raise TypeError("inner bug")

        with self.assertRaisesRegex(TypeError, "inner bug"):
            curry(broken)(1)
        self.assertEqual([1], calls)

        def needs_two(a: Any, b: Any) -> Any:
            raise TypeError("inner bug")

        with self.assertRaisesRegex(TypeError, "inner bug"):
            curry(needs_two)(1)(2)

    def test_curry_with_builtins(self):
        self.assertEqual(3, curry(len)("abc"))
        self.assertEqual(7, curry(int)("7"))

    def test_pipe(self):
        self.assertEqual(10, pipe()(10))
        self.assertEqual(11, pipe(lambda x: x)(11))
        self.assertEqual(12, pipe(lambda y: y + 2, lambda x: x)(10))

        f = lambda x: x + 1000
        g = lambda y: y * 42
        self.assertEqual(compose(g, f)(10), pipe(f, g)(10))

    def test_identity(self):
        self.assertEqual(10, identity(10))
        self.assertEqual("a", identity("a"))

    def test_const(self):
        self.assertEqual(10, const(10)("ignored"))
        self.assertEqual(10, const(10)(None))

    def test_flip(self):
        subtract = lambda a, b: a - b

        self.assertEqual(subtract(2, 10), flip(subtract)(10, 2))
