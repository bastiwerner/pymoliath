import unittest

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
