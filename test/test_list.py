import unittest
from typing import Tuple

from pymoliath.list import ListMonad
from pymoliath.option import Nil, Some
from pymoliath.util import compose


class TestMonadList(unittest.TestCase):
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
        it’s the same as list taking the value and applying the function to it.
        """
        list_function = lambda x: ListMonad([x + 1, x + 2])

        self.assertEqual(list_function(10), ListMonad([10]).bind(list_function))

    def test_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        list_value = ListMonad(["Hello", "world"])

        self.assertEqual(list_value, list_value.bind(lambda x: ListMonad([x])))

    def test_list_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        list_value = ListMonad([42])
        f = lambda x: ListMonad([x + 1000])
        g = lambda y: ListMonad([y * 42])

        self.assertEqual(
            list_value.bind(f).bind(g), list_value.bind(lambda x: f(x).bind(g))
        )

    def test_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        self.assertEqual(ListMonad([10]).map(lambda x: x), ListMonad([10]))

    def test_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        list_value = ListMonad([42])

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(list_value.map(compose(f, g)), list_value.map(g).map(f))

    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        list_value = ListMonad([42, 10])

        self.assertEqual(list_value.apply(ListMonad([lambda x: x])), list_value)
        self.assertEqual(ListMonad([lambda x: x]).apply2(list_value), list_value)

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(ListMonad([x]).apply(ListMonad([f])), ListMonad([f(x)]))
        self.assertEqual(ListMonad([f]).apply2(ListMonad([x])), ListMonad([f(x)]))

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = ListMonad([42])
        u = ListMonad([lambda x: x + 42])
        v = ListMonad([lambda x: x * 42])
        composition = lambda f, g: compose(f, g)

        self.assertEqual(
            w.apply(v.apply(u.apply(ListMonad([composition])))), w.apply(v).apply(u)
        )
        self.assertEqual(
            ListMonad([composition]).apply2(u).apply2(v).apply2(w),
            u.apply2(v.apply2(w)),
        )

    def test_list_monad_representation(self):
        list = ListMonad(["a"])

        self.assertTrue(isinstance(ListMonad(["a"]), ListMonad))  # pyright: ignore[reportUnnecessaryIsInstance]
        self.assertEqual("ListMonad(['a'])", str(list))

    def test_list_monad_with_nested_list_and_enumeration_returns_correct_result(self):
        result: ListMonad[Tuple[int, str]] = (
            ListMonad([["hello", "world"]])
            .bind(lambda d: ListMonad(enumerate(d)))
            .map(lambda tuple_result: (tuple_result[0], tuple_result[1]))
        )

        self.assertEqual([(0, "hello"), (1, "world")], result)

    def test_list_monad_to_list_returns_correct_python_list(self):
        result = ListMonad([1, 2, 3, 4])

        self.assertEqual([1, 2, 3, 4], result.to_list())
        self.assertEqual("[1, 2, 3, 4]", str(result.to_list()))

    def test_list_monad_with_filter_and_take_returns_corrected_filtered_list(self):
        result = ListMonad([1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9], result.take(20))
        self.assertEqual([], result.skip(10))
        self.assertEqual([1, 2, 3, 4], result.take(4))
        self.assertEqual([6, 7, 8, 9], result.skip(5))
        self.assertEqual([3, 4], result.take(4).filter(lambda x: x > 2))

    def test_list_monad_fold_and_reduce(self):
        values = ListMonad([1, 2, 3, 4])

        self.assertEqual(10, values.fold(0, lambda acc, v: acc + v))
        self.assertEqual(Some(10), values.reduce(lambda acc, v: acc + v))
        self.assertEqual(Nil(), ListMonad().reduce(lambda acc, v: acc + v))

    def test_list_monad_enumerate(self):
        self.assertEqual(
            ListMonad([(0, "a"), (1, "b")]), ListMonad(["a", "b"]).enumerate()
        )

    def test_list_monad_zip_and_zip_with(self):
        values = ListMonad([1, 2, 3])

        self.assertEqual(ListMonad([(1, "a"), (2, "b")]), values.zip(["a", "b"]))
        self.assertEqual(
            ListMonad(["1a", "2b"]),
            values.zip_with(["a", "b"], lambda a, b: f"{a}{b}"),
        )

    def test_list_monad_chain_and_flatten(self):
        self.assertEqual(ListMonad([1, 2, 3, 4]), ListMonad([1, 2]).chain([3, 4]))
        self.assertEqual(
            ListMonad([1, 2, 3, 4]),
            ListMonad([[1, 2], [3, 4]]).flatten(),
        )

    def test_list_monad_any_all(self):
        values = ListMonad([1, 2, 3])

        self.assertTrue(values.any(lambda v: v > 2))
        self.assertFalse(values.any(lambda v: v > 3))
        self.assertTrue(values.all(lambda v: v > 0))
        self.assertFalse(values.all(lambda v: v > 1))

    def test_list_monad_find_and_position(self):
        values = ListMonad([1, 2, 3])

        self.assertEqual(Some(2), values.find(lambda v: v > 1))
        self.assertEqual(Nil(), values.find(lambda v: v > 3))
        self.assertEqual(Some(1), values.position(lambda v: v > 1))
        self.assertEqual(Nil(), values.position(lambda v: v > 3))

    def test_list_monad_min_and_max(self):
        values = ListMonad([3, 1, 2])

        self.assertEqual(Some(1), values.min())
        self.assertEqual(Some(3), values.max())
        self.assertEqual(Nil(), ListMonad().min())
        self.assertEqual(Nil(), ListMonad().max())

    def test_list_monad_min_by_key_and_max_by_key(self):
        values = ListMonad(["aaa", "b", "cc"])

        self.assertEqual(Some("b"), values.min_by_key(lambda v: len(v)))
        self.assertEqual(Some("aaa"), values.max_by_key(lambda v: len(v)))
        self.assertEqual(Nil(), ListMonad().min_by_key(lambda v: len(v)))
        self.assertEqual(Nil(), ListMonad().max_by_key(lambda v: len(v)))

    def test_list_monad_partition(self):
        values = ListMonad([1, 2, 3, 4, 5])

        self.assertEqual(
            (ListMonad([2, 4]), ListMonad([1, 3, 5])),
            values.partition(lambda v: v % 2 == 0),
        )

    def test_list_monad_take_while_and_skip_while(self):
        values = ListMonad([1, 2, 3, 1, 2])

        self.assertEqual(ListMonad([1, 2]), values.take_while(lambda v: v < 3))
        self.assertEqual(ListMonad([3, 1, 2]), values.skip_while(lambda v: v < 3))

    def test_list_monad_step_by(self):
        values = ListMonad([0, 1, 2, 3, 4, 5])

        self.assertEqual(ListMonad([0, 2, 4]), values.step_by(2))

        with self.assertRaises(ValueError):
            values.step_by(0)

    def test_list_monad_rev(self):
        self.assertEqual(ListMonad([3, 2, 1]), ListMonad([1, 2, 3]).rev())

    def test_list_monad_sum(self):
        self.assertEqual(6, ListMonad([1, 2, 3]).sum(0))
        self.assertEqual(106, ListMonad([1, 2, 3]).sum(100))

    def test_list_monad_dedup_and_distinct(self):
        values = ListMonad([1, 1, 2, 2, 1, 3])

        self.assertEqual(ListMonad([1, 2, 1, 3]), values.dedup())
        self.assertEqual(ListMonad([1, 2, 3]), values.distinct())

    def test_list_monad_sorted_and_sort_by(self):
        values = ListMonad([3, 1, 2])

        self.assertEqual(ListMonad([1, 2, 3]), values.sorted())
        self.assertEqual(ListMonad([3, 2, 1]), values.sorted(reverse=True))
        self.assertEqual(
            ListMonad(["a", "bb", "ccc"]),
            ListMonad(["ccc", "a", "bb"]).sort_by(lambda v: len(v)),
        )

    def test_monad_list_examples(self):
        list = ListMonad(range(2))

        # test list comprehension
        self.assertEqual([0, 1], [x for x in list])

        # test list indexing
        self.assertEqual(0, list[0])

        # test list monad from generator
        def list_generator():
            for x in range(5):
                yield x

        list_monad = ListMonad(list_generator()).map(lambda x: x + 1)
        self.assertEqual([1, 2, 3, 4, 5], list_monad)

        # test list to str
        self.assertEqual("ListMonad([1, 2, 3, 4, 5])", str(list_monad))

        self.assertEqual(
            ["$2.00", "$100.00", "$5.00"],
            (
                ListMonad([1, 99, 4])
                .bind(lambda val: ListMonad([val + 1]))
                .bind(lambda val: ListMonad([f"${val}.00"]))
            ),
        )

        self.assertEqual(
            ["$2.00", "$100.00", "$5.00"],
            (
                ListMonad([1, 99, 4])
                .map(lambda val: val + 1)
                .bind(lambda val: ListMonad([f"${val}.00"]))
            ),
        )
