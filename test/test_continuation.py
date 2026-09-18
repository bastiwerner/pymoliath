from __future__ import annotations

import concurrent.futures
import multiprocessing
import unittest
from typing import Callable

from pymoliath.continuation import Continuation
from pymoliath.util import compose


def delay(args: str) -> str:
    return args


class TestContinuationMonad(unittest.TestCase):
    def test_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        continuation_value: Continuation[int, int] = Continuation(
            lambda callback: callback(42)
        )
        applicative = Continuation(lambda callback: callback(lambda x: x))

        self.assertEqual(
            continuation_value.apply(applicative).run(lambda x: x),  # pyright: ignore[reportArgumentType]
            continuation_value.run(lambda x: x),
        )
        self.assertEqual(
            applicative.apply2(continuation_value).run(lambda x: x),  # pyright: ignore[reportAttributeAccessIssue]
            continuation_value.run(lambda x: x),
        )

    def test_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42

        self.assertEqual(
            Continuation(lambda callback: callback(x))
            .apply(Continuation(lambda callback: callback(f)))
            .run(lambda x: x),
            Continuation(lambda callback: callback(f(x))).run(lambda x: x),
        )
        self.assertEqual(
            Continuation(lambda callback: callback(f))
            .apply2(Continuation(lambda callback: callback(x)))  # pyright: ignore[reportAttributeAccessIssue]
            .run(lambda x: x),
            Continuation(lambda callback: callback(f(x))).run(lambda x: x),
        )

    def test_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = Continuation(lambda callback: callback(42))
        u = Continuation(lambda callback: callback(lambda x: x + 42))
        v = Continuation(lambda callback: callback(lambda x: x * 42))
        composition = lambda f, g: compose(f, g)
        applicative = Continuation(lambda callback: callback(composition))

        self.assertEqual(
            w.apply(v.apply(u.apply(applicative))).run(lambda x: x),  # pyright: ignore[reportArgumentType]
            w.apply(v).apply(u).run(lambda x: x),  # pyright: ignore[reportArgumentType]
        )
        self.assertEqual(
            applicative.apply2(u).apply2(v).apply2(w).run(lambda x: x),  # pyright: ignore[reportAttributeAccessIssue]
            u.apply2(v.apply2(w)).run(lambda x: x),  # pyright: ignore[reportAttributeAccessIssue]
        )

    def test_observable_with_concurrent_thread_pool(self):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        def observable(on_next: Callable[[str], None]) -> None:
            future = pool.submit(delay, "world")
            return on_next(future.result())

        observer_monad: Continuation[str, None] = Continuation(observable)

        actual_result = []
        observer_monad.map(lambda x: f"hi1 {x}").run(actual_result.append)
        observer_monad.map(lambda x: f"hi2 {x}").run(actual_result.append)

        pool.shutdown()

        self.assertEqual(["hi1 world", "hi2 world"], actual_result)

    def test_observable_with_multiprocessing_pool(self):
        pool = multiprocessing.Pool(processes=2)

        def observable(on_next: Callable[[str], None]) -> None:
            result = pool.apply_async(
                delay, args=("world",), error_callback=print
            ).get()
            return on_next(result)

        actual_result = []
        thread_observable: Continuation[str, None] = Continuation(observable)
        thread_observable.map(lambda x: f"hi1 {x}").run(actual_result.append)
        thread_observable.map(lambda x: f"hi2 {x}").run(actual_result.append)
        thread_observable.map(lambda x: f"hi3 {x}").run(actual_result.append)

        pool.close()
        pool.join()

        self.assertCountEqual(["hi1 world", "hi2 world", "hi3 world"], actual_result)
