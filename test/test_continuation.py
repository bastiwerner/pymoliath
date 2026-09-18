from __future__ import annotations

import concurrent.futures
import multiprocessing
import unittest
from typing import Callable

from pymoliath.continuation import Continuation


def delay(args: str) -> str:
    return args


class TestContinuationMonad(unittest.TestCase):
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
