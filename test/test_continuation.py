from __future__ import annotations

import concurrent.futures
import multiprocessing
import unittest
from time import sleep
from typing import Any, Callable

from pymoliath.continuation import Continuation


def delay(args):
    return args


class TestContinuationMonad(unittest.TestCase):

    def test_observable_with_concurrent_thread_pool(self):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        def observable(on_next: Callable[[str], Any]):
            return pool.submit(lambda args: on_next(args), 'world')

        observer_monad: Continuation[str, concurrent.futures.Future] = Continuation(observable)

        actual_result = []
        observer_monad.map(lambda x: f'hi1 {x}').run(actual_result.append)
        observer_monad.map(lambda x: f'hi2 {x}').run(actual_result.append)

        pool.shutdown()

        self.assertEqual(['hi1 world', 'hi2 world'], actual_result)

    def test_observable_with_multiprocessing_pool(self):
        pool = multiprocessing.Pool(processes=2)

        def observable(on_next: Callable[[str], Any]) -> multiprocessing.pool.ApplyResult:
            return pool.apply_async(delay, args=('world',), callback=on_next, error_callback=print)

        actual_result = []
        thread_observable: Continuation[str, multiprocessing.pool.ApplyResult] = Continuation(observable)
        thread_observable.map(lambda x: f'hi1 {x}').run(actual_result.append)
        thread_observable.map(lambda x: f'hi2 {x}').run(actual_result.append)
        thread_observable.map(lambda x: f'hi3 {x}').run(actual_result.append)

        pool.close()
        pool.join()

        self.assertCountEqual(['hi1 world', 'hi2 world', 'hi3 world'], actual_result)
