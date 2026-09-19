import unittest

from pymoliath.async_io import AsyncIO
from pymoliath.io import IO


class TestAsyncIO(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncIO itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(10, await AsyncIO.from_value(10))

    async def test_from_io(self):
        result = await AsyncIO.from_io(IO(lambda: 42))
        self.assertEqual(42, result)

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(42, await AsyncIO.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a map/bind chain must not call any callback until the AsyncIO is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncIO.from_value(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(3, result)

    async def test_bind_is_lazy(self):
        """Sync IO.bind() eagerly runs `self` when .bind() itself is called - AsyncIO must not."""
        started: list[str] = []

        def make_source() -> AsyncIO[int]:
            async def run() -> int:
                started.append("ran")
                return 1

            return AsyncIO(run)

        pipeline = make_source().bind(lambda x: AsyncIO.from_value(x + 1))
        self.assertEqual([], started)

        result = await pipeline
        self.assertEqual(["ran"], started)
        self.assertEqual(2, result)

    async def test_map_sync_function(self):
        result = await AsyncIO.from_value(10).map(lambda x: x * 2)
        self.assertEqual(20, result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncIO.from_value(10).map(double)
        self.assertEqual(20, result)

    async def test_bind_returning_async_io(self):
        result = await AsyncIO.from_value(10).bind(lambda x: AsyncIO.from_value(x + 1))
        self.assertEqual(11, result)

    async def test_bind_returning_io(self):
        result = await AsyncIO.from_value(10).bind(lambda x: IO(lambda: x + 1))
        self.assertEqual(11, result)

    async def test_bind_returning_awaitable(self):
        async def to_value(x: int) -> int:
            return x + 1

        result = await AsyncIO.from_value(10).bind(to_value)
        self.assertEqual(11, result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncIO.from_value(10).apply(AsyncIO.from_value(f))
        self.assertEqual(20, result)

    async def test_apply2(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncIO.from_value(f).apply2(AsyncIO.from_value(10))
        self.assertEqual(20, result)

    async def test_reawaitable(self):
        """Awaiting the same AsyncIO more than once re-runs the pipeline each time (not memoized)."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncIO.from_value(10).map(track)

        self.assertEqual(10, await pipeline)
        self.assertEqual(10, await pipeline)
        self.assertEqual([10, 10], calls)


if __name__ == "__main__":
    unittest.main()
