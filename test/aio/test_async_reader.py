import unittest

from pymoliath.aio import AsyncReader
from pymoliath.reader import Reader


class TestAsyncReader(unittest.IsolatedAsyncioTestCase):
    async def test_run_resolves_with_env(self):
        """AsyncReader is not bare-awaitable - .run(env) is the terminal operation."""
        reader = AsyncReader.from_coroutine(async_add_env)
        self.assertEqual(11, await reader.run(1))
        self.assertEqual(101, await reader.run(91))

    async def test_from_value_ignores_env(self):
        reader = AsyncReader.from_value(42)
        self.assertEqual(42, await reader.run("anything"))
        self.assertEqual(42, await reader.run("something else"))

    async def test_from_reader(self):
        sync_reader = Reader(lambda env: f"Hello {env}")
        async_reader = AsyncReader.from_reader(sync_reader)
        self.assertEqual("Hello world", await async_reader.run("world"))

    async def test_pipeline_is_lazy_until_run_is_awaited(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncReader.from_coroutine(async_identity_env).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline.run(1)
        self.assertEqual([1, 2], calls)
        self.assertEqual(3, result)

    async def test_map_sync_function(self):
        reader = AsyncReader.from_coroutine(async_identity_env).map(lambda x: x * 2)
        self.assertEqual(20, await reader.run(10))

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        reader = AsyncReader.from_coroutine(async_identity_env).map(double)
        self.assertEqual(20, await reader.run(10))

    async def test_bind_returning_async_reader(self):
        reader = AsyncReader.from_value(10).bind(
            lambda x: AsyncReader.from_coroutine(async_add_env).map(lambda e: x + e)
        )
        self.assertEqual(25, await reader.run(5))

    async def test_bind_returning_plain_reader(self):
        reader = AsyncReader.from_value(10).bind(lambda x: Reader(lambda env: x + env))
        self.assertEqual(15, await reader.run(5))

    async def test_bind_returning_awaitable(self):
        async def to_value(x: int) -> int:
            return x + 1

        reader = AsyncReader.from_value(10).bind(to_value)
        self.assertEqual(11, await reader.run("unused"))

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        reader = AsyncReader.from_value(10).apply(AsyncReader.from_value(f))
        self.assertEqual(20, await reader.run("unused"))

    async def test_apply2(self):
        def f(x: int) -> int:
            return x * 2

        reader = AsyncReader.from_value(f).apply2(AsyncReader.from_value(10))
        self.assertEqual(20, await reader.run("unused"))

    async def test_ask(self):
        reader = AsyncReader.ask()
        self.assertEqual("the-env", await reader.run("the-env"))

    async def test_local(self):
        reader = AsyncReader.from_coroutine(async_identity_env).local(
            lambda env: env + 1
        )
        self.assertEqual(6, await reader.run(5))

    async def test_reader_specific_run_with_different_envs(self):
        """The same AsyncReader can be .run() with different environments, each independently."""
        reader = AsyncReader.from_coroutine(async_add_env).map(lambda x: x * 2)
        self.assertEqual(22, await reader.run(1))
        self.assertEqual(202, await reader.run(91))


async def async_add_env(env: int) -> int:
    return env + 10


async def async_identity_env(env: int) -> int:
    return env
