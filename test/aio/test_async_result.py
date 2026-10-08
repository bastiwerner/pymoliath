import unittest
from unittest.mock import AsyncMock, Mock

from pymoliath.aio import AsyncResult
from pymoliath.result import Err, Ok, Result


class TestAsyncResult(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncResult itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(Ok(10), await AsyncResult.from_ok(10))

    async def test_from_err(self):
        self.assertEqual(Err("boom"), await AsyncResult.from_err("boom"))

    async def test_from_result(self):
        self.assertEqual(Ok(10), await AsyncResult.from_result(Ok(10)))
        self.assertEqual(Err("boom"), await AsyncResult.from_result(Err("boom")))

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(Ok(42), await AsyncResult.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncResult is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncResult.from_ok(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(Ok(3), result)

    async def test_err_short_circuits_map_and_bind(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        result = (
            await AsyncResult.from_err("boom")
            .map(track)
            .bind(lambda v: AsyncResult.from_ok(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Err("boom"), result)

    async def test_ok_short_circuits_map_err_and_bind_err(self):
        calls: list[str] = []

        def track(value: str) -> str:
            calls.append(value)
            return value

        result = (
            await AsyncResult.from_ok(10)
            .map_err(track)
            .bind_err(lambda e: AsyncResult.from_err(track(e)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Ok(10), result)

    async def test_map_sync_function(self):
        result = await AsyncResult.from_ok(10).map(lambda x: x * 2)
        self.assertEqual(Ok(20), result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncResult.from_ok(10).map(double)
        self.assertEqual(Ok(20), result)

    async def test_map_on_err(self):
        result = await AsyncResult.from_err("boom").map(lambda x: x * 2)
        self.assertEqual(Err("boom"), result)

    async def test_map_err_sync_function(self):
        result = await AsyncResult.from_err("boom").map_err(lambda e: e.upper())
        self.assertEqual(Err("BOOM"), result)

    async def test_map_err_async_function(self):
        async def shout(e: str) -> str:
            return e.upper()

        result = await AsyncResult.from_err("boom").map_err(shout)
        self.assertEqual(Err("BOOM"), result)

    async def test_map_err_on_ok(self):
        result = await AsyncResult.from_ok(10).map_err(lambda e: e.upper())
        self.assertEqual(Ok(10), result)

    async def test_bind_returning_async_result(self):
        result = await AsyncResult.from_ok(10).bind(
            lambda x: AsyncResult.from_ok(x + 1)
        )
        self.assertEqual(Ok(11), result)

    async def test_bind_returning_plain_result(self):
        result = await AsyncResult.from_ok(10).bind(lambda x: Ok(x + 1))
        self.assertEqual(Ok(11), result)

        ten: AsyncResult[int, str] = AsyncResult.from_ok(10)
        result_err = await ten.bind(lambda _: Err("boom"))
        self.assertEqual(Err("boom"), result_err)

    async def test_bind_returning_awaitable_result(self):
        async def to_result(x: int) -> Result[int, str]:
            return Ok(x + 1)

        ten: AsyncResult[int, str] = AsyncResult.from_ok(10)
        result = await ten.bind(to_result)
        self.assertEqual(Ok(11), result)

    async def test_bind_on_err(self):
        result = await AsyncResult.from_err("boom").bind(
            lambda x: AsyncResult.from_ok(x + 1)
        )
        self.assertEqual(Err("boom"), result)

    async def test_bind_err_returning_async_result(self):
        result = await AsyncResult.from_err("boom").bind_err(
            lambda e: AsyncResult.from_err(e.upper())
        )
        self.assertEqual(Err("BOOM"), result)

    async def test_bind_err_returning_plain_result(self):
        result = await AsyncResult.from_err("boom").bind_err(lambda e: Err(e.upper()))
        self.assertEqual(Err("BOOM"), result)

        result_ok = await AsyncResult.from_err("boom").bind_err(lambda _: Ok(1))
        self.assertEqual(Ok(1), result_ok)

    async def test_bind_err_returning_awaitable_result(self):
        async def to_result(e: str) -> Result[int, str]:
            return Err(e.upper())

        boom: AsyncResult[int, str] = AsyncResult.from_err("boom")
        result = await boom.bind_err(to_result)
        self.assertEqual(Err("BOOM"), result)

    async def test_bind_err_on_ok(self):
        result = await AsyncResult.from_ok(10).bind_err(
            lambda e: AsyncResult.from_err(e.upper())
        )
        self.assertEqual(Ok(10), result)

    async def test_inspect_called_only_on_ok(self):
        ok_mock = Mock()
        await AsyncResult.from_ok(10).inspect(ok_mock)
        ok_mock.assert_called_once_with(10)

        err_mock = Mock()
        await AsyncResult.from_err("boom").inspect(err_mock)
        err_mock.assert_not_called()

    async def test_inspect_async(self):
        async_mock = AsyncMock()
        result = await AsyncResult.from_ok(10).inspect(async_mock)
        async_mock.assert_called_once_with(10)
        self.assertEqual(Ok(10), result)

    async def test_inspect_err_called_only_on_err(self):
        err_mock = Mock()
        await AsyncResult.from_err("boom").inspect_err(err_mock)
        err_mock.assert_called_once_with("boom")

        ok_mock = Mock()
        await AsyncResult.from_ok(10).inspect_err(ok_mock)
        ok_mock.assert_not_called()

    async def test_inspect_err_async(self):
        async_mock = AsyncMock()
        result = await AsyncResult.from_err("boom").inspect_err(async_mock)
        async_mock.assert_called_once_with("boom")
        self.assertEqual(Err("boom"), result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncResult.from_ok(10).apply(AsyncResult.from_ok(f))
        self.assertEqual(Ok(20), result)

    async def test_apply2(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncResult.from_ok(f).apply2(AsyncResult.from_ok(10))
        self.assertEqual(Ok(20), result)

    async def test_and(self):
        result = await AsyncResult.from_ok(10).and_(AsyncResult.from_ok("a"))
        self.assertEqual(Ok("a"), result)

        result_err = await AsyncResult.from_err("boom").and_(AsyncResult.from_ok("a"))
        self.assertEqual(Err("boom"), result_err)

    async def test_or(self):
        result = await AsyncResult.from_ok(10).or_(AsyncResult.from_ok(20))
        self.assertEqual(Ok(10), result)

        result_fallback = await AsyncResult.from_err("boom").or_(
            AsyncResult.from_ok(20)
        )
        self.assertEqual(Ok(20), result_fallback)

    async def test_zip(self):
        result = await AsyncResult.from_ok(10).zip(AsyncResult.from_ok("a"))
        self.assertEqual(Ok((10, "a")), result)

        ten: AsyncResult[int, str] = AsyncResult.from_ok(10)
        result_err = await ten.zip(AsyncResult.from_err("boom"))
        self.assertEqual(Err("boom"), result_err)

    async def test_flatten(self):
        nested = AsyncResult.from_ok(AsyncResult.from_ok(10))
        self.assertEqual(Ok(10), await nested.flatten())

        nested_err = AsyncResult.from_err("boom")
        self.assertEqual(Err("boom"), await nested_err.flatten())

    async def test_reawaitable(self):
        """Awaiting the same AsyncResult more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncResult.from_ok(10).map(track)

        self.assertEqual(Ok(10), await pipeline)
        self.assertEqual(Ok(10), await pipeline)
        self.assertEqual([10, 10], calls)
