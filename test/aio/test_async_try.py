from collections.abc import Callable
import unittest
from unittest.mock import AsyncMock, Mock

from pymoliath.aio import AsyncTry
from pymoliath.aio.async_try import map2
from pymoliath.exception import Failure, Success


class TestAsyncTry(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncTry itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(Success(10), await AsyncTry.from_success(10))

    async def test_from_failure(self):
        exc = ValueError("error")
        self.assertEqual(Failure(exc), await AsyncTry.from_failure(exc))

    async def test_from_try(self):
        self.assertEqual(Success(10), await AsyncTry.from_try(Success(10)))
        exc = ValueError("error")
        self.assertEqual(Failure(exc), await AsyncTry.from_try(Failure(exc)))

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(Success(42), await AsyncTry.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncTry is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncTry.from_success(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(Success(3), result)

    async def test_failure_short_circuits_map_and_bind(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        exc = ValueError("error")
        result = (
            await AsyncTry.from_failure(exc)
            .map(track)
            .bind(lambda v: AsyncTry.from_success(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Failure(exc), result)

    async def test_success_short_circuits_map_failure_and_bind_failure(self):
        calls: list[Exception] = []

        def track(exc: Exception) -> Exception:
            calls.append(exc)
            return exc

        result = (
            await AsyncTry.from_success(10)
            .map_failure(track)
            .bind_failure(lambda e: AsyncTry.from_failure(track(e)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Success(10), result)

    async def test_map_sync_function(self):
        result = await AsyncTry.from_success(10).map(lambda x: x * 2)
        self.assertEqual(Success(20), result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncTry.from_success(10).map(double)
        self.assertEqual(Success(20), result)

    async def test_map_on_failure(self):
        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).map(lambda x: x * 2)
        self.assertEqual(Failure(exc), result)

    async def test_map_catches_sync_exception(self):
        def boom(_: int) -> int:
            raise ValueError("boom")

        result = await AsyncTry.from_success(10).map(boom)
        self.assertEqual(Failure(ValueError("boom")), result)

    async def test_map_catches_async_exception(self):
        async def boom(_: int) -> int:
            raise ValueError("boom")

        result = await AsyncTry.from_success(10).map(boom)
        self.assertEqual(Failure(ValueError("boom")), result)

    async def test_map_failure_sync_function(self):
        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).map_failure(
            lambda e: TypeError(str(e))
        )
        self.assertEqual(Failure(TypeError("error")), result)

    async def test_map_failure_async_function(self):
        async def to_type_error(e: Exception) -> Exception:
            return TypeError(str(e))

        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).map_failure(to_type_error)
        self.assertEqual(Failure(TypeError("error")), result)

    async def test_map_failure_on_success(self):
        result = await AsyncTry.from_success(10).map_failure(
            lambda e: TypeError(str(e))
        )
        self.assertEqual(Success(10), result)

    async def test_map_failure_catches_exception(self):
        def boom(_: Exception) -> Exception:
            raise ValueError("boom")

        result = await AsyncTry.from_failure(ValueError("original")).map_failure(boom)
        self.assertEqual(Failure(ValueError("boom")), result)

    async def test_bind_returning_async_try(self):
        result = await AsyncTry.from_success(10).bind(
            lambda x: AsyncTry.from_success(x + 1)
        )
        self.assertEqual(Success(11), result)

    async def test_bind_returning_plain_try(self):
        result = await AsyncTry.from_success(10).bind(lambda x: Success(x + 1))
        self.assertEqual(Success(11), result)

        exc = ValueError("error")
        result_failure = await AsyncTry.from_success(10).bind(lambda _: Failure(exc))
        self.assertEqual(Failure(exc), result_failure)

    async def test_bind_returning_awaitable_try(self):
        async def to_try(x: int) -> Success[int]:
            return Success(x + 1)

        result = await AsyncTry.from_success(10).bind(to_try)
        self.assertEqual(Success(11), result)

    async def test_bind_on_failure(self):
        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).bind(
            lambda x: AsyncTry.from_success(x + 1)
        )
        self.assertEqual(Failure(exc), result)

    async def test_bind_catches_exception(self):
        def boom(_: int) -> AsyncTry[int]:
            raise ValueError("boom")

        result = await AsyncTry.from_success(10).bind(boom)
        self.assertEqual(Failure(ValueError("boom")), result)

    async def test_bind_failure_returning_async_try(self):
        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).bind_failure(
            lambda e: AsyncTry.from_success(str(e))
        )
        self.assertEqual(Success("error"), result)

    async def test_bind_failure_on_success(self):
        result = await AsyncTry.from_success(10).bind_failure(
            lambda e: AsyncTry.from_success(len(str(e)))
        )
        self.assertEqual(Success(10), result)

    async def test_bind_failure_catches_exception(self):
        def boom(_: Exception) -> AsyncTry[int]:
            raise ValueError("boom")

        result = await AsyncTry.from_failure(ValueError("original")).bind_failure(boom)
        self.assertEqual(Failure(ValueError("boom")), result)

    async def test_inspect_called_only_on_success(self):
        success_mock = Mock()
        await AsyncTry.from_success(10).inspect(success_mock)
        success_mock.assert_called_once_with(10)

        failure_mock = Mock()
        await AsyncTry.from_failure(ValueError("error")).inspect(failure_mock)
        failure_mock.assert_not_called()

    async def test_inspect_async(self):
        async_mock = AsyncMock()
        result = await AsyncTry.from_success(10).inspect(async_mock)
        async_mock.assert_called_once_with(10)
        self.assertEqual(Success(10), result)

    async def test_inspect_failure_called_only_on_failure(self):
        exc = ValueError("error")

        success_mock = Mock()
        await AsyncTry.from_success(10).inspect_failure(success_mock)
        success_mock.assert_not_called()

        failure_mock = Mock()
        await AsyncTry.from_failure(exc).inspect_failure(failure_mock)
        failure_mock.assert_called_once_with(exc)

    async def test_inspect_failure_async(self):
        async_mock = AsyncMock()
        exc = ValueError("error")
        result = await AsyncTry.from_failure(exc).inspect_failure(async_mock)
        async_mock.assert_called_once_with(exc)
        self.assertEqual(Failure(exc), result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncTry.from_success(10).apply(AsyncTry.from_success(f))
        self.assertEqual(Success(20), result)

    async def test_map2(self):
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(
            Success(3),
            await map2(AsyncTry.from_success(1), AsyncTry.from_success(2), add),
        )
        first: AsyncTry[int] = AsyncTry.from_failure(ValueError("first"))
        second: AsyncTry[int] = AsyncTry.from_failure(ValueError("second"))
        self.assertEqual(Failure(ValueError("first")), await map2(first, second, add))
        self.assertEqual(
            Failure(ValueError("second")),
            await map2(AsyncTry.from_success(1), second, add),
        )

    async def test_apply_function_side_wins(self):
        def f(x: int) -> int:
            return x * 2

        self.assertEqual(
            Success(20), await AsyncTry.from_success(10).apply(AsyncTry.from_success(f))
        )

    async def test_and(self):
        result = await AsyncTry.from_success(10).and_(AsyncTry.from_success("a"))
        self.assertEqual(Success("a"), result)

        exc = ValueError("error")
        result_failure = await AsyncTry.from_failure(exc).and_(
            AsyncTry.from_success("a")
        )
        self.assertEqual(Failure(exc), result_failure)

    async def test_or(self):
        result = await AsyncTry.from_success(10).or_(AsyncTry.from_success(20))
        self.assertEqual(Success(10), result)

        result_fallback = await AsyncTry.from_failure(ValueError("error")).or_(
            AsyncTry.from_success(20)
        )
        self.assertEqual(Success(20), result_fallback)

    async def test_zip(self):
        result = await AsyncTry.from_success(10).zip(AsyncTry.from_success("a"))
        self.assertEqual(Success((10, "a")), result)

        exc = ValueError("error")
        result_failure = await AsyncTry.from_success(10).zip(AsyncTry.from_failure(exc))
        self.assertEqual(Failure(exc), result_failure)

    async def test_flatten(self):
        nested = AsyncTry.from_success(AsyncTry.from_success(10))
        self.assertEqual(Success(10), await nested.flatten())

        exc = ValueError("error")
        nested_failure = AsyncTry.from_failure(exc)
        self.assertEqual(Failure(exc), await nested_failure.flatten())

    async def test_reawaitable(self):
        """Awaiting the same AsyncTry more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncTry.from_success(10).map(track)

        self.assertEqual(Success(10), await pipeline)
        self.assertEqual(Success(10), await pipeline)
        self.assertEqual([10, 10], calls)

    async def test_apply2(self):
        def increment(x: int) -> int:
            return x + 1

        function: AsyncTry[Callable[[int], int]] = AsyncTry.from_success(increment)
        failed: AsyncTry[Callable[[int], int]] = AsyncTry.from_failure(
            ValueError("function")
        )
        value: AsyncTry[int] = AsyncTry.from_success(1)
        error: AsyncTry[int] = AsyncTry.from_failure(ValueError("value"))

        self.assertEqual(Success(2), await function.apply2(value))
        self.assertEqual(Failure(ValueError("value")), await function.apply2(error))
        self.assertEqual(Failure(ValueError("function")), await failed.apply2(error))
