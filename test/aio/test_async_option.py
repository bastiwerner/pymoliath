from collections.abc import Callable
import unittest
from unittest.mock import AsyncMock, Mock

from pymoliath.aio import AsyncOption
from pymoliath.aio.async_option import map2
from pymoliath.option import Nil, Some


class TestAsyncOption(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncOption itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(Some(10), await AsyncOption.from_value(10))

    async def test_from_option(self):
        self.assertEqual(Some(10), await AsyncOption.from_option(Some(10)))
        self.assertEqual(Nil(), await AsyncOption.from_option(Nil()))

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(Some(42), await AsyncOption.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncOption is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncOption.from_value(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(Some(3), result)

    async def test_nil_short_circuits_pipeline(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        result = (
            await AsyncOption.from_option(Nil())
            .map(track)
            .bind(lambda v: AsyncOption.from_value(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Nil(), result)

    async def test_map_sync_function(self):
        result = await AsyncOption.from_value(10).map(lambda x: x * 2)
        self.assertEqual(Some(20), result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncOption.from_value(10).map(double)
        self.assertEqual(Some(20), result)

    async def test_map_on_nil(self):
        result = await AsyncOption.from_option(Nil()).map(lambda x: x * 2)
        self.assertEqual(Nil(), result)

    async def test_bind_returning_async_option(self):
        result = await AsyncOption.from_value(10).bind(
            lambda x: AsyncOption.from_value(x + 1)
        )
        self.assertEqual(Some(11), result)

    async def test_bind_returning_plain_option(self):
        result = await AsyncOption.from_value(10).bind(lambda x: Some(x + 1))
        self.assertEqual(Some(11), result)

        result_nil = await AsyncOption.from_value(10).bind(lambda _: Nil())
        self.assertEqual(Nil(), result_nil)

    async def test_bind_returning_awaitable_option(self):
        async def to_option(x: int) -> Some[int]:
            return Some(x + 1)

        result = await AsyncOption.from_value(10).bind(to_option)
        self.assertEqual(Some(11), result)

    async def test_bind_on_nil(self):
        result = await AsyncOption.from_option(Nil()).bind(
            lambda x: AsyncOption.from_value(x + 1)
        )
        self.assertEqual(Nil(), result)

    async def test_filter_sync(self):
        self.assertEqual(
            Some(10), await AsyncOption.from_value(10).filter(lambda x: x > 5)
        )
        self.assertEqual(
            Nil(), await AsyncOption.from_value(10).filter(lambda x: x > 50)
        )

    async def test_filter_async(self):
        async def is_positive(x: int) -> bool:
            return x > 0

        self.assertEqual(Some(10), await AsyncOption.from_value(10).filter(is_positive))
        self.assertEqual(Nil(), await AsyncOption.from_value(-10).filter(is_positive))

    async def test_filter_on_nil(self):
        result = await AsyncOption.from_option(Nil()).filter(lambda x: True)
        self.assertEqual(Nil(), result)

    async def test_inspect_called_only_on_some(self):
        some_mock = Mock()
        await AsyncOption.from_value(10).inspect(some_mock)
        some_mock.assert_called_once_with(10)

        nil_mock = Mock()
        await AsyncOption.from_option(Nil()).inspect(nil_mock)
        nil_mock.assert_not_called()

    async def test_inspect_async(self):
        async_mock = AsyncMock()
        result = await AsyncOption.from_value(10).inspect(async_mock)
        async_mock.assert_called_once_with(10)
        self.assertEqual(Some(10), result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncOption.from_value(10).apply(AsyncOption.from_value(f))
        self.assertEqual(Some(20), result)

    async def test_map2(self):
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(
            Some(3),
            await map2(AsyncOption.from_value(1), AsyncOption.from_value(2), add),
        )
        empty: AsyncOption[int] = AsyncOption.from_option(Nil())
        self.assertEqual(Nil(), await map2(empty, AsyncOption.from_value(2), add))
        self.assertEqual(Nil(), await map2(AsyncOption.from_value(1), empty, add))

    async def test_apply_function_side_wins(self):
        def f(x: int) -> int:
            return x * 2

        self.assertEqual(
            Some(20), await AsyncOption.from_value(10).apply(AsyncOption.from_value(f))
        )

    async def test_and(self):
        result = await AsyncOption.from_value(10).and_(AsyncOption.from_value("a"))
        self.assertEqual(Some("a"), result)

        result_nil = await AsyncOption.from_option(Nil()).and_(
            AsyncOption.from_value("a")
        )
        self.assertEqual(Nil(), result_nil)

    async def test_or(self):
        result = await AsyncOption.from_value(10).or_(AsyncOption.from_value(20))
        self.assertEqual(Some(10), result)

        result_fallback = await AsyncOption.from_option(Nil()).or_(
            AsyncOption.from_value(20)
        )
        self.assertEqual(Some(20), result_fallback)

    async def test_zip(self):
        result = await AsyncOption.from_value(10).zip(AsyncOption.from_value("a"))
        self.assertEqual(Some((10, "a")), result)

        result_nil = await AsyncOption.from_value(10).zip(
            AsyncOption.from_option(Nil())
        )
        self.assertEqual(Nil(), result_nil)

    async def test_flatten(self):
        nested = AsyncOption.from_value(AsyncOption.from_value(10))
        self.assertEqual(Some(10), await nested.flatten())

        nested_nil = AsyncOption.from_option(Nil())
        self.assertEqual(Nil(), await nested_nil.flatten())

    async def test_reawaitable(self):
        """Awaiting the same AsyncOption more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncOption.from_value(10).map(track)

        self.assertEqual(Some(10), await pipeline)
        self.assertEqual(Some(10), await pipeline)
        self.assertEqual([10, 10], calls)

    async def test_apply2(self):
        def increment(x: int) -> int:
            return x + 1

        function: AsyncOption[Callable[[int], int]] = AsyncOption.from_value(increment)
        failed: AsyncOption[Callable[[int], int]] = AsyncOption.from_option(Nil())
        value: AsyncOption[int] = AsyncOption.from_value(1)
        error: AsyncOption[int] = AsyncOption.from_option(Nil())

        self.assertEqual(Some(2), await function.apply2(value))
        self.assertEqual(Nil(), await function.apply2(error))
        self.assertEqual(Nil(), await failed.apply2(error))
