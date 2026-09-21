import unittest
from unittest.mock import AsyncMock, Mock

from pymoliath import AsyncMaybe
from pymoliath.maybe import Just, Nothing


class TestAsyncMaybe(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncMaybe itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(Just(10), await AsyncMaybe.from_value(10))

    async def test_from_maybe(self):
        self.assertEqual(Just(10), await AsyncMaybe.from_maybe(Just(10)))
        self.assertEqual(Nothing(), await AsyncMaybe.from_maybe(Nothing()))

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(Just(42), await AsyncMaybe.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncMaybe is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncMaybe.from_value(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(Just(3), result)

    async def test_nothing_short_circuits_pipeline(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        result = (
            await AsyncMaybe.from_maybe(Nothing())
            .map(track)
            .bind(lambda v: AsyncMaybe.from_value(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Nothing(), result)

    async def test_map_sync_function(self):
        result = await AsyncMaybe.from_value(10).map(lambda x: x * 2)
        self.assertEqual(Just(20), result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncMaybe.from_value(10).map(double)
        self.assertEqual(Just(20), result)

    async def test_map_on_nothing(self):
        result = await AsyncMaybe.from_maybe(Nothing()).map(lambda x: x * 2)
        self.assertEqual(Nothing(), result)

    async def test_bind_returning_async_maybe(self):
        result = await AsyncMaybe.from_value(10).bind(
            lambda x: AsyncMaybe.from_value(x + 1)
        )
        self.assertEqual(Just(11), result)

    async def test_bind_returning_plain_maybe(self):
        result = await AsyncMaybe.from_value(10).bind(lambda x: Just(x + 1))
        self.assertEqual(Just(11), result)

        result_nothing = await AsyncMaybe.from_value(10).bind(lambda _: Nothing())
        self.assertEqual(Nothing(), result_nothing)

    async def test_bind_returning_awaitable_maybe(self):
        async def to_maybe(x: int) -> Just[int]:
            return Just(x + 1)

        result = await AsyncMaybe.from_value(10).bind(to_maybe)
        self.assertEqual(Just(11), result)

    async def test_bind_on_nothing(self):
        result = await AsyncMaybe.from_maybe(Nothing()).bind(
            lambda x: AsyncMaybe.from_value(x + 1)
        )
        self.assertEqual(Nothing(), result)

    async def test_filter_sync(self):
        self.assertEqual(
            Just(10), await AsyncMaybe.from_value(10).filter(lambda x: x > 5)
        )
        self.assertEqual(
            Nothing(), await AsyncMaybe.from_value(10).filter(lambda x: x > 50)
        )

    async def test_filter_async(self):
        async def is_positive(x: int) -> bool:
            return x > 0

        self.assertEqual(Just(10), await AsyncMaybe.from_value(10).filter(is_positive))
        self.assertEqual(
            Nothing(), await AsyncMaybe.from_value(-10).filter(is_positive)
        )

    async def test_filter_on_nothing(self):
        result = await AsyncMaybe.from_maybe(Nothing()).filter(lambda x: True)
        self.assertEqual(Nothing(), result)

    async def test_inspect_called_only_on_just(self):
        just_mock = Mock()
        await AsyncMaybe.from_value(10).inspect(just_mock)
        just_mock.assert_called_once_with(10)

        nothing_mock = Mock()
        await AsyncMaybe.from_maybe(Nothing()).inspect(nothing_mock)
        nothing_mock.assert_not_called()

    async def test_inspect_async(self):
        async_mock = AsyncMock()
        result = await AsyncMaybe.from_value(10).inspect(async_mock)
        async_mock.assert_called_once_with(10)
        self.assertEqual(Just(10), result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncMaybe.from_value(10).apply(AsyncMaybe.from_value(f))
        self.assertEqual(Just(20), result)

    async def test_apply2(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncMaybe.from_value(f).apply2(AsyncMaybe.from_value(10))
        self.assertEqual(Just(20), result)

    async def test_and(self):
        result = await AsyncMaybe.from_value(10).and_(AsyncMaybe.from_value("a"))
        self.assertEqual(Just("a"), result)

        result_nothing = await AsyncMaybe.from_maybe(Nothing()).and_(
            AsyncMaybe.from_value("a")
        )
        self.assertEqual(Nothing(), result_nothing)

    async def test_or(self):
        result = await AsyncMaybe.from_value(10).or_(AsyncMaybe.from_value(20))
        self.assertEqual(Just(10), result)

        result_fallback = await AsyncMaybe.from_maybe(Nothing()).or_(
            AsyncMaybe.from_value(20)
        )
        self.assertEqual(Just(20), result_fallback)

    async def test_zip(self):
        result = await AsyncMaybe.from_value(10).zip(AsyncMaybe.from_value("a"))
        self.assertEqual(Just((10, "a")), result)

        result_nothing = await AsyncMaybe.from_value(10).zip(
            AsyncMaybe.from_maybe(Nothing())
        )
        self.assertEqual(Nothing(), result_nothing)

    async def test_flatten(self):
        nested = AsyncMaybe.from_value(AsyncMaybe.from_value(10))
        self.assertEqual(Just(10), await nested.flatten())

        nested_nothing = AsyncMaybe.from_maybe(Nothing())
        self.assertEqual(Nothing(), await nested_nothing.flatten())

    async def test_reawaitable(self):
        """Awaiting the same AsyncMaybe more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncMaybe.from_value(10).map(track)

        self.assertEqual(Just(10), await pipeline)
        self.assertEqual(Just(10), await pipeline)
        self.assertEqual([10, 10], calls)
