from collections.abc import Callable
import unittest
from unittest.mock import AsyncMock, Mock

from pymoliath.aio import AsyncEither
from pymoliath.either import Either, Left, Right


class TestAsyncEither(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncEither itself is awaitable - no .run()/execute call needed."""
        self.assertEqual(Right(10), await AsyncEither.from_right(10))

    async def test_from_left(self):
        self.assertEqual(Left("error"), await AsyncEither.from_left("error"))

    async def test_from_either(self):
        self.assertEqual(Right(10), await AsyncEither.from_either(Right(10)))
        self.assertEqual(Left("e"), await AsyncEither.from_either(Left("e")))

    async def test_from_coroutine(self):
        async def fetch() -> int:
            return 42

        self.assertEqual(Right(42), await AsyncEither.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncEither is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncEither.from_right(1).map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual(Right(3), result)

    async def test_left_short_circuits_right_channel(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        result = (
            await AsyncEither.from_left("error")
            .map(track)
            .bind(lambda v: AsyncEither.from_right(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Left("error"), result)

    async def test_right_short_circuits_left_channel(self):
        calls: list[str] = []

        def track(value: str) -> str:
            calls.append(value)
            return value

        result = (
            await AsyncEither.from_right(10)
            .map_left(track)
            .bind_left(lambda v: AsyncEither.from_left(track(v)))
        )

        self.assertEqual([], calls)
        self.assertEqual(Right(10), result)

    async def test_map_sync_function(self):
        result = await AsyncEither.from_right(10).map(lambda x: x * 2)
        self.assertEqual(Right(20), result)

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        result = await AsyncEither.from_right(10).map(double)
        self.assertEqual(Right(20), result)

    async def test_map_on_left(self):
        result = await AsyncEither.from_left("error").map(lambda x: x * 2)
        self.assertEqual(Left("error"), result)

    async def test_map_left_sync_function(self):
        result = await AsyncEither.from_left("error").map_left(str.upper)
        self.assertEqual(Left("ERROR"), result)

    async def test_map_left_async_function(self):
        async def shout(x: str) -> str:
            return x.upper()

        result = await AsyncEither.from_left("error").map_left(shout)
        self.assertEqual(Left("ERROR"), result)

    async def test_map_left_on_right(self):
        result = await AsyncEither.from_right(10).map_left(str.upper)
        self.assertEqual(Right(10), result)

    async def test_bind_returning_async_either(self):
        result = await AsyncEither.from_right(10).bind(
            lambda x: AsyncEither.from_right(x + 1)
        )
        self.assertEqual(Right(11), result)

    async def test_bind_returning_plain_either(self):
        result = await AsyncEither.from_right(10).bind(lambda x: Right(x + 1))
        self.assertEqual(Right(11), result)

        ten: AsyncEither[str, int] = AsyncEither.from_right(10)
        result_left = await ten.bind(lambda _: Left("bad"))
        self.assertEqual(Left("bad"), result_left)

    async def test_bind_returning_awaitable_either(self):
        async def to_either(x: int) -> Either[str, int]:
            return Right(x + 1)

        ten: AsyncEither[str, int] = AsyncEither.from_right(10)
        result = await ten.bind(to_either)
        self.assertEqual(Right(11), result)

    async def test_bind_on_left(self):
        result = await AsyncEither.from_left("error").bind(
            lambda x: AsyncEither.from_right(x + 1)
        )
        self.assertEqual(Left("error"), result)

    async def test_bind_left_returning_async_either(self):
        result = await AsyncEither.from_left("error").bind_left(
            lambda x: AsyncEither.from_left(x.upper())
        )
        self.assertEqual(Left("ERROR"), result)

    async def test_bind_left_returning_plain_either(self):
        result = await AsyncEither.from_left("error").bind_left(
            lambda x: Left(x.upper())
        )
        self.assertEqual(Left("ERROR"), result)

        result_right = await AsyncEither.from_left("error").bind_left(
            lambda _: Right(0)
        )
        self.assertEqual(Right(0), result_right)

    async def test_bind_left_returning_awaitable_either(self):
        async def to_either(x: str) -> Either[str, int]:
            return Left(x.upper())

        error: AsyncEither[str, int] = AsyncEither.from_left("error")
        result = await error.bind_left(to_either)
        self.assertEqual(Left("ERROR"), result)

    async def test_bind_left_on_right(self):
        result = await AsyncEither.from_right(10).bind_left(
            lambda x: AsyncEither.from_left(x.upper())
        )
        self.assertEqual(Right(10), result)

    async def test_inspect_called_only_on_right(self):
        right_mock = Mock()
        await AsyncEither.from_right(10).inspect(right_mock)
        right_mock.assert_called_once_with(10)

        left_mock = Mock()
        await AsyncEither.from_left("error").inspect(left_mock)
        left_mock.assert_not_called()

    async def test_inspect_async(self):
        async_mock = AsyncMock()
        result = await AsyncEither.from_right(10).inspect(async_mock)
        async_mock.assert_called_once_with(10)
        self.assertEqual(Right(10), result)

    async def test_inspect_left_called_only_on_left(self):
        left_mock = Mock()
        await AsyncEither.from_left("error").inspect_left(left_mock)
        left_mock.assert_called_once_with("error")

        right_mock = Mock()
        await AsyncEither.from_right(10).inspect_left(right_mock)
        right_mock.assert_not_called()

    async def test_inspect_left_async(self):
        async_mock = AsyncMock()
        result = await AsyncEither.from_left("error").inspect_left(async_mock)
        async_mock.assert_called_once_with("error")
        self.assertEqual(Left("error"), result)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        result = await AsyncEither.from_right(10).apply(AsyncEither.from_right(f))
        self.assertEqual(Right(20), result)

    async def test_apply_function_side_wins(self):
        def f(x: int) -> int:
            return x * 2

        self.assertEqual(
            Right(20), await AsyncEither.from_right(10).apply(AsyncEither.from_right(f))
        )

    async def test_and(self):
        result = await AsyncEither.from_right(10).and_(AsyncEither.from_right("a"))
        self.assertEqual(Right("a"), result)

        result_left = await AsyncEither.from_left("error").and_(
            AsyncEither.from_right("a")
        )
        self.assertEqual(Left("error"), result_left)

    async def test_or(self):
        result = await AsyncEither.from_right(10).or_(AsyncEither.from_right(20))
        self.assertEqual(Right(10), result)

        result_fallback = await AsyncEither.from_left("error").or_(
            AsyncEither.from_right(20)
        )
        self.assertEqual(Right(20), result_fallback)

    async def test_zip(self):
        result = await AsyncEither.from_right(10).zip(AsyncEither.from_right("a"))
        self.assertEqual(Right((10, "a")), result)

        ten: AsyncEither[str, int] = AsyncEither.from_right(10)
        result_left = await ten.zip(AsyncEither.from_left("error"))
        self.assertEqual(Left("error"), result_left)

        result_left_self = await AsyncEither.from_left("error").zip(
            AsyncEither.from_right("a")
        )
        self.assertEqual(Left("error"), result_left_self)

    async def test_flatten(self):
        nested = AsyncEither.from_right(AsyncEither.from_right(10))
        self.assertEqual(Right(10), await nested.flatten())

        nested_left = AsyncEither.from_left("error")
        self.assertEqual(Left("error"), await nested_left.flatten())

    async def test_reawaitable(self):
        """Awaiting the same AsyncEither more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncEither.from_right(10).map(track)

        self.assertEqual(Right(10), await pipeline)
        self.assertEqual(Right(10), await pipeline)
        self.assertEqual([10, 10], calls)

    async def test_apply2(self):
        def increment(x: int) -> int:
            return x + 1

        function: AsyncEither[str, Callable[[int], int]] = AsyncEither.from_right(
            increment
        )
        failed: AsyncEither[str, Callable[[int], int]] = AsyncEither.from_left(
            "function"
        )
        value: AsyncEither[str, int] = AsyncEither.from_right(1)
        error: AsyncEither[str, int] = AsyncEither.from_left("value")

        self.assertEqual(Right(2), await function.apply2(value))
        self.assertEqual(Left("value"), await function.apply2(error))
        self.assertEqual(Left("function"), await failed.apply2(error))

    async def test_from_left_and_run(self):
        self.assertEqual(Left("e"), await AsyncEither.from_left("e"))
        self.assertEqual(Right(1), await AsyncEither.from_right(1).run())

    async def test_and_then_is_bind(self):
        value: AsyncEither[str, int] = AsyncEither.from_right(1)
        self.assertEqual(Right(2), await value.and_then(lambda x: Right(x + 1)))

    async def test_or_else_is_bind_left(self):
        failed: AsyncEither[str, int] = AsyncEither.from_left("abc")
        self.assertEqual(Right(3), await failed.or_else(lambda e: Right(len(e))))

    async def test_filter(self):
        async def positive(x: int) -> bool:
            return x > 0

        value: AsyncEither[str, int] = AsyncEither.from_right(-1)
        self.assertEqual(
            Left("negative"), await value.filter(lambda x: x > 0, "negative")
        )
        one: AsyncEither[str, int] = AsyncEither.from_right(1)
        self.assertEqual(Right(1), await one.filter(positive, "negative"))

        predicate = Mock(return_value=True)
        failed: AsyncEither[str, int] = AsyncEither.from_left("e")
        self.assertEqual(Left("e"), await failed.filter(predicate, "negative"))
        predicate.assert_not_called()
