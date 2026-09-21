import unittest

from pymoliath import AsyncWriter
from pymoliath.writer import Writer


class TestAsyncWriter(unittest.IsolatedAsyncioTestCase):
    async def test_directly_awaitable(self):
        """The AsyncWriter itself is awaitable - no .run()/execute call needed."""
        self.assertEqual((10, "hi"), await AsyncWriter.from_value(10, "hi"))

    async def test_from_writer(self):
        self.assertEqual((10, "hi"), await AsyncWriter.from_writer(Writer(10, "hi")))

    async def test_from_coroutine(self):
        async def fetch() -> tuple[int, str]:
            return 42, "fetched"

        self.assertEqual((42, "fetched"), await AsyncWriter.from_coroutine(fetch))

    async def test_pipeline_is_lazy_until_awaited(self):
        """Building a chain must not call any callback until the AsyncWriter is awaited."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncWriter.from_value(1, "a").map(track).map(track)
        self.assertEqual([], calls)

        result = await pipeline
        self.assertEqual([1, 2], calls)
        self.assertEqual((3, "a"), result)

    async def test_map_sync_function(self):
        result = await AsyncWriter.from_value(1, "map Example").map(lambda v: v + 1)
        self.assertEqual((2, "map Example"), result)

    async def test_map_async_function(self):
        async def increment(v: int) -> int:
            return v + 1

        result = await AsyncWriter.from_value(1, "map Example").map(increment)
        self.assertEqual((2, "map Example"), result)

    async def test_bind_returning_async_writer(self):
        bind_function = lambda v: AsyncWriter.from_value(v * v, "bind function")

        result = await AsyncWriter.from_value(5, "bind value ").bind(bind_function)
        self.assertEqual((25, "bind value bind function"), result)

    async def test_bind_returning_plain_writer(self):
        bind_function = lambda v: Writer(v * v, "bind function")

        result = await AsyncWriter.from_value(5, "bind value ").bind(bind_function)
        self.assertEqual((25, "bind value bind function"), result)

    async def test_bind_returning_awaitable_tuple(self):
        async def bind_function(v: int) -> tuple[int, str]:
            return v * v, "bind function"

        result = await AsyncWriter.from_value(5, "bind value ").bind(bind_function)
        self.assertEqual((25, "bind value bind function"), result)

    async def test_apply(self):
        applicative = AsyncWriter.from_value(lambda v: v % 7, "apply function ")
        value = AsyncWriter.from_value(10, "apply value")

        result = await value.apply(applicative)
        self.assertEqual((3, "apply valueapply function "), result)

    async def test_apply2(self):
        applicative = AsyncWriter.from_value(lambda v: v % 7, "apply function ")
        value = AsyncWriter.from_value(10, "apply value")

        result = await applicative.apply2(value)
        self.assertEqual((3, "apply function apply value"), result)

    async def test_tell(self):
        result = await AsyncWriter.from_value(1, "tell example").tell(" monoid append")
        self.assertEqual((1, "tell example monoid append"), result)

    async def test_listen(self):
        result = await AsyncWriter.from_value(10, "listen log").listen()
        self.assertEqual(((10, "listen log"), "listen log"), result)

    async def test_pass(self):
        result = await AsyncWriter.from_value(
            (10, lambda w: w + " passing"), "monoid"
        ).pass_()
        self.assertEqual((10, "monoid passing"), result)

    async def test_full_pipeline(self):
        result = await (
            AsyncWriter.from_value(5.0, "")
            .tell("initial value ")
            .bind(lambda x: AsyncWriter.from_value(x + 1, "add 1 "))
            .apply(AsyncWriter.from_value(lambda i: i * 2, "multiply by 2 "))
            .map(lambda x: x / 2)
            .tell("divide by 2")
        )
        self.assertEqual((6.0, "initial value add 1 multiply by 2 divide by 2"), result)

    async def test_reawaitable(self):
        """Awaiting the same AsyncWriter more than once re-runs the pipeline each time."""
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value

        pipeline = AsyncWriter.from_value(10, "log").map(track)

        self.assertEqual((10, "log"), await pipeline)
        self.assertEqual((10, "log"), await pipeline)
        self.assertEqual([10, 10], calls)
