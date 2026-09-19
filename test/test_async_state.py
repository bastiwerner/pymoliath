import unittest

from pymoliath.async_state import AsyncState
from pymoliath.state import State


class TestAsyncState(unittest.IsolatedAsyncioTestCase):
    async def test_run_resolves_to_new_state_and_value(self):
        """AsyncState is not bare-awaitable - .run(state) is the terminal operation."""
        state = AsyncState.from_coroutine(async_increment)
        self.assertEqual((2, 1), await state.run(1))
        self.assertEqual((11, 10), await state.run(10))

    async def test_from_value_leaves_state_unchanged(self):
        state = AsyncState.from_value(42)
        self.assertEqual(("s", 42), await state.run("s"))

    async def test_from_state(self):
        sync_state = State(lambda s: (s, f"Hello {s}"))
        async_state = AsyncState.from_state(sync_state)
        self.assertEqual(("world", "Hello world"), await async_state.run("world"))

    async def test_pipeline_is_lazy_until_run_is_awaited(self):
        calls: list[int] = []

        def track(value: int) -> int:
            calls.append(value)
            return value + 1

        pipeline = AsyncState.from_coroutine(async_increment).map(track).map(track)
        self.assertEqual([], calls)

        new_state, result = await pipeline.run(1)
        self.assertEqual([1, 2], calls)
        self.assertEqual(2, new_state)
        self.assertEqual(3, result)

    async def test_map_sync_function(self):
        state = AsyncState.from_coroutine(async_increment).map(lambda x: x * 2)
        self.assertEqual((2, 2), await state.run(1))

    async def test_map_async_function(self):
        async def double(x: int) -> int:
            return x * 2

        state = AsyncState.from_coroutine(async_increment).map(double)
        self.assertEqual((2, 2), await state.run(1))

    async def test_bind_returning_async_state(self):
        state = AsyncState.get().bind(
            lambda value: AsyncState.put(value + 1).map(lambda _: value)
        )
        self.assertEqual((11, 10), await state.run(10))

    async def test_bind_returning_plain_state(self):
        state = AsyncState.from_value(10).bind(lambda x: State(lambda s: (s + 1, x)))
        self.assertEqual((2, 10), await state.run(1))

    async def test_bind_returning_awaitable(self):
        async def to_state_tuple(x: int) -> tuple[str, int]:
            return "new-state", x + 1

        state = AsyncState.from_value(10).bind(to_state_tuple)
        self.assertEqual(("new-state", 11), await state.run("s"))

    async def test_state_threading_across_bind_chain(self):
        """State mutations compose in order across a chain of binds."""
        pipeline = (
            AsyncState.get()
            .bind(lambda value: AsyncState.put(value + 1).map(lambda _: value))
            .bind(lambda first: AsyncState.get().map(lambda second: (first, second)))
        )
        result_state, (first, second) = await pipeline.run(1)
        self.assertEqual(2, result_state)
        self.assertEqual(1, first)
        self.assertEqual(2, second)

    async def test_apply(self):
        def f(x: int) -> int:
            return x * 2

        state = AsyncState.from_value(10).apply(AsyncState.from_value(f))
        self.assertEqual(("s", 20), await state.run("s"))

    async def test_apply2(self):
        def f(x: int) -> int:
            return x * 2

        state = AsyncState.from_value(f).apply2(AsyncState.from_value(10))
        self.assertEqual(("s", 20), await state.run("s"))

    async def test_get(self):
        state = AsyncState.get()
        self.assertEqual((5, 5), await state.run(5))

    async def test_put(self):
        state = AsyncState.put(99)
        self.assertEqual((99, ()), await state.run(1))

    async def test_reawaitable_with_different_initial_states(self):
        """The same AsyncState can be .run() with different initial states, each independently."""
        pipeline = AsyncState.from_coroutine(async_increment).map(lambda x: x * 2)
        self.assertEqual((2, 2), await pipeline.run(1))
        self.assertEqual((11, 20), await pipeline.run(10))


async def async_increment(state: int) -> tuple[int, int]:
    return state + 1, state
