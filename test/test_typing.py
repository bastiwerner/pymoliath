# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static typing regression tests.

These run under pytest like any other test, but their real purpose is to be checked by pyright:
`assert_type` fails type checking if the inferred type drifts, and every `# pyright: ignore[...]`
must still be needed (`reportUnnecessaryTypeIgnoreComment`), so the negative cases keep proving
that wrong code is rejected.

The monads follow Rust's semantics: both variants carry all type parameters, so e.g. the error
type of a `Result` is fixed by the receiver and `bind` on a `Result[int, str]` is a
`Result[R, str]`. The parameters are covariant and the unused side of a variant is `Never`, so
bare `Ok(1)`/`Err("e")`/`Nil()` are assignable to any matching `Result`/`Option`.
"""

import asyncio
from collections.abc import Callable

from typing_extensions import Never, assert_never, assert_type

from pymoliath.aio.async_maybe import AsyncMaybe
from pymoliath.aio.async_result import AsyncResult
from pymoliath.aio.async_try import AsyncTry
from pymoliath.either import Either, Left, Right
from pymoliath.exception import Failure, Success, Try
from pymoliath.maybe import Just, Maybe, Nothing
from pymoliath import result as result_
from pymoliath.option import Nil, Option, Some, is_some
from pymoliath.result import Err, Ok, Result, is_err, is_ok, map2, result_safe
from pymoliath.util import flow


def parse(value: str) -> Result[int, str]:
    return Ok(int(value)) if value.isdigit() else Err("not a number")


def halve(value: int) -> Result[float, str]:
    return Ok(value / 2)


def test_result_bind_keeps_the_error_type() -> None:
    result = parse("4")

    assert_type(result.bind(lambda x: Ok(x / 2)), Result[float, str])
    assert_type(result.bind(halve), Result[float, str])
    assert_type(result.bind(lambda x: halve(x)), Result[float, str])
    assert_type(
        result.bind(lambda x: Ok(x + 1) if x > 0 else Err("negative")),
        Result[int, str],
    )

    # Rust: `let x: Result<u32, String> = Ok(10); x.and_then(|v| Ok("sdr"))` is `Result<&str, String>`.
    x: Result[int, str] = Ok(10)
    assert_type(x.bind(lambda _: Ok("sdr")), Result[str, str])


def test_result_chaining() -> None:
    result = parse("4")

    assert_type(result.map(lambda x: x + 1), Result[int, str])
    assert_type(result.map_err(lambda e: len(e)), Result[int, int])
    assert_type(
        result.bind(halve).map(lambda x: str(x)).map_err(lambda e: e.upper()),
        Result[str, str],
    )
    assert_type(result.zip(halve(1)), Result[tuple[int, float], str])
    assert_type(result.ok(), Option[int])
    assert_type(result.unwrap_or(0), int)
    assert_type(result.unwrap_or_else(lambda e: len(e)), int)


def test_result_constructors() -> None:
    # A bare Ok has no error type yet (`Never`), like in Rust. A bare Err leaves its Ok type open,
    # so it can be solved from context (e.g. the other branch of a conditional lambda).
    assert_type(Ok(1), Ok[int, Never])
    assert_type(Err("error"), Err[Never, str])

    # Covariance: bare variants are assignable without annotations at the construction site.
    bare_ok: Result[int, str] = Ok(1)
    bare_err: Result[int, str] = Err("error")
    assert bare_ok != bare_err

    widened: Result[int, str] = Ok(True)
    failed: Result[int, str] = Err("error")
    assert widened == Ok(True)
    assert failed == Err("error")


def test_result_rejects_wrong_types() -> None:
    # Err at runtime, so the lambdas below never run; pyright still checks both variants.
    result = parse("not a number")

    # The lambda parameter is typed in both variants, so misuse inside it is reported.
    result.bind(lambda x: halve(str(x)))  # pyright: ignore[reportArgumentType]
    # The error type is fixed by the receiver, so a different one is rejected at the bind.
    result.bind(lambda _: Err(3))  # pyright: ignore[reportArgumentType]
    # Like Rust, a bare `Ok(10)` has no error type yet (`Never`); annotate it to bind fallible code.
    Ok(10).bind(halve)  # pyright: ignore[reportArgumentType]


def lookup(value: int) -> Maybe[int]:
    return Just(value) if value > 0 else Nothing()


def reciprocal(value: int) -> Maybe[float]:
    return Just(1 / value) if value else Nothing()


def test_maybe_bind_with_lambda() -> None:
    maybe = lookup(4)

    assert_type(maybe.map(lambda x: str(x)), Maybe[str])
    assert_type(maybe.bind(lambda x: reciprocal(x)), Maybe[float])
    assert_type(maybe.bind(lambda x: Just(x / 2)), Maybe[float])
    # Nothing's type is left open, so it is solved from the other branch.
    assert_type(maybe.bind(lambda x: Just(x / 2) if x else Nothing()), Maybe[float])

    mapped: Maybe[str] = maybe.map(lambda x: str(x))
    assert mapped == Just("4")


def test_maybe_rejects_wrong_types() -> None:
    maybe = lookup(0)

    maybe.bind(lambda x: reciprocal(str(x)))  # pyright: ignore[reportArgumentType]


def divide(value: int) -> Either[str, float]:
    return Right(10 / value) if value else Left("division by zero")


def test_either_bind_keeps_the_left_type() -> None:
    either = divide(2)

    assert_type(either.map(lambda x: int(x)), Either[str, int])
    assert_type(either.bind(lambda x: divide(int(x))), Either[str, float])
    assert_type(either.bind(lambda x: Right(str(x))), Either[str, str])
    assert_type(either.map_left(lambda e: len(e)), Either[int, float])

    widened: Either[str, int] = Right(True)
    assert widened == Right(True)


def test_either_rejects_wrong_types() -> None:
    either = divide(0)

    either.bind(lambda x: divide(str(x)))  # pyright: ignore[reportArgumentType]
    either.bind(lambda _: Left(3))  # pyright: ignore[reportArgumentType]


def find(value: int) -> Option[int]:
    return Some(value) if value > 0 else Nil()


def half(value: int) -> Option[float]:
    return Some(value / 2)


def test_option_bind_with_lambda() -> None:
    option = find(4)

    assert_type(option.map(lambda x: str(x)), Option[str])
    assert_type(option.bind(lambda x: half(x)), Option[float])
    assert_type(option.bind(lambda x: Some(x / 2)), Option[float])
    # Converting methods return a union of the precise variants, which is a Result[int, str].
    converted: Result[int, str] = option.ok_or("missing")
    assert converted == Ok(4)
    # Nil's type is left open, so it is solved from the other branch.
    assert_type(option.bind(lambda x: Nil() if x > 10 else Some(x / 2)), Option[float])

    empty: Option[int] = Nil()
    assert empty == Nil()


def test_option_rejects_wrong_types() -> None:
    option = find(0)

    option.bind(lambda x: half(str(x)))  # pyright: ignore[reportArgumentType]
    option.or_(Some("text"))  # pyright: ignore[reportArgumentType]


def attempt(value: int) -> Try[int]:
    return Success(value) if value else Failure(ValueError("zero"))


def test_try_bind_with_lambda() -> None:
    attempted = attempt(4)

    assert_type(attempted.bind(lambda x: attempt(x * 2)), Try[int])
    assert_type(attempted.bind(lambda x: Success(str(x))), Try[str])
    assert_type(attempted.map(lambda x: x / 2), Try[float])
    assert_type(attempted.to_result(), Result[int, Exception])
    assert_type(attempted.to_either(), Either[Exception, int])
    # Failure's type is left open, so it is solved from the other branch.
    assert_type(
        attempted.bind(lambda x: Success(x / 2) if x else Failure(ValueError("zero"))),
        Try[float],
    )


def test_async_bind_keeps_the_error_type() -> None:
    async_result = AsyncResult.from_result(parse("4"))
    assert_type(async_result, AsyncResult[int, str])
    assert_type(async_result.bind(lambda x: Ok(x / 2)), AsyncResult[float, str])
    assert_type(async_result.bind(halve), AsyncResult[float, str])
    assert_type(AsyncResult.from_ok(1), AsyncResult[int, Never])

    # A Maybe[int] from a function (an annotated `= Nothing()` would narrow to `Nothing`).
    nothing = lookup(0)
    fallback = AsyncMaybe.from_maybe(nothing).or_(AsyncMaybe.from_value(20))
    assert_type(fallback, AsyncMaybe[int])

    failure: Try[str] = Failure(ValueError("x"))
    recovered = AsyncTry.from_try(failure).bind_failure(
        lambda e: AsyncTry.from_success(str(e))
    )
    assert_type(recovered, AsyncTry[str])

    async def run() -> None:
        assert await async_result.bind(halve) == Ok(2.0)
        assert await fallback == Just(20)
        assert await recovered == Success("x")

    asyncio.run(run())


def test_async_rejects_wrong_types() -> None:
    AsyncResult.from_result(parse("4")).bind(lambda _: Err(3))  # pyright: ignore[reportArgumentType]
    AsyncResult.from_ok(10).bind(halve)  # pyright: ignore[reportArgumentType]


def test_variant_methods_keep_the_precise_type() -> None:
    assert_type(Ok(1).map(lambda x: str(x)), Ok[str, Never])
    assert_type(Err("e").map_err(lambda e: len(e)), Err[Never, int])
    assert_type(Some(1).map(lambda x: x / 2), Some[float])
    assert_type(Nil().map(lambda x: x), Nil)
    assert_type(Success(1).map(lambda x: str(x)), Try[str])  # map catches exceptions
    assert_type(Right(1).map(lambda x: str(x)), Right[str, Never])


def test_narrowing() -> None:
    result = parse("4")
    if is_ok(result):
        assert_type(result, Ok[int, str])
        assert_type(result.value, int)
    if is_err(result):
        assert_type(result.error, str)

    match result:
        case Ok(value):
            assert_type(value, int)
        case Err(error):
            assert_type(error, str)
        case _:
            assert_never(result)

    option = find(4)
    if is_some(option):
        assert_type(option.value, int)


def test_flatten_map2_and_apply() -> None:
    nested: Result[Result[int, str], str] = Ok(Ok(1))
    assert_type(nested.flatten(), Result[int, str])
    assert_type(Some(Some(1)).flatten(), Option[int])

    assert_type(map2(parse("1"), parse("2"), lambda a, b: a / b), Result[float, str])
    function: Result[Callable[[int], str], str] = Ok(str)
    assert_type(parse("1").apply(function), Result[str, str])
    assert_type(parse("1").match(ok=lambda x: x * 2, err=lambda e: len(e)), int)
    assert_type(parse("1").merge(), int | str)


def unwrap_is_never() -> None:
    """Type-checked only, never called: unwrap on Err/Nil raises."""
    assert_type(Err("e").unwrap(), Never)
    assert_type(Nil().unwrap(), Never)


def test_unwrap_and_safe() -> None:
    assert_type(parse("1").unwrap(), int)
    assert_type(result_safe(lambda: 1), Result[int, Exception])
    assert_type(
        result_safe(lambda: int("x"), exceptions=(ValueError, KeyError)),
        Result[int, ValueError | KeyError],
    )


def test_flow_keeps_the_types() -> None:
    def double(x: int) -> int:
        return x * 2

    assert_type(flow(parse("1"), result_.map(double)), Result[int, str])
    assert_type(flow(parse("1"), result_.map(double), result_.unwrap_or(0)), int)
    # Lambdas passed directly to flow are inferred step by step.
    assert_type(flow(1, lambda x: x + 1, lambda x: str(x)), str)


def test_fixed_types_reject_mismatches() -> None:
    # unwrap_or on an Option[int] needs an int; a bare Ok(1) can't take a Result[float, str]
    # function (its error type is Never); Nil is not an Option[int] value source.
    find(1).unwrap_or("text")  # pyright: ignore[reportArgumentType]
    Ok(1).and_(parse("1"))  # pyright: ignore[reportArgumentType]
    parse("1").or_(Ok("text"))  # pyright: ignore[reportArgumentType]
