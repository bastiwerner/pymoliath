# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static typing regression tests.

These run under pytest like any other test, but their real purpose is to be checked by pyright:
`assert_type` fails type checking if the inferred type drifts, and every `# pyright: ignore[...]`
must still be needed (`reportUnnecessaryTypeIgnoreComment`), so the negative cases keep proving
that wrong code is rejected.

The monads follow Rust's semantics: both variants carry all type parameters, so e.g. the error
type of a `Result` is fixed by the receiver and `bind` on a `Result[int, str]` is a
`Result[R, str]`.
"""

import asyncio

from typing_extensions import Never, assert_type

from pymoliath.aio.async_maybe import AsyncMaybe
from pymoliath.aio.async_result import AsyncResult
from pymoliath.aio.async_try import AsyncTry
from pymoliath.either import Either, Left, Right
from pymoliath.exception import Failure, Success, Try
from pymoliath.maybe import Just, Maybe, Nothing
from pymoliath.option import Nil, Option, Some
from pymoliath.result import Err, Ok, Result


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
    assert_type(Err("error").error, str)

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
    assert_type(option.ok_or("missing"), Result[int, str])
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

    nothing: Maybe[int] = Nothing()
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
