"""Internal helpers shared by the Async* monads. Not part of the public API."""

import inspect
from typing import Awaitable, TypeVar, Union, cast

TypeResult = TypeVar("TypeResult")


async def resolve(value: Union[TypeResult, Awaitable[TypeResult]]) -> TypeResult:
    """Awaits `value` if it is awaitable, otherwise returns it unchanged.

    Lets map/bind/filter/inspect-style callbacks on the Async* monads be either plain sync
    functions or `async def` functions interchangeably: the shape of whatever they return is
    detected here rather than requiring the caller to declare it upfront.
    """
    if inspect.isawaitable(value):
        return await value
    return cast(TypeResult, value)
