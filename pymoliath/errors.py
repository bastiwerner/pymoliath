"""
# Errors

Exceptions raised by pymoliath itself.
"""


class UnwrapError(Exception):
    """Raised when `unwrap` is called on an empty or failed container (`Err`, `Left`, `Nil`,
    `Nothing`).

    The container that was unwrapped is available as `container`. If it wrapped an exception, that
    exception is chained as `__cause__`, so the original error is never lost.

    Examples
    --------
    >>> from pymoliath.result import Err
    >>> try:
    ...     Err(ValueError("boom")).unwrap()
    ... except UnwrapError as error:
    ...     print(error.container, type(error.__cause__).__name__)
    Err(boom) ValueError
    """

    container: object

    def __init__(self, container: object, message: str) -> None:
        super().__init__(message)
        self.container = container
