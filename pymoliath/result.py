from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Any, Generic, TypeVar

from pymoliath import Either, Left, Right

TypeReturn = TypeVar("TypeReturn")
TypePure = TypeVar("TypePure")
TypeOk = TypeVar("TypeOk")


class Result(Generic[TypeOk], abc.ABC):
    """Try monad interface

    Subclasses of try monad should handle exceptions by returning either the Success monad or the Err monad.

    Try implementations:
    * Ok: represents the correct way which contains the Success result
    * Err: represents the Err way which contains the actual exception
    """
    _ok_value: TypeOk  # Private try monad Success value which should not be modified
    _err_value: Exception  # Private try monad Err value which should not be modified
    _is_err: bool  # Private try monad is Err flag which should not be modified

    def map(self: Result[TypeOk], function: Callable[[TypeOk], TypeReturn]) -> Result[TypeReturn]:
        """Result Monad functor interface (>=, map).

        The map function will execute the function by checking for any exception.
        It will return either a Result Monad of type Ok or of type Err.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult.

        Returns
        -------
        try: Result[TypeResult]
            Returns either a Result Monad of type Ok with the function result or otherwise
            a Result Monad of type Err containing the exception.
        """
        try:
            if self._is_err:
                return Err(self._err_value)
            else:
                return Ok(function(self._ok_value))
        except Exception as e:
            return Err(e)

    def bind(self: Result[TypeOk], function: Callable[[TypeOk], Result[TypeReturn]]) -> Result[TypeReturn]:
        """Ok try (either) monad bind interface (>>=, bind, flatMap).

        The bind function will execute the passed function and is also checking for any exception.
        It will return either a Result Monad of type Ok or of type Err.

        Parameters
        ----------
        function: Callable[[TypeSource], Result[TypeResult]]
            Function which takes a value of type TypeSource and returns a try monad with a new value type.

        Returns
        -------
        result: Result[TypeResult]
            Returns either a Result Monad of type Ok/Err from the function result or otherwise
            a Result Monad of type Err containing the exception.
        """
        try:
            if self._is_err:
                return Err(self._err_value)
            else:
                return function(self._ok_value)
        except Exception as e:
            return Err(e)

    def apply(self: Result[TypeOk], applicative: Result[Callable[[TypeOk], TypeReturn]]) -> Result[TypeReturn]:
        """Result Monad applicative interface for Result Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Result[Callable[Callable[[TypeSource], TypeResult]]
            Applicative Result Monad which contains a function.

        Returns
        -------
        result: Result[TypeResult]
            Returns a new Result Monad (Ok/Err) with the applicative applied.
        """

        def binder(applicative_function: Callable[[TypeOk], TypeReturn]) -> Result[TypeReturn]:
            def inner(x: TypeOk) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: Result[Callable[[TypePure], TypeReturn]], applicative_value: Result[TypePure]) -> Result[
        TypeReturn]:
        """Result Monad applicative interface for Result Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Result[TypePure]
            Result Monad value which will be applied to the function within the Result Monad of self.

        Returns
        -------
        try: Result[TypeResult]:
            Returns a new Result Monad (Ok/Err) with the value applied to the internal applicative function.
        """

        def binder(applicative_function: Callable[[TypePure], TypeReturn]) -> Result[TypeReturn]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def ok_or_else(self, default_value: TypeOk) -> TypeOk:
        if self._is_err:
            return default_value
        else:
            return self._ok_value

    def err_or_else(self, default_value: Exception) -> Exception:
        if self._is_err:
            return self._err_value
        else:
            return default_value

    def match(self: Result[TypeOk], err_function: Callable[[Exception], TypeReturn],
              ok_function: Callable[[TypeOk], TypeReturn]) -> TypeReturn:
        """Result Monad specific function to handle railroad orientated types.

        Parameters
        ----------
        err_function: Callable[[TypeErr], TypeResult]
            Callback function for either monads of type Err
        ok_function: Callable[[TypeSource], TypeResult]
            Callback function for either monads of type Ok
        """
        if self._is_err:
            return err_function(self._err_value)
        else:
            return ok_function(self._ok_value)

    def to_either(self: Result[TypeOk]) -> Either[Exception, TypeOk]:
        """Result Monad specific function to return an Either Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Result Monad as Either Monad
        """
        if self._is_err:
            return Left(self._err_value)
        else:
            return Right(self._ok_value)

    def is_ok(self: Result[TypeOk]) -> bool:
        """Try monad is Success function

        Returns
        -------
        result: bool
            True: if try monad is of type Success, False: if try monad is of type Err
        """
        return not self._is_err

    def is_err(self: Result[TypeOk]) -> bool:
        """Try monad is Err function

        Returns
        -------
        result: bool
            True: if try monad is of type Err, False: if try monad is of type Success
        """
        return self._is_err

    @classmethod
    def safe(cls, function: Callable[[], TypeReturn], msg: str = '') -> Result[TypeReturn]:
        """Either Monad method (try_except) which wraps a function that may raise an exception.

        Parameters
        ----------
        function: Callable[[], TypeResult]
            Callable function which may raise an exception
        msg: str (Optional)
            Message to be added in case of an exception

        Returns
        -------
        either: Either[Exception, TypeResult]
            Returns an Either Monad which contains either the function result or an Exception with a message added.
        """
        try:
            return Ok(function())
        except Exception as e:
            return Err(Exception(f'{msg}{e}'))

    def expect(self: Result[TypeOk], msg: str) -> Result[TypeOk]:
        if self._is_err:
            new_exception = self._err_value
            new_exception.args = (msg + new_exception.args[0],) + new_exception.args[1:]
            return Err(new_exception)
        else:
            return self

    @abc.abstractmethod
    def __str__(self: Result[TypeOk]) -> str:
        pass

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
      pass

    def __repr__(self: Result[TypeOk]) -> str:
        return str(self)


class Ok(Result[TypeOk]):

    def __init__(self, value: TypeOk):
        self._ok_value = value
        self._err_value = Any
        self._is_err = False

    def __str__(self: Ok[TypeOk]) -> str:
        return f'Ok({self._ok_value})'

    def __eq__(self: Ok[TypeOk], other: object) -> bool:
        return isinstance(other, Ok) and str(self) == str(other) and type(self._ok_value) == type(other._ok_value)  # type: ignore


class Err(Result[Any]):

    def __init__(self, value: Exception):
        assert isinstance(value, Exception), "Err value must be of type Exception"
        self._ok_value = Any
        self._err_value = value
        self._is_err = True

    def __str__(self: Err) -> str:
        return f'Err({self._err_value})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Err) and str(self) == str(other) and type(self._err_value) == type(other._err_value)
