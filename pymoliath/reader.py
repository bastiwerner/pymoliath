"""
# Reader Monad

The Reader Monad (also called the Environment monad) represents a computation that can read
values from a shared environment. It wraps a `Callable[[TypeEnv], TypeSource]`: a function that,
given an environment, produces a value, allowing environment-dependent computations to be composed
with `map`/`bind` without threading the environment through every function call by hand.

* Haskell: [Control.Monad.Reader](https://hackage.haskell.org/package/mtl/docs/Control-Monad-Reader.html)

This implementation is heavily inspired by Haskell's `Reader` monad.

The `Reader[TypeEnv, TypeSource]` type wraps a `Callable[[TypeEnv], TypeSource]`: a computation
which, given a value of type `TypeEnv`, produces a value of type `TypeSource`.

```python
Reader[TypeEnv, TypeSource]
```

## Practical Examples and Benefits:

The Reader Monad is particularly useful for dependency injection: instead of passing a
configuration object, database connection, or other shared context as an explicit parameter to
every function in a call chain, functions are written as `Reader`s over that environment and
composed with `map`/`bind`; the environment is only supplied once, when the whole pipeline is run.

### Benefits:
1. Implicit Environment Threading: The environment doesn't need to be passed explicitly through
   every function in a call chain - `bind` takes care of it.
2. Composability: Environment-dependent computations can be built up with `map`/`bind` just like
   any other monad, and only evaluated once `run(env)` is called.
3. Local Overrides: `local` lets a sub-computation run against a modified environment without
   affecting the environment seen by the rest of the pipeline.

#### Example: Reading configuration without threading it through every call.

```python
# Without Reader (Imperative)
def get_greeting(config):
    return f"{config['greeting']}, {get_name(config)}!"

def get_name(config):
    return config["name"]

print(get_greeting({"greeting": "Hello", "name": "World"}))

# With Reader (Functional)
get_name = Reader(lambda config: config["name"])
get_greeting = get_name.map(lambda name: f"Hello, {name}!")
print(get_greeting.run({"name": "World"}))
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Type, TypeVar

from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeEnv = TypeVar("TypeEnv")
TypeResult = TypeVar("TypeResult")


class Reader(Generic[TypeEnv, TypeSource]):
    """Reader Monad implementation

    The Reader monad (also called the Environment monad). Represents a computation, which can read values from a shared
    environment, pass values from function to function, and execute sub-computations in a modified environment.

    """

    __slots__ = ("_value",)

    _value: Callable[
        [TypeEnv], TypeSource
    ]  # Private reader monad value of type callable which should not be modified

    def __init__(self, value: Callable[[TypeEnv], TypeSource]) -> None:
        """Reader Monad constructor which takes a callable of type Callable[[TypeEnv], TypeSource].

        Parameters
        ----------
        value: Callable[[TypeEnv], TypeSource]
            Callable to be stored in the Reader Monad, invoked with the environment when `run` is called.

        Examples
        --------
        >>> reader: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> reader.run({"value": 10})
        10
        """
        if not callable(value):
            raise TypeError("Reader value must be of type Callable")
        self._value = value

    def map(
        self: Reader[TypeEnv, TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> Reader[TypeEnv, TypeResult]:
        """Reader monad functor interface (>=, map).

        Definition: Reader(f: e -> a) >= f: a -> b => Reader(f: e -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        reader: Reader[TypeEnv, TypeResult]
            Returns a reader monad with the function result as value

        Examples
        --------
        >>> reader: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> reader.map(lambda x: x + 1).run({"value": 10})
        11
        """
        return Reader(lambda env: function(self.run(env)))

    def bind(
        self: Reader[TypeEnv, TypeSource],
        function: Callable[[TypeSource], Reader[TypeEnv, TypeResult]],
    ) -> Reader[TypeEnv, TypeResult]:
        """Reader monad bind interface (>>=, bind, flatMap).

        Definition: Reader(f: e -> a) >= f: a -> Reader(f: e -> b) => Reader(f: e -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], Reader[TypeEnv, TypeResult]]
            Function which takes a value of type TypeSource and returns a reader monad of type TypeResult.

        Returns
        -------
        reader: Reader[TypeEnv, TypeResult]
            Returns a reader monad with the function result

        Examples
        --------
        >>> reader: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> reader.bind(lambda x: Reader(lambda env: x + env["extra"])).run({"value": 10, "extra": 1})
        11
        """
        return Reader(lambda x: function(self.run(x)).run(x))

    def apply(
        self: Reader[TypeEnv, TypeSource],
        applicative: Reader[TypeEnv, Callable[..., TypeResult]],
    ) -> Reader[TypeEnv, TypeResult]:
        """Reader monad applicative interface for reader monads containing a value (<*>).

        Definition: Reader(f: e -> a) <*> Reader(f: e -> f: a -> b) => Reader(f: e -> b)

        Parameters
        ----------
        applicative: Reader[TypeEnv, Callable[[TypeSource], TypeResult]]
            Applicative reader monad which contains a function and will be applied to the reader monad containing
            a value.

        Returns
        -------
        reader: Reader[TypeEnv, TypeResult]
            Applies a reader monad containing a value to a reader monad containing a function.

        Examples
        --------
        >>> val: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> func: Reader[dict, Callable[[int], int]] = Reader(lambda env: (lambda x: x * 2))
        >>> val.apply(func).run({"value": 10})
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Reader[TypeEnv, TypeResult]:
            """Maps the applicative's function, curried, over this Reader monad's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Reader[TypeEnv, Callable[..., TypeResult]],
        applicative_value: Reader[TypeEnv, Any],
    ) -> Reader[TypeEnv, TypeResult]:
        """Reader monad applicative interface for reader monads containing a function (<*>).

        Definition: Reader(f: e -> f: a -> b) <*> Reader(f: e -> a) => Reader(f: e -> b)

        Parameters
        ----------
        applicative_value: Reader[TypeEnv, TypeSource]
            Reader monad value which will be applied to the reader monad containing a function

        Returns
        -------
        reader: Reader[TypeEnv, TypeResult]
            Applies a reader monad containing a function to a reader monad with a value or function.

        Examples
        --------
        >>> func: Reader[dict, Callable[[int], int]] = Reader(lambda env: (lambda x: x * 2))
        >>> val: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> func.apply2(val).run({"value": 10})
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Reader[TypeEnv, TypeResult]:
            """Maps the applicative value's function, curried, over this Reader monad's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    @classmethod
    def ask(cls: Type[Reader[TypeEnv, TypeEnv]]) -> Reader[TypeEnv, TypeEnv]:
        """Reader monad special function ask to return a new reader monad which returns just the environment.

        Definition: Reader.ask() => Reader(f: env -> env)

        Returns
        -------
        result: TypeSource
            Calls the reader monad function by passing the environment and returns the result.

        Examples
        --------
        >>> Reader[dict, dict].ask().run({"value": 10})
        {'value': 10}
        """

        def identity(env: TypeEnv) -> TypeEnv:
            """Returns the environment unchanged."""
            return env

        return cls(identity)

    def local(
        self: Reader[TypeEnv, TypeSource], function: Callable[[TypeEnv], TypeEnv]
    ) -> Reader[TypeEnv, TypeSource]:
        """Reader monad specific function local. Allows to affect the environment before the next reader gets it.

        Parameters
        ----------
        function: Callable[[TypeEnv], TypeEnv]
            The function to modify the environment.

        Returns
        -------
        reader: Reader[TypeEnv, TypeSource]
            Returns a new reader instance with the modified environment from the passed function.

        Examples
        --------
        >>> reader: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> reader.local(lambda env: {**env, "value": env["value"] + 1}).run({"value": 10})
        11
        """
        return Reader(lambda env: self.run(function(env)))

    def run(self: Reader[TypeEnv, TypeSource], env: TypeEnv) -> TypeSource:
        """Reader monad lazy run function to start the reader and pass an environment.

        Definition: Reader(f: env -> a).run(env) => a

        Returns
        -------
        result: TypeSource
            Calls the reader monad function by passing the environment and returns the result.

        Examples
        --------
        >>> reader: Reader[dict, int] = Reader(lambda env: env["value"])
        >>> reader.run({"value": 10})
        10
        """
        return self._value(env)

    def __str__(self: Reader[TypeEnv, TypeSource]) -> str:
        """Returns the string representation of the Reader Monad.

        Examples
        --------
        >>> str(Reader(lambda env: env))  # doctest: +ELLIPSIS
        'Reader(<function...>)'
        """
        return f"Reader({self._value})"

    def __repr__(self: Reader[TypeEnv, TypeSource]) -> str:
        """Returns the string representation of the Reader Monad (same as __str__).

        Examples
        --------
        >>> repr(Reader(lambda env: env))  # doctest: +ELLIPSIS
        'Reader(<function...>)'
        """
        return str(self)
