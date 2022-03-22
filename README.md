# Pymoliath

Python Monads library for Functional Programming

# About

Pymoliath is a python library containing Monads for monadic functional programming. This library is inspired by Haskell
as well as existing Monad implementations in python.

* [Pymonad](https://github.com/jasondelaat/pymonad) by jasondelaat
* [OSlash](https://github.com/dbrattli/OSlash) by dbrattli
* [typesafe-monads](https://github.com/correl/typesafe-monads) by correl
* [std::result::Result](https://doc.rust-lang.org/std/result/enum.Result.html#) of rustlang

# Getting started

The pymoliath repository uses [poetry](https://python-poetry.org/) as dependency management tool. The project can be
installed using the following command.

```
poetry install
```

> Poetry will create a virtual environment and install all dependencies.
> Configuration: `poetry config virtualenvs.create false --local`

# Test

Run pymoliath tests using poetry run scripts. This command will execute all unittest in the virtual environment.

```
poetry run test
```

# Monads

Every Monad implementation of Pymoliath has the typical haskell Monad interface.

* `map` (>>): Monad(a).map(f a -> b) => Monad(b)
* `bind` (>>=): Monad(a).bind(f a -> Monad(b)) => Monad(b)
* `apply` (<*>): Monad(a).apply(Monad(f a -> b)) => Monad(b)
* `apply2` (<*>): Monad(f a -> b).apply2(Monad(a)) => Monad(b)
* `run`: (IO, Reader, Writer, State, Lazy)

## Maybe

Haskell : [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

```python
just: Maybe[int] = Just(10)
nothing: Maybe[int] = Nothing()

just.is_just()
just.is_nothing()
just.or_else(default_value)
just.maybe(just_function, default_value)
just.from_optional(value_or_none)
```

## Either

Haskell : [Data.Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)

```python
right: Either[str, int] = Right(10)
left: Either[str, int] = Left("error")

right.is_right()
right.is_left()
right.left_or_else(value_type_left)
right.right_or_else(value_type_right)
right.either(left_function, right_function)

Either.safe(unsafe_function, message)
```

## Result

The result monad is a combination of the rust result and scala try monads. It represents and handles computations which
can be fail. The return value is either the specified result type or a python exception type.

* Rust: [std::result::Result](https://doc.rust-lang.org/std/result/enum.Result.html#)
* Scala: [scala.util.Try](https://www.scala-lang.org/api/2.12.4/scala/util/Try.html)

```python
result: Result[int] = Ok(10)
error: Result[int] = Err(TypeError("error"))

result.is_ok()
result.is_err()
result.match(err_function, ok_function)
result.to_either()
result.expect(extend_error_message)

Result.safe(unsafe_function, message)
```

## IO

Haskell [System.IO](https://hackage.haskell.org/package/base-4.16.0.0/docs/System-IO.html#t:IO)

```python
# IO[TypeSource]
print_io: IO[None] = lambda x: IO(lambda: print(x))
print_io(10).run()
```

## ListMonad

Haskell: [Data.List](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-List.html)

```python
list_monad: ListMonad[int] = ListMonad([1, 2, 3, 4, 5, 6])
list_monad.take(4)  # ListMonad([1, 2, 3, 4])
list_monad.filter(lambda v: v > 4)  # ListMonad([5, 6])
list_monad.to_list()
```

## Sequence (lazy list)

Evaluates a given list in a lazy way.

```python
sequence: Sequence[int] = Sequence([1, 2, 3, 4, 5, 6])
sequence.take(4)  # ListMonad([1, 2, 3, 4])
sequence.filter(lambda v: v > 4)  # ListMonad([5, 6])
sequence.run()  # Lazy list evaluation result
```

## Reader

Haskell: [Control.Monad.Reader](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Reader.html)

```python
# Reader[TypeEnvironment, TypeSource]
reader: Reader[int, str] = Reader(lambda env: str(env))
reader.run(12)  # '12'
reader.local(change_environment_function)

Reader.ask()
```

## Writer

Haskell: [Control.Monad.Writer.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Writer-Lazy.html)

```python
# Writer[TypeSource, TypeMonoid]
writer: Writer[int, str] = Writer(10, 'hello')
writer.run(12)  # (12, 'hi')

writer.tell(' world')  # Writer(10, 'hi world')
writer.listen()  # Writer((10, 'hi'), 'hi')
Writer((10, lambda w: w + " world"), "hello").pass_()  # Writer((10, "hello world")
```

## State

Haskell: [Control.Monad.State.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-State-Lazy.html)

```python
# State[TypeState, TypeSource]
state: State[str, int] = State(lambda state: (state, 10))
state.run('hi')  # (hi, 10)

State.get()  # State(lambda state: (state, state))
State.put(new_state)  # State(lambda state: (new_state, ()))
```

## LazyMonad

* [LazyMonad](https://www.philliams.com/monads-in-python/)

```python
lazy: LazyMonad[int] = LazyMonad(lambda: 10)
lazy.run()  # 10
```
