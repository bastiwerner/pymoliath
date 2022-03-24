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
# Maybe[TypeSource]
just: Maybe[int] = Just(10)
nothing: Maybe[int] = Nothing()

just.is_just()
just.is_nothing()

just.unwrap_or(default_value)

just.filter(filter_function)
just.match(just_function, default_function)

just.from_optional(value_or_none)
```

## Either

Haskell : [Data.Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)

```python
# Either[TypeLeft, TypeRight]
right: Either[str, int] = Right(10)
left: Either[str, int] = Left("error")

right.is_right()
right.is_left()

right.map(right_map_function)
right.map_left(left_map_function)

right.bind(right_bind_function)
right.bind_left(left_bind_function)

right.unwrap_or(default_value_of_type_right)
right.unwrap_left_or(default_value_of_type_left)

right.match(left_function, right_function)
right.to_result()  # Result[TypeRight, TypeLeft]

Either.safe(unsafe_function, message)  # Either[Exception, TypeRight]
```

## Result

* Rust: [std::result::Result](https://doc.rust-lang.org/std/result/enum.Result.html#)

```python
# Result[TypeOk, TypeErr]
result: Result[int, str] = Ok(10)
error: Result[int, str] = Err("error")

result.is_ok()
result.is_err()

result.map(ok_map_function)
result.map_err(err_map_function)

result.bind(ok_bind_function)
result.bind_err(err_bind_function)

result.unwrap_or(default_value_of_type_ok)
result.unwrap_err_or(default_exception_value)

result.match(err_function, ok_function)
result.to_either()  # Either[TypeErr, TypeOk]

Result.safe(unsafe_function)  # Result[TypeOk, Exception]
```

## Try

It represents and handles computations which might be fail. The return value is either the specified result type or a
python exception type.

* Scala: [scala.util.Try](https://www.scala-lang.org/api/2.12.4/scala/util/Try.html)

```python
# Try[TypeSource]
success: Try[int] = Success(10)
failure: Try[int] = Failure(Exception("error"))

success.is_ok()
success.is_err()

success.map(ok_map_function)
success.map_failure(err_map_function)

success.bind(ok_bind_function)
success.bind_failure(err_bind_function)

success.unwrap_or(default_value_of_type_ok)
success.unwrap_failure_or(default_exception_value)

success.match(err_function, ok_function)
success.to_either()  # Either[Exception, TypeSource]
success.to_result()  # Result[TypeSource, Exception]

Try.safe(unsafe_function)  # Try[TypeSource]
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
# ListMonad[TypeSource]
list_monad: ListMonad[int] = ListMonad([1, 2, 3, 4, 5, 6])
list_monad.take(4)  # ListMonad([1, 2, 3, 4])
list_monad.filter(lambda v: v > 4)  # ListMonad([5, 6])
list_monad.to_list()  # List[TypeSource]
```

## Sequence (lazy list)

Evaluates a given list in a lazy way.

```python
# Sequence[TypeSource]
sequence: Sequence[int] = Sequence([1, 2, 3, 4, 5, 6])
sequence.take(4)  # Sequence([1, 2, 3, 4])
sequence.filter(lambda v: v > 4)  # Sequence([5, 6])
sequence.run()  # List[TypeSource]
```

## Reader

Haskell: [Control.Monad.Reader](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Reader.html)

```python
# Reader[TypeEnvironment, TypeSource]
reader: Reader[int, str] = Reader(lambda env: str(env))
reader.run(12)  # '12'
reader.local(change_environment_function)

# Reader[TypeEnvironment, TypeEnvironment]
Reader.ask()  # Reader(lambda env: env)
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
# LazyMonad[TypeSource]
lazy: LazyMonad[int] = LazyMonad(lambda: 10)
lazy.run()  # TypeSource: 10
```
