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

* `map` (`>>`):
    * Monad(a).map(f a -> b) => Monad(b)
* `bind` (`>>=`):
    * Monad(a).bind(f a -> Monad(b)) => Monad(b)
* `apply` (`<*>`):
    * Monad(a).apply(Monad(f a -> b)) => Monad(b)
* `apply2` (`<*>`):
    * Monad(f a -> b).apply2(Monad(a)) => Monad(b)
* `run`: (IO, Reader, Writer, State, Lazy, Sequence)

## Maybe

Haskell : [Data.Maybe](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Maybe.html)

```python
# Maybe[TypeSource]
just: Maybe[int] = Just(10)
nothing: Maybe[int] = Nothing()
applicative = Just(lambda x: x + 1)

just.is_just()
just.is_nothing()

# Monad functions
just.map(mapping_function)  # Maybe maps the function (if not nothing)
just.bind(bind_function)  # Maybe binds the function (if not nothing)
just.apply(applicative)  # Apply the just value to the applicative (if not nothing)
applicative.apply2(just)  # Apply the applicative to the just value (if not nothing)

just.unwrap()  # Returns the just value or raises an Exception
just.unwrap_or(20)  # Returns the just value or else a default value
just.unwrap_or_else(lambda: 20)  # Return the just value or the result from the passed fucntion

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
applicative = Right(lambda x: x + 1)

right.is_right()
right.is_left()

# Monad functions
right.map(right_map_function)  # Maps only the right value with the function
right.map_left(left_map_function)  # Maps only the left value with the function
right.bind(right_bind_function)  # Binds only the right value with the function
right.bind_left(left_bind_function)  # Binds only the left value with the function
right.apply(applicative)  # Apply the right value to the applicative (if not left)
applicative.apply2(right)  # Apply the applicative to the right value (if not left)

right.unwrap()  # Returns the right value or raises an Exception.
right.unwrap_or(default_value_of_type_right)
right.unwrap_left_or(default_value_of_type_left)
right.unwrap_or_else(left_function)

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
applicative = Result(lambda x: x + 1)

result.is_ok()
result.is_err()

# Monad functions
result.map(ok_map_function)  # Maps only the ok value with the function
result.map_err(err_map_function)  # Maps only the err value with the function
result.bind(ok_bind_function)  # Binds only the ok value with the function
result.bind_err(err_bind_function)  # Binds only the err value with the function
result.apply(applicative)  # Apply the ok value to the applicative (if not err)
applicative.apply2(result)  # Apply the applicative to the ok value (if not err)

result.unwrap()  # Returns the result Ok value or raises an Exception containing the Err value.
result.unwrap_or(default_value_of_type_ok)
result.unwrap_or_else(err_function)
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
applicative = Success(lambda x: x + 1)

success.is_ok()
success.is_err()

# Monad functions
success.map(ok_map_function)  # Maps only the success value with the function
success.map_failure(err_map_function)  # Maps only the failure value with the function
success.bind(ok_bind_function)  # Binds only the success value with the function
success.bind_failure(err_bind_function)  # Binds only the failure value with the function
success.apply(applicative)  # Apply the success value to the applicative (if not failure)
applicative.apply2(success)  # Apply the applicative to the success value (if not failure)

success.unwrap()  # Returns the success value or raises the Exception contained in the Failure.
success.unwrap_or(default_value_of_type_ok)
success.unwrap_or_else(err_function)
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
io_value = IO[int] = IO(lambda: 10)
applicative = IO[Callable[[int], int]] = IO(lambda: lambda x: x + 1)

# Monad functions
io_value.map(map_function).run()  # Maps the resulting io value with the function
io_value.bind(bind_function).run()  # Binds the resulting io value with the function
io_value.apply(applicative).run()  # Apply the resulting io value to the applicative
applicative.apply2(io_value).run()  # Apply the resulting applicative to the io value

io_value.run()  # Returns the value 10
```

## ListMonad

Haskell: [Data.List](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-List.html)

```python
# ListMonad[TypeSource]
list_monad: ListMonad[int] = ListMonad([1, 2, 3, 4, 5, 6])
list_monad.take(4)  # ListMonad([1, 2, 3, 4])
list_monad.filter(lambda v: v > 4)  # ListMonad([5, 6])
list_monad.to_list()  # List[TypeSource]
applicative = ListMonad([lambda x: x + 1])

# Monad functions
list_monad.map(map_function)  # Maps all values with the function
list_monad.bind(bind_function)  # Binds all values the function
list_monad.apply(applicative)  # Applies all values to the applicative
applicative.apply2(list_monad)  # Applies the applicative to all values
```

## Sequence (lazy list)

Evaluates a given list in a lazy way.

```python
# Sequence[TypeSource]
sequence: Sequence[int] = Sequence([1, 2, 3, 4, 5, 6])
sequence.take(4)  # Sequence([1, 2, 3, 4])
sequence.filter(lambda v: v > 4)  # Sequence([5, 6])
sequence.run()  # List[TypeSource]
applicative = Sequence([lambda x: x + 1])

# Monad functions
sequence.map(map_function).run()  # Lazy mapping of all values with the function
sequence.bind(bind_function).run()  # Lazy binding of all values with the function
sequence.apply(applicative).run()  # Lazy applies all values to the applicative
applicative.apply2(sequence).run()  # Lazy applies the applicative to all values
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

applicative = Reader(lambda env: lambda x: x + env)

# Monad functions
reader.map(map_function).run(10)  # Map resulting value with the function
reader.bind(bind_function).run(10)  # Bind resulting value with the function
reader.apply(applicative).run(10)  # Apply resulting value to the applicative
applicative.apply2(reader).run(10)  # Apply resulting applicative to the value
```

## Writer

Haskell: [Control.Monad.Writer.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Writer-Lazy.html)

```python
# Writer[TypeSource, TypeMonoid]
writer: Writer[int, str] = Writer(10, 'hello')
writer.run()  # (10, 'hello')

writer.tell(' world')  # Writer(10, 'hello world')
writer.listen()  # Writer((10, 'hello world'), 'hello world')
Writer((10, lambda w: w + " world"), "hello").pass_()  # Writer((10, "hello world")

applicative = Writer(lambda x: x, '')

# Monad functions
writer.map(map_function).run()  # Map resulting value with the function
writer.bind(bind_function).run()  # Bind resulting value with the function
writer.apply(applicative).run()  # Apply resulting value to the applicative
applicative.apply2(writer).run()  # Apply resulting applicative to the value
```

## State

Haskell: [Control.Monad.State.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-State-Lazy.html)

```python
# State[TypeState, TypeSource]
state: State[str, int] = State(lambda state: (state, 10))
state.run('hi')  # (hi, 10)

State.get()  # State(lambda state: (state, state))
State.put(new_state)  # State(lambda state: (new_state, ()))

applicative = State(lambda state: (state, lambda x: x * 42))

# Monad functions
state.map(map_function).run('hi')  # Map resulting value with the function
state.bind(bind_function).run('hi')  # Bind resulting value with the function
state.apply(applicative).run('hi')  # Apply resulting value to the applicative
applicative.apply2(state).run('hi')  # Apply resulting applicative to the value
```

## LazyMonad

* [LazyMonad](https://www.philliams.com/monads-in-python/)

```python
# LazyMonad[TypeSource]
lazy: LazyMonad[int] = LazyMonad(lambda: 10)
lazy.run()  # TypeSource: 10

applicative = LazyMonad(lambda: lambda x: x * 42)

# Monad functions
lazy.map(map_function).run()  # Map resulting value with the function
lazy.bind(bind_function).run()  # Bind resulting value with the function
lazy.apply(applicative).run()  # Apply resulting value to the applicative
applicative.apply2(lazy).run()  # Apply resulting applicative to the value
```
