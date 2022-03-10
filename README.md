# Pymoliath
Python Monads library for Functional Programming

# About

Pymoliath is a python library containing Monads for monadic functional programming.
This library is inspired by Haskell as well as existing Monad implementations in python.

* [Pymonad](https://github.com/jasondelaat/pymonad) by jasondelaat
* [OSlash](https://github.com/dbrattli/OSlash) by dbrattli
* [typesafe-monads](https://github.com/correl/typesafe-monads) by correl

# Getting started

The pymoliath repository uses [poetry](https://python-poetry.org/) as dependency management tool.
The project can be installed using the following command.
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
```

## Either
Haskell : [Data.Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)
```python
right: Either[str, int] = Right(10)
left: Either[str, int] = Left("error")
```

## Try
Scala : [scala.util.Try](https://www.scala-lang.org/api/2.12.4/scala/util/Try.html)
```python
success: Try[int] = Success(10)
failure: Try[int] = Failure(TypeError("error"))
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
list_monad = ListMonad([10, 20])
```

## Reader
Haskell: [Control.Monad.Reader](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Reader.html)
```python
# Reader[TypeEnvironment, TypeSource]
reader: Reader[int, str] = Reader(lambda env: str(env))
reader.run(12) # '12'
```

## Writer
Haskell: [Control.Monad.Writer.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-Writer-Lazy.html)
```python
# Writer[TypeSource, TypeMonoid]
writer: Writer[int, str] = Writer(10, 'hi')
writer.run(12) # (12, 'hi')
```

## State
Haskell: [Control.Monad.State.Lazy](https://hackage.haskell.org/package/mtl-2.2.2/docs/Control-Monad-State-Lazy.html)
```python
# State[TypeState, TypeSource]
state: State[str, int] = State(lambda state: (state, 10))
state.run('hi') # (hi, 10)
```

## LazyMonad
* [LazyMonad](https://www.philliams.com/monads-in-python/)
```python
# State[TypeState, TypeSource]
lazy: LazyMonad[int] = LazyMonad(lambda: 10)
state.run() # 10
```