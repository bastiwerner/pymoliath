import unittest
from typing import Any

from pymoliath.state import State
from pymoliath.util import compose


class TestState(unittest.TestCase):
    """
    Monad operations:
    ≡       Identical to
    >>=     bind, flatMap
    (.)     Function composition
    <*>     Applicative functor: (<$>) :: (Functor f) => (a -> b) -> f a -> f b
    <$>     Applicative functor: (<*>) :: f (a -> b) -> f a -> f b
    """

    def test_state_monad_left_identity_law(self):
        """Left identity law: return a >>= f ≡ f a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Left identity: The first monad law states that if we take a value,
        put it in a default context with return and then feed it to a function by using >>=,
        it’s the same as just taking the value and applying the function to it.
        """
        state_function = lambda x: State(lambda state: (state, x))

        self.assertEqual(state_function(10).run(1), State(lambda state: (state, 10)).bind(state_function).run(1))
        self.assertEqual((1, 10), state_function(10).run(1))
        self.assertEqual((1, 10), State(lambda state: (state, 10)).bind(state_function).run(1))

    def test_state_monad_right_identity_law(self):
        """Right identity law: m >>= return ≡ m
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Right identity: The second law states that if we have a monadic value
        and we use >>= to feed it to return, the result is our original monadic value.
        """
        state_value = State(lambda state: (state, f'Hello {state}'))
        bind_unit = lambda v: State(lambda state: (state, v))

        self.assertEqual(state_value.run('world'), state_value.bind(bind_unit).run('world'))
        self.assertEqual(('world', 'Hello world'), state_value.run('world'))
        self.assertEqual(('world', 'Hello world'), state_value.bind(bind_unit).run('world'))

    def test_just_monad_associativity_law(self):
        """Associativity law: (m >>= f) >>= g ≡ m >>= (x -> f x >>= g)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The final monad law says that when we have a chain of monadic function applications with >>=,
        it shouldn’t matter how they’re nested.
        """
        state_value = State(lambda state: (state, 42))
        f = lambda x: State(lambda state: (state, x + 1000))
        g = lambda y: State(lambda state: (state, y * 42))

        self.assertEqual(state_value.bind(f).bind(g).run(1), state_value.bind(lambda x: f(x).bind(g)).run(1))
        self.assertEqual((1, 43764), state_value.bind(f).bind(g).run(1))
        self.assertEqual((1, 43764), state_value.bind(lambda x: f(x).bind(g)).run(1))

    def test_state_monad_functor_identity_law(self):
        """Functors identity law: (m a >= f x -> x) ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Map the identity function over a monad container, the result should be the same monad container object.
        """
        state_value = State(lambda state: (state, 10))
        identity_function = lambda x: x

        self.assertEqual(state_value.map(identity_function).run(1), state_value.run(1))
        self.assertEqual((1, 10), state_value.map(lambda x: x).run(1))
        self.assertEqual((1, 10), state_value.run(1))

    def test_state_monad_functor_composition_law(self):
        """Functors composition law: map (f . g) x ≡ map f (map g x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The functor implementation should not break the composition of functions.
        """
        state_value = State(lambda state: (state, 42))

        f = lambda x: x + 1000
        g = lambda y: y * 42

        self.assertEqual(state_value.map(compose(f, g)).run(1), state_value.map(g).map(f).run(1))

    def test_state_monad_applicative_identity_law(self):
        """Applicative identity law: m (f x -> x) <*> m a ≡ m a
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        Wrap the identity function with a monad container. Apply a monad container over the result.
        The applicative identity law states this should result in an identical object.
        """
        state_value = State(lambda state: (state, 42))
        applicative_state = State(lambda state: (state, lambda x: x))

        self.assertEqual(state_value.apply(applicative_state).run(1), state_value.run(1))
        self.assertEqual(applicative_state.apply2(state_value).run(1), state_value.run(1))

    def test_state_monad_applicative_homomorphism_law(self):
        """Applicative homomorphism law: pure f <*> pure x = pure (f x)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        x = 42
        f = lambda x: x * 42
        state_value = State(lambda state: (state, x))
        state_applicative = State(lambda state: (state, f))

        self.assertEqual(state_value.apply(state_applicative).run(1), State(lambda state: (state, f(x))).run(1))
        self.assertEqual(state_applicative.apply2(state_value).run(1), State(lambda state: (state, f(x))).run(1))

    def test_state_monad_applicative_composition_law(self):
        """Applicative composition law: pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
        https://miklos-martin.github.io/learn/fp/2016/03/10/monad-laws-for-regular-developers.html

        The second law is the homomorphism law. If we wrap a function and an object in pure.
        We can then apply the wrapped function over the wrapped object.
        """
        w = State(lambda state: (state, 42))
        u = State(lambda state: (state, lambda x: x + 42))
        v = State(lambda state: (state, lambda x: x * 42))
        state_function_composition = State(lambda state: (state, lambda f, g: compose(f, g)))

        self.assertEqual(w.apply(v.apply(u.apply(state_function_composition))).run(1), w.apply(v).apply(u).run(1))
        self.assertEqual(state_function_composition.apply2(u).apply2(v).apply2(w).run(1), u.apply2(v.apply2(w)).run(1))

    def test_state_monad_representation(self):
        state_greeter = (State.get()
                         .bind(lambda name: (State.put("tintin")
                                             .bind(lambda _: State(lambda state: (state, f"hello, {name}!")))
                                             )
                               )
                         )

        result = state_greeter.run('adit')
        self.assertEqual(('tintin', 'hello, adit!'), result)
        self.assertEqual('tintin', result[0])
        self.assertEqual('hello, adit!', result[1])

        x = (State
             .get()
             .bind(lambda name: State(lambda _: ('tintin', name)))
             .bind(lambda name: State(lambda state: (state, f"hello, {name}!")))
             )
        self.assertEqual(('tintin', 'hello, adit!'), x.run('adit'))

        def test_function(state: Any):
            return state, 'a'

        state_value = State(test_function)

        self.assertEqual(str(state_value), f'State {str(test_function)}')
