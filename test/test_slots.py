import unittest

from pymoliath.continuation import Continuation
from pymoliath.either import Left, Right
from pymoliath.exception import Failure, Success
from pymoliath.io import IO
from pymoliath.lazy import LazyMonad, Sequence
from pymoliath.list import ListMonad
from pymoliath.maybe import Just, Nothing
from pymoliath.option import Nil, Some
from pymoliath.reader import Reader
from pymoliath.result import Err, Ok
from pymoliath.state import State
from pymoliath.writer import Writer


class TestSlots(unittest.TestCase):
    """Locks in __slots__ on the monad container classes: instances must not gain a __dict__
    or accept arbitrary new attributes, so a future refactor can't silently reintroduce them.
    """

    def instances(self):
        return {
            "Right": Right(1),
            "Left": Left(1),
            "Some": Some(1),
            "Nil": Nil(),
            "Just": Just(1),
            "Nothing": Nothing(),
            "Ok": Ok(1),
            "Err": Err(1),
            "Success": Success(1),
            "Failure": Failure(Exception("error")),
            "IO": IO(lambda: 1),
            "Reader": Reader(lambda env: env),
            "State": State(lambda state: (state, 1)),
            "Writer": Writer(1, "log"),
            "Continuation": Continuation(lambda k: k(1)),
            "LazyMonad": LazyMonad(lambda: 1),
            "Sequence": Sequence([1, 2, 3]),
            "ListMonad": ListMonad([1, 2, 3]),
        }

    def test_no_instance_dict(self):
        for name, instance in self.instances().items():
            with self.subTest(name):
                self.assertFalse(hasattr(instance, "__dict__"))

    def test_rejects_arbitrary_attribute_assignment(self):
        for name, instance in self.instances().items():
            with self.subTest(name):
                with self.assertRaises(AttributeError):
                    instance.new_attribute = 1  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    unittest.main()
