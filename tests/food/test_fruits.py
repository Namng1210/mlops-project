from food.fruits import is_crisp
import pytest


# def test_is_crisp():
#     assert is_crisp("apple")
#     assert is_crisp("Apple")
#     assert not is_crisp(fruit="orange")
#     with pytest.raises(ValueError):
#         is_crisp(fruit=None)


class Fruit(object):
    def __init__(self, name):
        self.name = name


class TestFruit(object):
    @classmethod
    def setup_class(cls):
        """Set up the state for any class instance."""
        pass

    @classmethod
    def teardown_class(cls):
        """Teardown the state created in setup_class."""
        pass

    def setup_method(self):
        self.fruit = Fruit(name="apple")

    def teardown_method(self):
        """Called after every method to teardown any state."""
        del self.fruit

    def test_init(self):
        assert self.fruit.name == "apple"


@pytest.mark.parametrize(
    "fruit, crisp",
    [
        ("apple", True),
        ("Apple", True),
        ("orange", False)
    ]
)
def test_is_crisp_parametize(fruit, crisp):
    assert is_crisp(fruit=fruit) == crisp


@pytest.mark.parametrize(
    "fruit, crisp",
    [
        (None, ValueError)
    ]
)
def test_is_crisp_exception(fruit, crisp):
    with pytest.raises(ValueError):
        assert is_crisp(fruit=fruit) == crisp


@pytest.fixture
def my_fruit():
    fruit = Fruit(name="pomelo")

    yield fruit
    del fruit


# @pytest.mark.fruits
# def test_fruit(my_fruit):
#     assert my_fruit.name == "pomelo"
