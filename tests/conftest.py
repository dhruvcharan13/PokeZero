import pytest

from pokezero.env.cabt import CabtEnv


@pytest.fixture
def first_decision():
    env = CabtEnv()
    d = env.reset()
    yield d
    env.close()
