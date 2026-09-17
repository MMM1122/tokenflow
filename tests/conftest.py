import pytest

from tokenflow import Optimizer


@pytest.fixture(scope="session")
def engine():
    return Optimizer()
