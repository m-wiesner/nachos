from nachos.data.Input import TSVLoader
from nachos.constraints import build_constraints
import yaml
import pytest
from pathlib import Path


@pytest.fixture()
def connected_config():
    return yaml.safe_load(open("test/fixtures/config.yaml"))


@pytest.fixture()
def dummy_config():
    return yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))


@pytest.fixture()
def connected_eg():
    config = yaml.safe_load(open("test/fixtures/config.yaml"))
    return TSVLoader.load("test/fixtures/connected_eg.tsv", config)


@pytest.fixture()
def dummy_eg():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return TSVLoader.load("test/fixtures/dummy_constraints.tsv", config)


@pytest.fixture()
def constraints():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return build_constraints(config)


def test_constraints_n(dummy_eg, constraints):
    assert(
        constraints(dummy_eg, ({'a'}, {'b'}), n=0) == 1
        and constraints(dummy_eg, ({'a'}, {'c'}), n=0) == 1
        and constraints(dummy_eg, ({'a'}, {'b', 'c'}), n=0) == 0.5
        and constraints(dummy_eg, ({'a', 'b'}, {'c', 'd', 'e'}), n=0) == 1
        and constraints(dummy_eg, ({'a', 'b'}, {'c', 'd'}), n=1) == 5.75
    )


def test_kl():
    # Try to split on speaker and match the prompt distribution
    config = yaml.safe_load(open("test/fixtures/test_kl.yaml"))
    constraints = build_constraints(config)
    ds = TSVLoader.load("test/fixtures/dummy_kl.tsv", config)
    assert (
        constraints(ds, ({'a', 'b', 'c'}, {'d', 'e', 'f'})) != 0
        and constraints(ds, ({'a'}, {'b'})) == 0   
        and constraints(ds, ({'a', 'c', 'e'}, {'b', 'd', 'f'})) == 0
    )
