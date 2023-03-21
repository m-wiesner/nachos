from nachos.data.Input import TSVLoader
from nachos.similarity_functions import build_similarity_functions as build_sims
import yaml
import pytest
from pathlib import Path


@pytest.fixture()
def connected_config():
    return yaml.safe_load(open("test/fixtures/config.yaml"))


@pytest.fixture()
def dummy_config():
    return yaml.safe_load(open("test/fixtures/dummy.yaml"))


@pytest.fixture()
def connected_eg():
    config = yaml.safe_load(open("test/fixtures/config.yaml"))
    return TSVLoader.load("test/fixtures/connected_eg.tsv", config)


@pytest.fixture()
def dummy_eg():
    config = yaml.safe_load(open("test/fixtures/dummy.yaml"))
    return TSVLoader.load("test/fixtures/dummy.tsv", config)


@pytest.fixture()
def filename():
    return "test/fixtures/graph.gml"


@pytest.fixture()
def dummy_filename():
    return "test/fixtures/dummy.gml"


@pytest.fixture()
def sim_fns():
    config = yaml.safe_load(open("test/fixtures/dummy.yaml"))
    return build_sims(config)


@pytest.fixture()
def dummy_set():
    config = yaml.safe_load(open("test/fixtures/dummy.yaml"))
    ds = TSVLoader.load("test/fixtures/dummy.tsv", config)
    return ds[0::3]


def test_similarity_functions_point_to_point(dummy_eg, sim_fns):
    assert (
        sim_fns(dummy_eg[0], dummy_eg[1]) == 1
        and sim_fns(dummy_eg[0], dummy_eg[2]) == 0
        and sim_fns(dummy_eg[0], dummy_eg[3]) == 0
        and sim_fns(dummy_eg[1], dummy_eg[0]) == 1
        and sim_fns(dummy_eg[1], dummy_eg[2]) == 1
        and sim_fns(dummy_eg[1], dummy_eg[3]) == 0
        and sim_fns(dummy_eg[3], dummy_eg[5]) == 1
        and sim_fns(dummy_eg[3], dummy_eg[6]) == 1
        and sim_fns(dummy_eg[5], dummy_eg[6]) == 2
     )


def test_similarity_functions_point_to_set_n(dummy_eg, sim_fns):
    assert(
        sim_fns(dummy_eg[0], dummy_eg[1:5], 0) == 1
        and sim_fns(dummy_eg[0], dummy_eg[1:5], 1) == 0
        and sim_fns(dummy_eg[0], dummy_eg[2:5], 0) == 0
        and sim_fns(dummy_eg[5], dummy_eg[0:5], 1) == 1
        and sim_fns(dummy_eg[5], dummy_eg[0:5], 0) == 0
    )


def test_similarity_functions_point_to_set(dummy_eg, sim_fns):
    assert(
        sim_fns(dummy_eg[0], dummy_eg[1:5]) == 1
        and sim_fns(dummy_eg[0], dummy_eg[2:5]) == 0
        and sim_fns(dummy_eg[5], dummy_eg[0:5]) == 1
        and sim_fns(dummy_eg[6], dummy_eg[4:6]) == 2
    )
