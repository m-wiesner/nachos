from nachos.data.Input import TSVLoader
from nachos.constraints import build_constraints
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.splitters import build_splitter
from nachos.data.Data import FactoredSplit, collapse_factored_split
import yaml
import pytest
from pathlib import Path


@pytest.fixture()
def connected_config():
    return yaml.safe_load(open("test/fixtures/connected_test_constraints.yaml"))


@pytest.fixture()
def dummy_config():
    return yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))


@pytest.fixture()
def connected_eg():
    config = yaml.safe_load(open("test/fixtures/connected_test_constraints.yaml"))
    return TSVLoader.load("test/fixtures/connected_fraction_constraints.tsv", config)


@pytest.fixture()
def dummy_eg():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return TSVLoader.load("test/fixtures/dummy_fraction_constraints.tsv", config)


@pytest.fixture()
def filename():
    return "test/fixtures/graph.gml"


@pytest.fixture()
def dummy_filename():
    return "test/fixtures/dummy.gml"


@pytest.fixture()
def constraints():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return build_constraints(config)


@pytest.fixture()
def sim_fns():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return build_sims(config)


@pytest.fixture()
def random_splitter():
    config = yaml.safe_load(open("test/fixtures/dummy_test_constraints.yaml"))
    return build_splitter(config)


#def test_random_splitter(dummy_eg, random_splitter):
#    best_split, scores = random_splitter(dummy_eg)
#    for i in range(1, len(scores)):
#        assert(scores[i] < scores[i-1])


def test_splitter_connected(connected_eg):
    config = yaml.safe_load(open("test/fixtures/connected_test_constraints.yaml"))
    splitter = build_splitter(config)

    split, scores = splitter(connected_eg)
    for i in range(1, len(scores)):
        assert(scores[i] < scores[i-1])
    for idx_s, s in enumerate(split):
        constraint_stats = splitter.constraint_fn.stats(connected_eg, s)
        print(constraint_stats)
    test_sets = connected_eg.make_overlapping_test_sets(split)
    for idx_s, s in enumerate(test_sets):
        constraint_stats = splitter.constraint_fn.stats(connected_eg, test_sets[s])
        print(constraint_stats)
    for i in range(len(test_sets)):
        for j in range(len(test_sets)):
            if i != j:
                overlap_stats = connected_eg.overlap_stats(test_sets[i], test_sets[j])
                print(f'{j} overlap with {i}')
                print(overlap_stats)   
