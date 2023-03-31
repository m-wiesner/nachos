from nachos.data.Input import TSVLoader, LhotseLoader
from nachos.similarity_functions import build_similarity_functions as build_sims
import yaml
import pytest
from pathlib import Path
import random


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
def connected_eg_lhotse():
    config = yaml.safe_load(open("test/fixtures/config.yaml"))
    sups_train = "test/fixtures/supervisions_train_intv.jsonl.gz"
    sups_dev = "test/fixtures/supervisions_dev_a.jsonl.gz"

    ds = LhotseLoader.load([sups_train, sups_dev], config,)
    return ds

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
    config = yaml.safe_load(open("test/fixtures/config.yaml"))
    return build_sims(config)


@pytest.fixture()
def random_idxs():
    idxs = []
    for i in range(20):
        idxs.append(random.randint(0, 400))
    return idxs


@pytest.fixture()
def random_slices():
    slices = []
    for i in range(20):
        start = random.randint(0, 100)
        stop = random.randint(101, 200)
        step = random.randint(1, 4)
        slices.append(slice(start, stop, step))
    return slices


def test_lhotse_make_graph(connected_eg_lhotse, sim_fns):
    connected_eg_lhotse.make_graph(sim_fns)  
    assert connected_eg_lhotse.graph is not None
    import networkx as nx
    G_ = nx.compose_all(
        connected_eg_lhotse.graphs[n][f] 
        for n in connected_eg_lhotse.graphs 
            for f in connected_eg_lhotse.graphs[n]
    )  
    assert G_.edges == connected_eg_lhotse.graph.edges


def test_make_graph(connected_eg, sim_fns):
    connected_eg.make_graph(sim_fns)  
    assert connected_eg.graph is not None
    import networkx as nx
    G_ = nx.compose_all(
        connected_eg.graphs[n][f] 
        for n in connected_eg.graphs 
            for f in connected_eg.graphs[n]
    )  
    assert G_.edges == connected_eg.graph.edges

def test_export_graph(connected_eg, sim_fns, filename):
    connected_eg.make_graph(sim_fns)
    connected_eg.export_graph(filename)
    assert Path(filename).is_file() 


def test_dataset_getitem_int(connected_eg, random_idxs):
    new_sets = [] 
    for i in random_idxs:
        new_sets.append(connected_eg[i])
    assert all(
        ns.data[0].id == connected_eg[random_idxs[i]].data[0].id
        for i, ns in enumerate(new_sets)
    )


def test_dataset_getitem_slice(connected_eg, random_slices):
    new_sets = []
    for i in random_slices:
        new_sets.append(connected_eg[i])
    assert all(
        len(ns.data) == len(range(*sl.indices(len(connected_eg)))) 
        for ns, sl in zip(new_sets, random_slices)
    )  
