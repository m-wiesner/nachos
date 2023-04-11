from nachos.data.Input import TSVLoader, LhotseLoader
from nachos.constraints import build_constraints
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.splitters import build_splitter
from nachos.data.Data import new_from_components
import yaml
import argparse
from pathlib import Path


def main(args):
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    config = yaml.safe_load(open(args.config))
    splitter = build_splitter(config)
    if config['format'] == 'tsv':
        data_eg = TSVLoader.load(args.metadata[0], config)
    elif config['format'] == 'lhotse':
        data_eg = LhotseLoader.load(args.metadata, config)
  
    data_eg.make_graph(splitter.sim_fn) 
    # If the graph is disconnected, reconstruct the dataset using the
    # components as the factors, keep the original constraints, and use the
    # components themselves as the records. Keep a map from the components
    # back to the original records
    components = None
    if data_eg.is_disconnected():
        data_eg, components = new_from_components(
            data_eg, splitter.sim_fn
        )
    
    # Get the splits
    partition = {}
    split, scores = splitter(data_eg)
    test_sets = data_eg.make_overlapping_test_sets(split)
    for s_idx, s in enumerate(split):
        for s_item in s:
            if components is not None:
                for i in components[s_item]:
                    partition[i] = s_idx
            else:
                partition[s_item] = s_idx   
    
    for s_idx, s in enumerate(test_sets, 2):
        if len(test_sets[s]) > 0:
            for s_item in s:
                if components is not None:
                    for i in components[s_item]:
                        partition[i] = s_idx
                else:
                    partition[s_item] = s_idx

    with open(odir / 'partition', 'w') as f:
        for i, v in partition.items():
            print(f'{i} {v}', file=f)

    # Get split stats
    with open(odir / 'stats', 'w') as f:
        for idx_s, s in enumerate(split):
            constraint_stats = splitter.constraint_fn.stats(data_eg, s)
            print(f'Split {idx_s}: {constraint_stats}', file=f)
        for idx_s, s in enumerate(test_sets):
            if len(test_sets[s]) > 0:
                constraint_stats = splitter.constraint_fn.stats(data_eg, test_sets[s])
                print(f'Split {idx_s+2}: {constraint_stats}', file=f)
            else:
                print(f'Split {idx_s+2}: length = 0', file=f)
        
        # Get split overlaps
        sets = {}
        sets[0] = split[0]
        sets[1] = split[1]
        for i in range(2, len(test_sets)+2):
            sets[i] = test_sets[i-2]
        for i in range(0, len(sets)):
            for j in range(0, len(sets)):
                if i != j and len(sets[i]) > 0 and len(sets[j]) > 0:
                    overlap_stats = data_eg.overlap_stats(sets[i], sets[j])
                    print(f'{j} overlap with {i}: {overlap_stats}', file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('odir', type=str, help='the path to the output '
        'directory where created files will be stored.'
    )
    parser.add_argument('config', type=str, help='path to the yaml config file'
        'defining the splitting hyperparameters.'
    )
    parser.add_argument('metadata', type=str, nargs='+',
        help='path(s) to the metadata file(s), .tsv or lhotse manifests, that '
        'define(s) the data elements to be split.'
    )
    
    args = parser.parse_args() 
    main(args)
