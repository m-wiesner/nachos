from nachos.data.Input import TSVLoader, LhotseLoader
from nachos.constraints import build_constraints
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.splitters import build_splitter
from nachos.data.Data import new_from_components
import yaml
import argparse
from pathlib import Path
import json
from matplotlib import pyplot as plt


def main(args):
    if args.partition_and_split is not None:
        included_records = set()
        partition_file, split = args.partition_and_split.split(':')
        split = int(split)
        with open(partition_file, 'r') as f:
            for l in f:
                record_name, split_num = l.strip().split()
                if int(split_num) == split:
                    included_records.add(record_name)
    
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    config = yaml.safe_load(open(args.config))
    if config['format'] == 'tsv':
        data_eg = TSVLoader.load(args.metadata[0], config)
    elif config['format'] == 'lhotse':
        data_eg = LhotseLoader.load(args.metadata, config)

    data_eg.make_constraint_inverted_index()
    # For KL divergence, it is probably better to set the vocabulary. It is not
    # necessary, but it makes things easier if comparing across multiple splits.
    for i, c in enumerate(config['constraints']):
        if c['name'] in ('kl', 'kl_tuple') and 'vocab' not in c['values']:
            c['values']['vocab'] = sorted(data_eg.constraint_values[i])
    
    # This line is where we subset the dataset to include only those records in
    # the requested split of the partition file (from a previous split). This
    # is used when you want to recursively split dataset.
    if args.partition_and_split is not None:
        assert len(included_records) >= 2, "Length of the requested partition is too small (<2)"
        new_data = []
        # We will have to renormalize the weights if we are only using a subset
        # of the data. First we get the total weight of the selected items and
        # then we will renormalize each record's weight by the new normalizer.
        total_weight = sum(
            sum(data_eg.data[data_eg.id_to_idx[r]].weights) for r in included_records
        )
        for r in included_records:
            idx = data_eg.id_to_idx[r]
            new_data_item = data_eg.data[idx].copy()
            new_data_item.weights = [
                w/total_weight for w in new_data_item.weights
            ]
            new_data.append(new_data_item)
        data_eg = data_eg.subset_from_data(new_data)
    if args.seed is not None:
        config['seed'] = args.seed
         
    config['partition_and_split'] = (partition_file, split) if args.partition_and_split is not None else None
    data_eg.make_constraint_inverted_index()
    splitter = build_splitter(config)
    data_eg.make_graph(splitter.sim_fn) 
    
    # If the graph is disconnected, reconstruct the dataset using the
    # components as the factors, keep the original constraints, and use the
    # components themselves as the records. Keep a map from the components
    # back to the original records
    components = None
    config['use_components'] = False
    if args.use_components:
        if data_eg.is_disconnected():
            data_eg, components = new_from_components(
                data_eg, splitter.sim_fn
            )
            config['use_components'] = True
    
    # Save the configurations used for splitting
    with open(odir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Get the splits
    partition = {}
    
    # This is where most of the work happens
    out = splitter(data_eg)
    if config['splitter'] == 'vns':
        split, constraint_scores, indices = out
        scores = [c['total'] for c in constraint_scores]
        with open(odir / 'indices.json', 'w') as f:
            json.dump(indices, f, indent=4)

        # Plot the powerset indices
        for f in range(len(indices[0])):
            indices_over_time_bin = [
                [int(format(indices[i][f], f'0{len(data_eg.graphs[f])}b')[j]) for i in range(len(indices))]
                for j in range(len(data_eg.graphs[f]))
            ]

            indices_over_time = [indices[i][f] for i in range(len(indices))]
            for i, e in enumerate(indices_over_time_bin):
                plt.step(range(len(e)), [j + i*2 for j in e]) 
            plt.xscale('log')
            plt.savefig(f'{str(odir)}/powerset_{f}_bin.png')
            plt.clf()

            plt.step(range(len(indices_over_time)), indices_over_time)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig(f'{str(odir)}/powerset_{f}.png')
            plt.clf()

        plt.plot(scores)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f'{str(odir)}/scores.png')
    else:
        split, constraint_scores, = out
        scores = [c['total'] for c in constraint_scores]

    # Print final score out to file 
    with open(odir / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=4)
    
    # Also dump all the final constraint scores to a file
    with open(odir / 'constraint_scores.json', 'w') as f:
        json.dump(constraint_scores, f, indent=4)

    # The rest is just printing out the partitions and associated statistics 
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
            for s_item in test_sets[s]:
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
    parser.add_argument('--seed', type=int, help='The random seed.') 
    parser.add_argument('--use-components', action='store_true')
    parser.add_argument('--partition-and-split', type=str, help='specify this '
        'when splitting on a subset specified by the output of a previous '
        "split. The format is 'filename:0' to split on the split labeled 0 in"
        " filename, which stores a previous splits output.", default=None,
    )
    
    args = parser.parse_args() 
    main(args)
