import json
import glob
from matplotlib import pyplot as plt
import argparse
from pathlib import Path


def plot_seed_indices(d: str, figpath: str):
    min_score = 999999999.0
    files = glob.glob(f'{d}/seed*/score')
    best_seed = 0 
    for f in files:
        seed = int(str(Path(f).parent.stem).split('d')[1])
        with open(f, 'r') as f:
            score = float(f.readline())
            if score < min_score:
                min_score = score
                best_seed = seed
  
    files = glob.glob(f'{d}/seed*/indices.json')
    all_indices = [] 
    for f in files:
        all_indices.append(json.load(open(f)))

    best_file_idx = files.index(f'{d}/seed{best_seed}/indices.json') 
    min_max_vals = [
        (
            min(index[-1][factor_idx] for index in all_indices),
            max(index[-1][factor_idx] for index in all_indices),
        )
        for factor_idx in range(len(all_indices[0][0]))
    ]
    for i, indices in enumerate(all_indices):
        num_indices = len(indices[0])
        for factor_idx in range(1, num_indices+1):
            plt.subplot(num_indices, 1, factor_idx)
            plt.step(
                range(len(indices)),
                [j[factor_idx-1] for j in indices],
                alpha=0.2, linewidth=0.3, color='r'
            )
    for factor_idx in range(1, num_indices+1):
        plt.subplot(num_indices, 1, factor_idx)
        plt.step(
            range(len(all_indices[best_file_idx])),
            [j[factor_idx-1] for j in all_indices[best_file_idx]],
            alpha=0.7, linewidth=1.3, color='lime',
        )
        plt.xscale('log')
        plt.yscale('log')
    #plt.supxlabel('Iteration #')
    #plt.supylabel('Powerset Index')
    plt.savefig(figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('splitdir')
    parser.add_argument('figpath')
    args = parser.parse_args()
    plot_seed_indices(args.splitdir, args.figpath)
