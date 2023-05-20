import glob
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import json
from nachos.constraints.Constraints import WORST_SCORE


def plot_constraints(d: str, figpath: str):
    files = glob.glob(f'{d}/seed*/constraint_scores.json')
    with open(files[0]) as f:
        scores = json.load(f)
    constraint_names = sorted(scores[0].keys())

    for fname in files:
        with open(fname) as f:
            scores = json.load(f) 
        for cn_idx, cn in enumerate(constraint_names, 1):
            plt.subplot(len(constraint_names), 1, cn_idx)
            max_score = max(s[cn] for s in scores if s[cn] != WORST_SCORE)
            constraint_scores = [s[cn] if s[cn] < WORST_SCORE else max_score for s in scores] 
            plt.plot(constraint_scores, alpha=0.2, linewidth=0.3, color='r')
    
    for i, cn in enumerate(constraint_names, 1):
        plt.subplot(len(constraint_names), 1, i).set_title(f'{cn}')
        plt.xscale('log')
        plt.yscale('log')
    plt.savefig(figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('splitdir')
    parser.add_argument('figpath')
    args = parser.parse_args()
    plot_constraints(args.splitdir, args.figpath) 
