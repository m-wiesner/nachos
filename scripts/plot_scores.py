import glob
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import json


def plot_scores(d: str, figpath: str):
    files = glob.glob(f'{d}/seed*/scores.json')
    for fname in files:
        with open(fname) as f:
            scores = json.load(f) 
        plt.plot(scores, alpha=0.2, linewidth=0.3, color='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('splitdir')
    parser.add_argument('figpath')
    args = parser.parse_args()
    plot_scores(args.splitdir, args.figpath) 
