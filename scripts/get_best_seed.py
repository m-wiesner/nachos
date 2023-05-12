import glob
import argparse
import json
from queue import PriorityQueue
from pathlib import Path


def main(args):
    files = glob.glob(f'{args.dir}/seed*/scores.json')
    q = PriorityQueue(maxsize=args.top_k)  
    for i, fname in enumerate(files):
        with open(fname) as f:
            scores_f = json.load(f)
        if not q.full():
            q.put((-scores_f[-1], fname))
            continue;

        worst_item = q.get()
        if -scores_f[-1] > worst_item[0]:
            q.put((-scores_f[-1], fname))
        else:
            q.put(worst_item) 
    
    for item in q.queue:
        print('======================================')
        print(f'{item[1]} -- {item[0]}')
        print('------------------------------')
        with open(Path(item[1]).parent / 'stats', encoding='utf-8') as f:
            for l in f:
                print(l.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--top-k', type=int, default=1)
    args = parser.parse_args()
    main(args)
