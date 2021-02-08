import argparse
import json
import os
import pickle as pkl
from tqdm import tqdm


# not use

def convert(args):
    data = []
    load_path = os.path.join('./data', '{}.json'.format(args.type))
    save_path = os.path.join('./data', '{}.pkl'.format(args.type))

    count = 0
    with open(load_path, 'r') as f:
        line = f.readline()
        while line:
            data.append(json.loads(line))
            line = f.readline()
            count += 1
            if count % 1000 == 0:
                print(count)
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'valid', 'test'], type=str, default='valid')
    args = parser.parse_args()
    convert(args)
# 287014
