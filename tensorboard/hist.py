import argparse
import glob
import numpy as np
import sys

from tensorboardX import SummaryWriter
import torch

def main(args):
    path_prefix = args.path_prefix
    paths = glob.glob(path_prefix + "*")
    paths.sort()
    writer = SummaryWriter("runs/exp0-bert")
    for i in range(len(paths)):
        path = paths[i]
        model = torch.load(path)["model"]
        for layer_name in model:
            layer = model[layer_name]
            writer.add_histogram(layer_name, np.array(layer.reshape(-1)), i)
            print("Parsed", path, "layer shape:", layer.shape)
    writer.flush()
    

def build_arg_parser():
    parser = argparse.ArgumentParser(description='plot CDF')
    parser.add_argument('--path-prefix', type=str)
    return parser

if __name__ == '__main__':
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    main(args)

