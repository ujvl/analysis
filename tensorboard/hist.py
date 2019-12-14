import argparse
import glob
import logging
import numpy as np
import os
import sys

from tensorboardX import SummaryWriter
import torch

logging.basicConfig(level=logging.INFO)


def main(args):
    path_prefix = args.path_prefix
    paths = glob.glob(path_prefix + "*")
    paths.sort()
    writer = SummaryWriter("runs/exp0-bert")
    for i in range(len(paths)):
        path = paths[i]
        basename = os.path.basename(path)
        model = torch.load(path)["model"]
        for layer_name in model:
            layer = model[layer_name]
            writer.add_histogram(layer_name, np.array(layer.reshape(-1)), i)
            logging.info("Parsed %s (%s) with shape %s", layer_name, basename, layer.shape)
    logging.info("Ordering: %s", [os.path.basename(path) for path in paths])
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
    logging.info(args)
    main(args)

