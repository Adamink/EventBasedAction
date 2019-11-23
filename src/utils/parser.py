import argparse
import os
import sys

def import_def():
    sys.path.append('../')
    import definitions
    sys.path.remove('../')
    return definitions

def set_parser():
    definitions = import_def()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--gpus', nargs='+', type=str, default=['0','1','2','3'])
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--epochs', type=int, default=150, help='epochs')
    parser.add_argument('--version', type=str, default='noversion')
    parser.add_argument('--dataset_dir', type=str, default='', help="dataset path")
    parser.add_argument('--results_dir', type=str, default=definitions.results_dir)
    parser.add_argument('--checkpoints_dir', type=str, default=definitions.checkpoints_dir)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    return parser 

if __name__ == '__main__':
    print(sys.path)
    a = set_parser()
    print(sys.path)