import os
import yaml
import numpy as np

import torch
import torch.nn as nn

from utils.parser import set_parser 
from utils.importer import import_class
from utils.logger import get_version, set_log_and_board, set_savepth
from utils.trainer import test_long_one_epoch

def gen(gpus, cfg):
    # parse config
    with open(cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    parser = set_parser()
    args = parser.parse_known_args()[0]
    args.__dict__.update(cfg)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set gpu
    args.gpus = gpus
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(args.gpus)

    # prepare dataset
    Dataset = import_class(args.dataset)
    train_dataset = Dataset(**args.train_dataset_args)
    test_dataset = Dataset(**args.test_dataset_args)
      
    # prepare model
    Model = import_class(args.model)
    model = Model(**args.model_args)
    model = nn.DataParallel(model)
    model.cuda()

    acc = test_long_one_epoch(model, test_dataset, len(gpus), args.arch_option)
    print(acc)

if __name__=='__main__':
    gpus = ['0','1','2']
    cfg = '../configs/action_long_feat.yaml'
    gen(gpus, cfg)