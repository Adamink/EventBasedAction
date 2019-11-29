import os
import yaml
import numpy as np

import torch
import torch.nn as nn

from utils.parser import set_parser 
from utils.importer import import_class
from utils.logger import get_version, set_log_and_board, set_savepth
from utils.trainer import train

def main(gpus, cfg):
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

    # prepare logger, board, savepth
    args.version = get_version()
    logger, board = set_log_and_board(args.results_dir, args.experiment, args.version)
    args.model_savepth, args.optim_savepth = set_savepth(args.checkpoints_dir, args.experiment, args.version)
    logger(args)

    # prepare dataset
    Dataset = import_class(args.dataset)
    train_dataset = Dataset(**args.train_dataset_args)
    test_dataset = Dataset(**args.test_dataset_args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size * len(gpus),
        shuffle=True,
        num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size * len(gpus),
        shuffle=True,
        num_workers=args.workers)
        
    # prepare model
    Model = import_class(args.model)
    model = Model(**args.model_args)
    model = nn.DataParallel(model)
    model.cuda()

    # prepare loss, optimizer, scheduler
    criterion = import_class(args.criterion)
    metric = import_class(args.metric)

    optimizer = import_class(args.optimizer)(model.parameters(), lr = args.lr)  
    scheduler = import_class(args.scheduler)(optimizer, **args.scheduler_args)
    trainer = import_class(args.trainer)
    best_metric, best_epoch = trainer(
     args, model, train_dataloader, test_dataloader, optimizer, 
     scheduler, criterion, metric, logger, board, args.test_interval, args.max_metric)
    logger("Best Epoch {:3d} test_metric: {:.4f}".format(best_epoch, best_metric))
    logger.close()
    board.close()
