import os
import yaml
import numpy as np

import torch
import torch.nn as nn

from utils.parser import set_parser 
from utils.importer import import_class
from utils.logger import get_version, set_log_and_board, set_savepth
from utils.trainer import stat_one_epoch

import definitions

from utils.draw import plot_confusion_matrix, plot_histogram
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

    y_pred, y_true = stat_one_epoch(model, test_dataloader)
    np.save(os.path.join(definitions.stat_dir, 'pred.npy'), y_pred)
    np.save(os.path.join(definitions.stat_dir, 'true.npy'), y_true)

def draw_confusion(y_pred_path, y_true_path, class_title_path, figure_path):
    with open(class_title_path, 'r') as f:
        class_names = [_.split('-')[1].strip() for _ in f.readlines()]
    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path) # [0, 32]

    ax = plot_confusion_matrix(y_true, y_pred, class_names, title = 'Confusion Matrix of Action Recognition on DHP19 Dataset', normalize = True)
    plt.savefig(figure_path, dpi = 100)
    plt.close()

def draw_histogram(y_pred_path, y_true_path, class_title_path, figure_path):
    with open(class_title_path, 'r') as f:
        class_names = [_.split('-')[1].strip() for _ in f.readlines()]
    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path) # [0, 32]
    total = np.zeros((33,))
    correct = np.zeros((33,))
    for i in range(len(y_pred)):
        total[y_true[i]] += 1.0
        if y_true[i]==y_pred[i]:
            correct[y_true[i]] += 1.0
    
    acc = correct / total
    print(acc)
    ax = plot_histogram(acc, class_names, 'Per-class Recognition Accuracy on DHP19 Dataset')
    if 'eps' in figure_path:
        plt.savefig(figure_path, format='eps')
    else:
        plt.savefig(figure_path, dpi=100)
    plt.close()

if __name__=='__main__':
    # gpus = ['0','1','2','6']
    # cfg = '../configs/action_mean.yaml'
    # gen(gpus, cfg)
    draw_confusion(os.path.join(definitions.stat_dir, 'pred.npy'), os.path.join(definitions.stat_dir, 'true.npy'), 
    os.path.join(definitions.config_dir, 'DHP19_classes.txt'), os.path.join(definitions.figures_dir, 'stat/confusion_matrix_1.eps'))
    draw_confusion(os.path.join(definitions.stat_dir, 'pred.npy'), os.path.join(definitions.stat_dir, 'true.npy'), 
    os.path.join(definitions.config_dir, 'DHP19_classes.txt'), os.path.join(definitions.figures_dir, 'stat/confusion_matrix_1.png'))
    draw_histogram(os.path.join(definitions.stat_dir, 'pred.npy'), os.path.join(definitions.stat_dir, 'true.npy'), 
     os.path.join(definitions.config_dir, 'DHP19_classes.txt'), os.path.join(definitions.figures_dir, 'stat/histogram_1.eps'))
    draw_histogram(os.path.join(definitions.stat_dir, 'pred.npy'), os.path.join(definitions.stat_dir, 'true.npy'), 
     os.path.join(definitions.config_dir, 'DHP19_classes.txt'), os.path.join(definitions.figures_dir, 'stat/histogram_1.png'))