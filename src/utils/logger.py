import os
import sys
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter

class Logger():
    def __init__(self, file_pth):
        self.file = open(file_pth, 'w+')
    
    def print(self, *args, **kwargs):
        print(*args, file = self.file, flush = True)
        print(*args, flush = True)
    
    def close(self):
        self.file.close()
    
    def __call__(self, *args, **kwargs):
        self.print(*args, file=self.file, flush=True)

def get_version():
    version = datetime.now().strftime("%m%d_%H:%M:%S")
    return version

def set_log(log_dir, experiment_name, version = None):
    if version is None:
        version = get_version()
    subfd = os.path.join(log_dir, experiment_name)
    if not os.path.exists(subfd):
        os.makedirs(subfd)
    log_pth = os.path.join(log_dir, experiment_name, version + '.log')
    logger = Logger(log_pth)
    return logger

def set_board(log_dir, experiment_name, version = None):
    if version is None:
        version = get_version()
    subfd = os.path.join(log_dir, experiment_name)
    if not os.path.exists(subfd):
        os.makedirs(subfd)
    board_pth = os.path.join(subfd, version)
    os.makedirs(board_pth)
    board = SummaryWriter(board_pth, flush_secs = 5)
    return board

def set_savepth(checkpoints_dir, experiment_name, version = None):
    if version is None:
        version = get_version()
    subfd = os.path.join(checkpoints_dir, experiment_name)
    if not os.path.exists(subfd):
        os.makedirs(subfd)
    model_savepth = os.path.join(subfd, version + '_model.pt')
    optimizer_savepth = os.path.join(subfd, version + '_optim.pt')
    return model_savepth, optimizer_savepth

def set_log_and_board(log_dir, experiment_name, version = None):
    if version is None:
        version = get_version()
    logger = set_log(log_dir, experiment_name, version = version)
    board = set_board(log_dir, experiment_name, version = version)
    return logger, board

