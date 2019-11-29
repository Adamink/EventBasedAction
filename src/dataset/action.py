import os
import sys
import h5py

import numpy as np
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from random import sample

import torch
import torch.utils.data as data

from utils import label_mapping, slicing_windows, parse_7500_filename
class Action(data.Dataset):
    def __init__(self, data_dir, mode = 'test', cameras = [2, 3],
     window_size = 10, step_size = 5, use_percentage = 100):
        self.data_dir = data_dir
        self.mode = mode
        self.cameras = cameras

        self.window_size = window_size
        self.step_size = step_size

        self.table = []

        for skeleton_pth in glob(os.path.join(data_dir, '*_label.h5')):
            data_pth = skeleton_pth.replace('_label.h5','.h5')
            subject, session, mov = parse_7500_filename(os.path.basename(data_pth))
            feature = '_'.join([str(x) for x in [subject, session, mov]])
            label, is_train = label_mapping(subject, session, mov)
            if (is_train and mode=='test') or (not is_train and mode=='train'):
                continue
            f = h5py.File(skeleton_pth,'r')
            skeleton = np.array(f['XYZ'], dtype=np.float32) #(num_frame, 3, 13)
            num_frame = len(skeleton)
            window_num, _ = slicing_windows(num_frame, self.window_size, self.step_size)
            for cam in cameras:
                for i in range(0, window_num):
                    self.table.append((data_pth, feature, label, cam, i, num_frame))               

        self.label = [t[2] for t in self.table]
        self.len = len(self.table)
        dest_len = (int)(self.len * (use_percentage / 100.0))
        self.set_len(dest_len)

    def set_len(self, l):
        sample_index = sample(list(range(self.len)), l)
        self.label = [self.label[_] for _ in sample_index]
        self.table = [self.table[_] for _ in sample_index]
        self.len = l

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        data_pth, feature, label, cam, window_id, num_frame = self.table[index]
        f = h5py.File(data_pth,'r')

        start_idx = self.step_size * window_id
        end_idx = start_idx + self.window_size
        events = np.array(f['DVS'][start_idx:end_idx, :,:,cam], dtype=np.uint8) #(10, 260, 344)
        for i in range(self.window_size):
            s = np.max(events[i])
            if s > 0.:
                events[i] = (255 * events[i] / s)
        events = np.expand_dims(events, 1) #(10, 1, 260, 344)
        events = torch.from_numpy(events).type(torch.FloatTensor)
        label = torch.tensor(label).type(torch.LongTensor)
        return events, label

def test_dataset():
    from utils import import_def
    definitions = import_def()
    d = Action(definitions.action_dir, 'test')
    print(d.len)
    event, label = d[0]
    print(event.size())
    print(event.type())
    print(label)
    print(label.type())
if __name__=='__main__':
    test_dataset()
