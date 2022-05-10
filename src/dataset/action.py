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
if __name__=='__main__':
    from utils import label_mapping, slicing_windows, parse_7500_filename, visualize_event
else:
    from .utils import label_mapping, slicing_windows, parse_7500_filename, visualize_event
class Action(data.Dataset):
    def __init__(self, data_dir, mode = 'test', cameras = [2, 3],
     window_size = 10, step_size = 5, use_percentage = 100, experiment = 'action'):
        self.data_dir = data_dir
        self.mode = mode
        self.cameras = cameras

        self.window_size = window_size
        self.step_size = step_size
        self.experiment = experiment
        self.table = []

        for skeleton_pth in glob(os.path.join(data_dir, '*_label.h5')):
            data_pth = skeleton_pth.replace('_label.h5','.h5')
            subject, session, mov = parse_7500_filename(os.path.basename(data_pth))
            feature = '_'.join([str(x) for x in [subject, session, mov]])
            label, is_train = label_mapping(subject, session, mov)
            if (is_train and mode=='test') or (not is_train and mode=='train'):
                continue
            # f = h5py.File(skeleton_pth,'r')
            # skeleton = np.array(f['XYZ'], dtype=np.float32) #(num_frame, 3, 13)
            # num_frame = len(skeleton)
            f = h5py.File(data_pth, 'r')
            num_frame = f['DVS'].shape[0]
            if experiment=='action':
                window_num, _ = slicing_windows(num_frame, self.window_size, self.step_size)
                for cam in cameras:
                    for i in range(0, window_num):
                        self.table.append((data_pth, feature, label, cam, i, num_frame))
            elif experiment=='pose':
                for cam in cameras:
                    for i in range(0, num_frame):
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
    
    def get_event(self, index):
        data_pth, feature, label, cam, window_id, num_frame = self.table[index]
        # print(data_pth)
        f = h5py.File(data_pth,'r')
        if self.experiment=='action':
            start_idx = self.step_size * window_id
            end_idx = start_idx + self.window_size
            events = np.array(f['DVS'][start_idx:end_idx, :,:,cam], dtype=np.uint8) #(10, 260, 344)
        elif self.experiment=='pose':
            events = np.array(f['DVS'][window_id, :, :, cam], dtype = np.uint8) #（260, 344)
        return events, label
    def __getitem__(self, index):
        events, label = self.get_event(index)
        events = events.astype(np.float32)
        if self.experiment=='action':
            for i in range(self.window_size):
                s = np.max(events[i])
                if s > 0.:
                    events[i] = (255 * events[i] / s)
            events = np.expand_dims(events, 1) #(10, 1, 260, 344)
        elif self.experiment=='pose':
            s = np.max(events)
            if s > 0.:
                events = (255 * events / s)
            events = np.expand_dims(events, 0)
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

def test_action_and_pose7500():
    from utils import import_def
    definitions = import_def()
    d = Action(definitions.pose_7500_dir, 'test')
    s = 0
    for j in range(100, 150):
        events, label = d.get_event(j)
        s += (events.sum())
    print(s)
    d1 = Action(definitions.action_dir, 'test')
    s1 = 0
    for j in range(100, 150):
        events, label = d1.get_event(j)
        s1 += (events.sum())
    print(s1)

def test_pose():
    from utils import import_def
    definitions = import_def()
    d = Action(definitions.action_dir, 'train', experiment='pose')
    print(len(d))

if __name__=='__main__':
    test_pose()
