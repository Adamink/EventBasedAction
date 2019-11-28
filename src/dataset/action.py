import os
import sys

import numpy as np
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from random import sample

import torch
import torch.utils.data as data

from .utils import label_mapping, slicing_windows

class Action(data.Dataset):
    def __init__(self, data_dir, mode = 'test', cameras = [2, 3], downsample_step = 10,
     events_per_frame = 7500, window_size = 100, step_size = 50, use_percentage = 100):
        self.data_dir = data_dir
        self.mode = mode
        self.cameras = cameras

        self.downsample_step = downsample_step
        self.events_per_frame = events_per_frame
        self.window_size = window_size
        self.step_size = step_size

        self.table = []

        for data_pth in glob(os.path.join(data_dir, '*')):
            feature = os.path.basename(data_pth).split('.')[0]
            subject, session, mov = [int(x) for x in feature.split('_')]
            label, is_train = label_mapping(subject, session, mov)
            if (is_train and mode=='test') or (not is_train and mode=='train'):
                continue
            data = np.load(data_pth, mmap_mode='r') #(4, num_frame, 260 ,344)
            num_frame = data.shape[1]
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
        data = np.load(data_pth, mmap_mode = 'r')[cam] #(3, num_frame, 344, 260)
        start_idx = self.step_size * window_id
        end_idx = start_idx + self.window_size
        idxes = np.arange(start_idx, end_idx, self.downsample_step)
        seq_events = np.empty([len(idxes), 1, 260, 344])
        for i, frame_idx in enumerate(idxes):
            events = np.array(data[frame_idx]) #(344, 260)
            total_events = np.sum(events)
            idx_before = frame_idx
            idx_after = frame_idx
            while(total_events < self.events_per_frame):
                idx_before -= 1
                idx_after += 1
                idx_before = max(idx_before, 0)
                idx_after = min(idx_after, num_frame - 1)
                events += data[idx_before] + data[idx_after]
                total_events = np.sum(events)
            events = events.transpose([1,0]) #(260, 344)
            # print('total events: ' + str(np.sum(events)))
            # events = np.reshape(events, events.shape + (1,)).astype(np.float32) #(260, 344, 1)
            if np.max(events) > 0.:
                events = (255 * events / np.max(events)) # normalize to [0,255]
                  
            seq_events[i, 0] = events 

        meta = {
            'file':data_pth,
            'feature':feature,
            'frame_index':frame_idx,
            'total_frame': num_frame,
            'cam':cam,
            'label':label
        }

        events = torch.from_numpy(seq_events).type(torch.FloatTensor)
        label = torch.tensor(label).type(torch.LongTensor)

        return events, label#, meta

def test_dataset():
    from utils import import_def
    definitions = import_def()
    d = Action(definitions.event_high_frame_dir, 'train')
    d.set_len(10000)
    print(d.len) # test: 13226, train: 32016
    event, label = d[0]
    print(event.size())
    print(event.type())
    print(label)
    print(label.type())
if __name__=='__main__':
    test_dataset()
