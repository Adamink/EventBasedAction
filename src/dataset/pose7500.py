from __future__ import absolute_import

import os
import sys
import h5py
import cv2
import PIL

from tqdm import tqdm 
from glob import glob
from scipy.io import loadmat, savemat
from PIL import Image
from matplotlib import cm
from random import sample

import numpy as np
import scipy.io as scio

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

if __name__=='__main__':
    from utils import *
else:
    from .utils import *

class Pose7500(data.Dataset):
    def __init__(self, data_dir, p_mat_dir, input_size = (260, 344), mode = 'test', 
     cameras = [2,3], experiment = 'dhpcnn'):
        self.data_dir = data_dir
        self.p_mat_dir = p_mat_dir
        self.cameras = cameras
        self.experiment = experiment

        self.load_p_mat_cam()
        self.init_transform()
        self.image_h, self.image_w = input_size
        self.num_joints = 13
        train_table = []
        test_table = []
        all_table = []
        nocam_table = []
        action_table = []
        which_table = {'train':train_table, 'test':test_table, 'all': all_table, 'nocam': nocam_table}

        for skeleton_pth in glob(os.path.join(data_dir, '*_7500events_label.h5')):
            event_pth = skeleton_pth.replace('_label.h5', '.h5')
            skeleton_basename = os.path.basename(skeleton_pth)
            subject = (int)(skeleton_basename.split('_')[0][1:])
            session = (int)(skeleton_basename.split('_')[1][7:])
            mov = (int)(skeleton_basename.split('_')[2][3:])
            label, is_train = label_mapping(subject, session, mov)
            f = h5py.File(skeleton_pth, 'r')
            skeleton = np.array(f['XYZ'], dtype=np.float32) #(num_frame, 3, 13)
            num_frame = len(skeleton)
            for i in range(num_frame):
                feature ='_'.join([str(x) for x in [subject, session, mov]])
                which_table['nocam'].append((skeleton_basename, feature, label, -1, i))
                for cam in self.cameras:
                    feature = '_'.join([str(x) for x in [subject, session, mov, cam]])
                    which_table['all'].append((skeleton_basename, feature, label, cam, i))
                    if is_train:
                        which_table['train'].append((skeleton_basename, feature, label, cam, i))
                    else:
                        which_table['test'].append((skeleton_basename, feature, label, cam, i))
        if experiment=='action':
            self.table = action_table
        else:
            self.table = which_table[mode]
        self.label = [t[1] for t in which_table[mode]]
        self.len = len(self.table)

    def load_p_mat_cam(self):
        P1 = np.load(os.path.join(self.p_mat_dir,'P1.npy'))
        P2 = np.load(os.path.join(self.p_mat_dir,'P2.npy'))
        P3 = np.load(os.path.join(self.p_mat_dir,'P3.npy'))
        P4 = np.load(os.path.join(self.p_mat_dir,'P4.npy'))
        self.P_mat_cam = np.stack([P4,P1,P3,P2])
        self.p_mat_cam = self.P_mat_cam
        self.camera_positions = np.load(os.path.join(self.p_mat_dir, 'camera_positions.npy'))
        
    def gen_2dpose(self, skeleton, cam = 3):
        # input:(3,13)
        # output: (image_h, image_w, num_joints)
        p_mat_cam = self.p_mat_cam[cam]
        # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
        vicon_xyz_homog = np.concatenate([skeleton, np.ones([1,13])], axis=0)
        coord_pix_all_cam2_homog = np.matmul(p_mat_cam, vicon_xyz_homog)
        coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog/coord_pix_all_cam2_homog[-1]
        # print(coord_pix_all_cam2_homog)
        # print(coord_pix_all_cam2_homog_norm)
        u = coord_pix_all_cam2_homog_norm[0]
        v = self.image_h - coord_pix_all_cam2_homog_norm[1] # flip v coordinate to match the image direction

        u = np.nan_to_num(u)
        v = np.nan_to_num(v)
    
        # pixel coordinates
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        u = np.clip(u, 0, self.image_w - 1)
        v = np.clip(v, 0, self.image_h - 1)

        pose_weight = np.ones(shape = u.shape, dtype=np.float32)
        pose_weight[u==0] = 0.
        pose_weight[v==0] = 0.
        pose_weight[u==self.image_w - 1] = 0.
        pose_weight[v==self.image_h - 1] = 0.

        pose = np.stack((v, u))
        return pose, pose_weight

    def gen_heatmap(self, pose, decay_maps_flag = True):
        # initialize the heatmaps
        return gen_heatmap(pose)

    def get_raw_event(self, index):
        skeleton_basename, feature, label, cam, frame_index = self.table[index]
        skeleton_pth = os.path.join(self.data_dir, skeleton_basename)
        event_pth = skeleton_pth.replace('_label.h5', '.h5')
        f = h5py.File(skeleton_pth,'r')
        skeleton = np.array(f['XYZ'])
        num_frame = len(skeleton)
        f = h5py.File(event_pth,'r')
        event = np.array(f['DVS'][frame_index][:,:,cam], dtype=np.uint8) #(260, 346)

        event = np.nan_to_num(event)
        return event, label
    def get_raw_item(self, index):
        skeleton_basename, feature, label, cam, frame_index = self.table[index]
        skeleton_pth = os.path.join(self.data_dir, skeleton_basename)
        event_pth = skeleton_pth.replace('_label.h5', '.h5')

        f = h5py.File(skeleton_pth,'r')
        skeleton = np.array(f['XYZ'])
        num_frame = len(skeleton)
        skeleton = np.array(skeleton[frame_index], dtype=np.float32) #(3, 13)
        pose, pose_weight = self.gen_2dpose(skeleton, cam) #(2,13), (13,)
        pose_weight = np.reshape(pose_weight, pose_weight.shape + (1,))
        heatmap = self.gen_heatmap(pose) #(260, 346, 13)
        
        f = h5py.File(event_pth,'r')
        event = np.array(f['DVS'][frame_index][:,:,cam], dtype=np.uint8) #(260, 346)

        event = np.nan_to_num(event)
        heatmap = np.nan_to_num(heatmap)
        skeleton = np.nan_to_num(skeleton)
        pose = np.nan_to_num(pose)
        pose_weight = np.nan_to_num(pose_weight)
        
        meta = {
            'file':skeleton_basename,
            'feature':feature,
            'frame_index':frame_index,
            'total_frame': num_frame,
            'cam':cam
        }
        return event, heatmap, skeleton, pose, pose_weight, label, meta
    def init_transform(self):
        self.transform = transforms.Compose([
        transforms.Resize([192, 256]),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    def __getitem__(self, index):
        if self.experiment=='dhpcnn':
            return self.dhpcnn_getitem(index)
        elif self.experiment=='mm_cnn':
            return self.mm_cnn_getitem(index)
        elif self.experiment=='action':
            return self.action_getitem(index)
        elif self.experiment=='visualize':
            return self.visualize_getitem(index)
    def visualize_getitem(self, index):
        event, heatmap, skeleton, pose, pose_weight, label, meta = self.get_raw_item(index)
        event = event.astype('float32')
        return event, heatmap, skeleton, pose, label, meta
    def dhpcnn_getitem(self, index):
        event, heatmap, skeleton, pose, pose_weight, label, meta = self.get_raw_item(index)
        event = event.astype('float32')
        if np.max(event) > 0.:
            event = (255 * event / np.max(event)) # normalize to [0,255]
        event = event.reshape((1,) + event.shape) # (1, 260, 344)
        event = torch.from_numpy(event).type(torch.FloatTensor)
        heatmap = np.transpose(heatmap, [2,0,1]) #(13, 260, 346)
        heatmap = torch.from_numpy(heatmap).type(torch.FloatTensor)
        return event, heatmap
    
    def mm_cnn_getitem(self, index):
        event, label = self.get_raw_event(index)
        event = event.astype('float32')
        if np.max(event) > 0.:
            event = (255 * event / np.max(event)) # normalize to [0,255]
        event = event.reshape((1,) + event.shape) # (1, 260, 344)
        event = torch.from_numpy(event).type(torch.FloatTensor)
        label = torch.tensor(label).type(torch.LongTensor)
        return event, label
    def action_getitem(self, index):
        skeleton_basename, feature, label, cam, frame_index = self.table[index]
        skeleton_pth = os.path.join(self.data_dir, skeleton_basename)
        event_pth = skeleton_pth.replace('_label.h5', '.h5')
        f = h5py.File(event_pth,'r')
        event = np.array(f['DVS'][:,:,:,cam], dtype=np.uint8) #(260, 346)
        event = np.nan_to_num(event)
    def __len__(self):
        return self.len

class FasterPose7500(data.Dataset):
    def __init__(self, data_dir, p_mat_dir = '', skeleton_dir='', mode = 'test', cameras = [2,3], use_percentage = 100, experiment = 'pose',
    window_size = 10, step_size = 5):
        self.data_dir = data_dir
        self.cameras = cameras
        self.table = []
        self.experiment = experiment
        self.window_size = window_size
        self.step_size = step_size
        for pth in glob(os.path.join(data_dir, '*')):
            data = np.load(pth, mmap_mode = 'r')
            feature = os.path.basename(pth).split('.')[0]
            subject, session, mov = [int(_) for _ in feature.split('_')]
            label, is_train = label_mapping(subject, session, mov)
            if (is_train and mode=='test') or (not is_train and mode=='train'):
                continue
            num_frame = len(data)
            if 'long' in experiment:
                for cam in cameras:
                    self.table.append((pth, feature, label, cam, -1))
            elif experiment=='short' or experiment=='short_repeat':
                window_num, _ = slicing_windows(num_frame, self.window_size, self.step_size)
                for cam in cameras:
                    for i in range(window_num):
                        self.table.append((pth, feature, label, cam, i))
            else:
                for cam in cameras:
                    for i in range(num_frame):
                        self.table.append((pth, feature, label, cam, i))
             
        self.label = [t[2] for t in self.table]
        self.len = len(self.table)
        dest_len = (int)(self.len * (use_percentage / 100.0))
        self.set_len(dest_len)
        
        if p_mat_dir!='':
            self.p_mat_dir = p_mat_dir
            self.skeleton_dir = skeleton_dir
            self.load_p_mat_cam()

    def __getitem__(self, index):
        pth, feature, label, cam, i = self.table[index]
        if self.experiment=='long':
            event = np.load(pth, mmap_mode='r')[:,:,:,cam]
            event = np.nan_to_num(event)
            event = event.astype('float32')
            for i in range(self.window_size):
                max_event = np.max(event[i])
                if max_event > 0.:
                    event[i] = (255 * event[i] / max_event)
            event = np.expand_dims(event, axis = 1)
            event = torch.from_numpy(event).type(torch.FloatTensor)
            return event, label, feature
        elif self.experiment=='long_sample':
            event = np.load(pth, mmap_mode='r')
            num_frame = len(event)
            window_num, _ = slicing_windows(num_frame, self.window_size, self.step_size)
            list_event = []
            for i in range(window_num):
                idx = self.step_size * i
                x = event[idx, :,:,cam]
                x = np.nan_to_num(x)
                x = x.astype('float32')
                if np.max(x) > 0.:
                    x = (255 * x / np.max(x))
                x = np.expand_dims(x, axis = 0)
                list_event.append(x)
            event = np.stack(list_event)
            # print(event.shape)
            event = torch.from_numpy(event).type(torch.FloatTensor)
            return event, label, feature
        elif self.experiment=='long_random':
            event = np.load(pth, mmap_mode='r')
            num_frame = len(event)
            window_num, _ = slicing_windows(num_frame, self.window_size, self.step_size)
            idx = np.random.randint(0, high = window_num)
            start_idx = idx * self.step_size
            end_idx = start_idx + self.window_size
            event = event[start_idx:end_idx,:,:,cam]
            event = np.nan_to_num(event)
            event = event.astype('float32')
            for i in range(self.window_size):
                max_event = np.max(event[i])
                if max_event > 0.:
                    event[i] = (255 * event[i] / max_event)
            event = np.expand_dims(event, axis = 1)
            event = torch.from_numpy(event).type(torch.FloatTensor)
            return event, label, feature
        elif self.experiment=='short':
            start_idx = self.step_size * i
            end_idx = start_idx + self.window_size
            event = np.load(pth, mmap_mode='r')[start_idx:end_idx,:,:,cam]
            event = np.nan_to_num(event)
            event = event.astype('float32')
            for i in range(self.window_size):
                max_event = np.max(event[i])
                if max_event > 0.:
                    event[i] = (255 * event[i] / max_event)
            event = np.expand_dims(event, axis = 1)
        elif self.experiment=='mm_cnn' or self.experiment=='pose' or self.experiment=='shuffle':
            event = np.load(pth, mmap_mode='r')[i,:,:,cam]
            event = np.nan_to_num(event)
            event = event.astype('float32')
            if np.max(event) > 0.:
                event = (255 * event / np.max(event)) # normalize to [0,255]
            event = event.reshape((1,) + event.shape) # (1, 260, 344)
        elif self.experiment=='dhpcnn':
            event = np.load(pth, mmap_mode='r')[i,:,:,cam] #(260, 344)
            event = np.nan_to_num(event)
            event = event.astype('float32')
            if np.max(event) > 0.:
                event = (255 * event / np.max(event)) # normalize to [0,255]
            event = event.reshape((1,) + event.shape) # (1, 260, 344)
            subject, session, mov = [int(x) for x in feature.split('_')]
            skeleton_pth = os.path.join(self.skeleton_dir,
              'S{:d}_session{:d}_mov{:d}_7500events_label.h5'.format(subject, session, mov))
            skeleton = np.array(h5py.File(skeleton_pth,'r')['XYZ'][i], dtype=np.float32) #(3, 13)
            pose, pose_weight = self.gen_2dpose(skeleton, cam)
            heatmap = self.gen_heatmap(pose)
            heatmap = np.nan_to_num(heatmap)
            event = torch.from_numpy(event).type(torch.FloatTensor)
            heatmap = np.transpose(heatmap, [2,0,1])
            heatmap = torch.from_numpy(heatmap).type(torch.FloatTensor)
            return event, heatmap
        else:
            raise ValueError("Wrong experiment name")
        event = torch.from_numpy(event).type(torch.FloatTensor)
        label = torch.tensor(label).type(torch.LongTensor)
        return event, label

    def set_len(self, l):
        sample_index = sample(list(range(self.len)), l)
        self.label = [self.label[_] for _ in sample_index]
        self.table = [self.table[_] for _ in sample_index]
        self.len = l
    def __len__(self):
        return self.len

    def load_p_mat_cam(self):
        P1 = np.load(os.path.join(self.p_mat_dir,'P1.npy'))
        P2 = np.load(os.path.join(self.p_mat_dir,'P2.npy'))
        P3 = np.load(os.path.join(self.p_mat_dir,'P3.npy'))
        P4 = np.load(os.path.join(self.p_mat_dir,'P4.npy'))
        self.P_mat_cam = np.stack([P4,P1,P3,P2])
        self.p_mat_cam = self.P_mat_cam
        self.camera_positions = np.load(os.path.join(self.p_mat_dir, 'camera_positions.npy'))
        self.image_h, self.image_w = 260,344
    def gen_2dpose(self, skeleton, cam = 3):
        # input:(3,13)
        # output: (image_h, image_w, num_joints)
        return pose, pose_weight

    def gen_heatmap(self, pose, decay_maps_flag = True):
        # initialize the heatmaps
        return gen_heatmap(pose)

    def get_raw_event(self, index):
        skeleton_basename, feature, label, cam, frame_index = self.table[index]
        skeleton_pth = os.path.join(self.data_dir, skeleton_basename)
        event_pth = skeleton_pth.replace('_label.h5', '.h5')
        f = h5py.File(skeleton_pth,'r')
        skeleton = np.array(f['XYZ'])
        num_frame = len(skeleton)
        f = h5py.File(event_pth,'r')
        event = np.array(f['DVS'][frame_index][:,:,cam], dtype=np.uint8) #(260, 346)

        event = np.nan_to_num(event)
        return event, label
def GenPose7500Numpy():
    definitions = import_def()
    data_dir = definitions.new_pose_7500_dir
    output_dir = definitions.new_pose_numpy_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for skeleton_pth in tqdm(glob(os.path.join(data_dir, '*_7500events_label.h5'))):
        event_pth = skeleton_pth.replace('_label.h5', '.h5')
        skeleton_basename = os.path.basename(skeleton_pth)
        subject = (int)(skeleton_basename.split('_')[0][1:])
        session = (int)(skeleton_basename.split('_')[1][7:])
        mov = (int)(skeleton_basename.split('_')[2][3:])
        label, is_train = label_mapping(subject, session, mov)
        f = h5py.File(event_pth, 'r')
        event = np.array(f['DVS'][:,:,:,:], dtype=np.uint8) #(260, 346)
        np.save(os.path.join(output_dir, '{:d}_{:d}_{:d}.npy'.format(subject, session, mov)), event)

def test_faster():
    definitions = import_def()
    d = FasterPose7500(definitions.pose7500numpy_dir, 'test', experiment = 'short')
    event, label = d[0]
    print(len(d))
    print(event.size())
    print(label.size())

def test_len():
    definitions = import_def()
    d = FasterPose7500(definitions.pose7500numpy_dir, 'train')
    print(len(d))

def visualize_them():
    definitions = import_def()
    d = Pose7500(definitions.pose_7500_dir, definitions.p_mat_dir, mode='test', experiment='visualize')
    for i in range(97,99):
        event, heatmap, skeleton, pose, label, meta = d[i]
        # print(event.sum())
        feature, cam, index = meta['feature'], meta['cam'], meta['frame_index']
        if cam==3:
            output_str=  '{}_{}_{}'.format(i, feature, index)
            subdir = 'new'
            dst_dir = os.path.join(definitions.figures_dir, subdir)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            visualize_event(event, os.path.join(dst_dir, '{}_event.png'.format(output_str)))
            visualize_heatmap(heatmap, os.path.join(dst_dir, '{}_heatmap.png'.format(output_str)))
            visualize_pose(pose, os.path.join(dst_dir, '{}_pose.png'.format(output_str)))
    
def visualize_raw():
    definitions = import_def()
    # pth = os.path.join(definitions.pose_7500_dir, 'S13_session3_mov4_7500events.h5')
    pth = os.path.join('/mnt/hdd/wuxiao/DHP19/matlab_official/h5_dataset_7500_events/344x260', 'S13_session3_mov4_7500events.h5')
    f = h5py.File(pth, 'r')
    event = np.array(f['DVS'][6,:,:,3])
    print(event.shape)
    visualize_event(event, os.path.join(definitions.figures_dir, 'visualize','13_13_3_4_3_3_6_event_official.png'))
    
def test_faster_dhpcnn():
    definitions = import_def() 
    d = FasterPose7500(definitions.new_pose_numpy_dir, definitions.p_mat_dir, definitions.new_pose_7500_dir, 'test', experiment = 'dhpcnn')
    event, heatmap = d[0]
    print(event.shape)
    print(heatmap.shape)
def get_len():
    definitions = import_def() 
    d = FasterPose7500(definitions.new_pose_numpy_dir, '', '', 'train', experiment = 'mm_cnn')
    print(len(d))
if __name__=='__main__':
    get_len()