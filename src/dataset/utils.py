import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def import_def():
    sys.path.append('../')
    import definitions
    sys.path.remove('../')
    return definitions
def label_mapping(subject, session, mov):
    # return (label, is_train)
    mov_numbers = [0, 8, 6, 6, 6, 7]
    accu_numbers = [0,8,14,20,26,33]
    label = accu_numbers[session - 1] + mov - 1
    train_subject_range = range(1,13)
    is_train = subject in train_subject_range
    return (label, is_train)

def gen_2dpose_from_heatmap(heatmap):
    # heatmap: (13, 260, 344)
    c, w, h = heatmap.shape
    m = np.reshape(heatmap, (c, -1)).argmax(axis = 1) #(13, )
    indices= np.stack([m // h, m % h]) #(2, 13)
    return indices

def gen_heatmap(pose, image_size = [260, 344], decay = True):
    # initialize the heatmaps
    image_h, image_w = image_size
    num_joints = 13
    heatmap = np.zeros((image_h, image_w, num_joints), dtype=np.float32)
    
    v = pose[0]
    u = pose[1]
    k = 2 # constant used to better visualize the joints when not using decay

    mask = np.ones(u.shape).astype(np.float32)
    mask[u >= image_w - 1] = 0
    mask[u <= 0] = 0
    mask[v >= image_h - 1] = 0
    mask[v <= 0] = 0
    
    def decay_heatmap(heatmap, sigma2=4):
        heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma2)
        heatmap /= np.max(heatmap) # to keep the max to 1
        return heatmap

    for fmidx,pair in enumerate(zip(v,u, mask)):
        x, y, m = pair
        if m:
            if decay:
                heatmap[x, y, fmidx] = 1
                heatmap[:,:,fmidx] = decay_heatmap(heatmap[:,:,fmidx])
            else:
                heatmap[(x-k):(x+k+1),(y-k):(y+k+1), fmidx] = 1
    return heatmap

def visualize_event_heatmap(event, heatmap, save_pth):
    # event: (H, W)
    # heatmap: (H, W, 13)
    plt.figure()
    plt.imshow(event, cmap = 'gray')
    plt.imshow(np.sum(heatmap, axis=-1), alpha=.5)
    plt.savefig(save_pth)
    return 

def visualize_heatmap(heatmap, save_pth):
    # heatmap: (260, 346, 13)
    plt.figure()
    plt.imshow(np.sum(heatmap, axis =-1), alpha = .5)
    plt.savefig(save_pth)

def gen_bone_from_joint(joint):
    C, T, V, M = joint.shape
    directed_edges = [(1,3),(2,4),(3,5),(4,6),(1,7),(2,8),(7,9),(8,10),(9,11),(10,12)]
    bone = np.zeros(shape = (C, T, len(directed_edges), M)) #(3,10,10,1)
    for idx,(v1,v2) in enumerate(directed_edges):
        bone[:,:,idx,:] = joint[:,:,v1,:] - joint[:,:,v2,:]
    return bone
    
def slicing_windows(length, window, step):
    window_num = (length - window - 1) // step + 1 if length >= window else 0
    return window_num, [x * step for x in range(window_num)]
    
def parse_raw_filename(name):
    # S14_session1_mov2_raw_raw.mat
    # return subject, session, mov
    name = name.split('.')[0]
    subject = (int)(name.split('_')[0][1:])
    session = (int)(name.split('_')[1][7:])
    mov = (int)(name.split('_')[2][3:])
    return subject, session, mov

if __name__=='__main__':
    definitions = import_def()
    print(definitions.src_dir)
