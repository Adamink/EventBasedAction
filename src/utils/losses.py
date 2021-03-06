import torch
import torch.nn.functional as F
import numpy as np
import cv2

def equal(y_pred, y_label):
    acc = (y_pred==y_label).sum()
    return acc
    
def accuracy(y_pred, label):
    max_index = y_pred.max(dim = 1)[1]
    acc = (max_index==label).sum()
    return acc

def cross_entropy(y_pred, label):
    return F.cross_entropy(y_pred, label)

def mse2D(y_pred, y_true):
    # y_pred:(batch, 13, 260, 346)
    diff = y_pred - y_true
    mean_over_ch = torch.mean(diff**2, dim = 1)
    final = torch.sum(mean_over_ch)
    return final

def get_indices(y):
    # y: (batch, 13, 260, 346)
    batch, num_joints, height, width = y.size()
    m = y.view(batch, num_joints, -1).argmax(-1)
    indices = torch.stack([(m // width), (m % width)], dim = 2)
    return indices

def mpjpe(y_pred, y_true):
    # y_pred:(batch, 13, 260, 344)
    m_pred = get_indices(y_pred)
    m_true = get_indices(y_true)
    diff = m_pred - m_true
    np_pred = m_pred.cpu().numpy()
    np_true = m_true.cpu().numpy()
    np_diff = diff.cpu().numpy()
    ret = torch.sqrt(torch.sum((diff**2).float(), dim = 2)) #(batch, 13, 0)
    ret = torch.sum(torch.mean(ret, dim = 1), dim = 0)
    return ret

def mpjpe_small(y_pred, y_true):
    # y_pred:(batch, 13, 56, 56)
    def resize(x):
        # input: (batch, 13, 56, 56)
        # output: (batch, 13, 260, 344)
        x = x.reshape((-1, 56, 56))
        x = np.transpose(x, (1,2,0))
        print(x.shape)
        x = cv2.resize(x, dsize=(260,344))
        x = np.transpose(x, (2,0,1))
        print(x.shape)
    y_pred = torch.FloatTensor(resize(y_pred.cpu().numpy()))
    y_true = torch.FloatTensor(resize(y_true.cpu().numpy()))
    return mpjpe(y_pred, y_true)

    
def gen_2dpose_from_heatmap(heatmap):
    # heatmap: (13, 260, 344)
    c, w, h = heatmap.shape
    m = np.reshape(heatmap, (c, -1)).argmax(axis = 1) #(13, )
    indices= np.stack([m // h, m % h]) #(2, 13)
    return indices

def mse2D_numpy(y_pred, y_true):
    #(batch, 13, 260, 344)
    diff = y_true - y_pred
    mean_over_ch = np.mean(diff ** 2, axis = 1)
    final = np.sum(mean_over_ch)
    return final

def get_indices_numpy(y):
    batch, num_joints, height, width = y.shape
    m = np.argmax(y.reshape((batch, num_joints, -1)), axis = -1) #(batch, 13)
    indices = np.stack([m // width, m % width], axis = 2) 
    return indices
def mpjpe_numpy(y_pred, y_true):
    # (batch, 13, 260, 344)
    b, j, h, w = y_true.shape
    m_pred = get_indices_numpy(y_pred)
    m_true = get_indices_numpy(y_true) #(batch, 13, 2)
    # mask = np.ones((b, h, 2))
    # mask[m_true==0] = 0
    diff = m_pred - m_true # (batch, 13, 2)
    diff[m_true==0] = 0

    ret = np.sqrt(np.sum(diff ** 2, axis = 2)) #(batch, 13)
    return np.sum(np.mean(ret, axis = 1))

def mse_regression(y_pred, y_true):
    # (batch, 26)
    diff = y_pred - y_true
    # diff[y_true==0.] = 0 # TODO: check it
    ret = torch.sum(torch.mean((diff ** 2), dim = 1))
    return ret

def mpjpe_regression(y_pred, y_true):
    diff = y_pred - y_true
    # diff[y_true==0.] = 0 # TODO: check it
    batch = diff.size()[0]
    diff = diff.view((batch, 13, 2))
    ret = torch.sqrt(torch.sum((diff**2).float(), dim = 2)) #(batch, 13, 0)
    ret = torch.sum(torch.mean(ret, dim = 1), dim = 0)
    return ret

if __name__=='__main__':
    # a = np.random.rand(3, 13, 260, 344)
    # b = np.random.rand(3, 13, 260, 344)
    # a_ind = get_indices_numpy(a)
    # print(a_ind.shape)
    # l1 = mse2D_numpy(a, b)
    # l1_g = mse2D(torch.Tensor(a), torch.Tensor(b))
    # print(l1)
    # print(l1_g)
    # l2 = mpjpe_numpy(a, b)
    # l2_g = mpjpe(torch.Tensor(a), torch.Tensor(b))
    # print(l2)
    # print(l2_g)
    a = torch.randn(16, 26)
    b = torch.randn(16, 26)
    l = mse_regression(a, b)
    acc = mpjpe_regression(a,b)
    print(l)
    print(acc)
