import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class DHP_CNN(nn.Module):
    def __init__(self, input_size = (1, 260, 344), pretrain_pth=''):
        # input: (batch, 1/3, 260, 344)
        # output: (batch, 13, 260, 344)
        super().__init__()
        c, h, w = input_size
        assert w==344, 'wrong input size!'
        self.conv1 = nn.Conv2d(c, 16, 3, padding = 1, bias = False)
        self.activation_1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2a = nn.Conv2d(16, 32, 3, padding = 1, bias = False)
        self.activation_2 = nn.ReLU()
        self.conv2b = nn.Conv2d(32, 32, 3, padding = 1, bias = False)
        self.activation_3 = nn.ReLU()
        self.conv2d = nn.Conv2d(32, 32, 3, padding = 1, bias = False)
        self.activation_4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3a = nn.Conv2d(32, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_5 = nn.ReLU()
        self.conv3b = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_6 = nn.ReLU()
        self.conv3c = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_7 = nn.ReLU()
        self.conv3d = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_8 = nn.ReLU()
        self.conv3_up = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.activation_9 = nn.ReLU()
        self.conv4a = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_10 = nn.ReLU()
        self.conv4b = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_11 = nn.ReLU()
        self.conv4c = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_12 = nn.ReLU()
        self.conv4d = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_13 = nn.ReLU()
        self.conv4_up = nn.ConvTranspose2d(32, 16, 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.activation_14 = nn.ReLU()
        self.conv5a = nn.Conv2d(16, 16, 3, padding = 1, bias = False)
        self.activation_15 = nn.ReLU()
        self.conv5d = nn.Conv2d(16, 16, 3, padding = 1, bias = False)
        self.activation_16 = nn.ReLU()
        self.pred_cube = nn.Conv2d(16, 13, 3, padding = 1, bias = False)
        self.activation_17 = nn.ReLU()

        if pretrain_pth!='':
            self.load_pretrain(pretrain_pth)
    def forward(self,x):
        x = self.conv1(x)
        x = self.activation_1(x)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.activation_2(x)
        x = self.conv2b(x)
        x = self.activation_3(x)
        x = self.conv2d(x)
        x = self.activation_4(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.activation_5(x)
        x = self.conv3b(x)
        x = self.activation_6(x)
        x = self.conv3c(x)
        x = self.activation_7(x)
        x = self.conv3d(x)
        x = self.activation_8(x)
        x = self.conv3_up(x)
        x = self.activation_9(x)
        x = self.conv4a(x)
        x = self.activation_10(x)
        x = self.conv4b(x)
        x = self.activation_11(x)
        x = self.conv4c(x)
        x = self.activation_12(x)
        x = self.conv4d(x)
        x = self.activation_13(x)
        x = self.conv4_up(x)
        x = self.activation_14(x)
        x = self.conv5a(x)
        x = self.activation_15(x)
        x = self.conv5d(x)
        x = self.activation_16(x)
        x = self.pred_cube(x)
        x = self.activation_17(x)
        return x
    def load_pretrain(self, model_pth):
        print("load model from " + model_pth)
        input_state = torch.load(model_pth)
        to_load = OrderedDict()

        state = self.state_dict()

        for k, v in input_state.items():
            name = k.replace('module.', '')
            if name in state:
                to_load[name] = v
        state.update(to_load)
        self.load_state_dict(state)
class Heatmap2Pose(nn.Module):
    def __init__(self):
        super(Heatmap2Pose, self).__init__()
    
    def forward(self, x):
        batch, num_joints, height, width = x.size()
        m = x.view(batch, num_joints, -1).argmax(-1)
        indices = torch.cat([(m // width), (m % width)], dim = 1).float()
        return indices

class MergeFC(nn.Module):
    def __init__(self, input_size, output_size):
        super(MergeFC, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, *x):
        x = torch.cat(x,dim = 1)
        x = self.fc(x)
        return x

class MM_CNN(nn.Module):
    def __init__(self, fc_size = (1024, 1024, 26), num_classes = 33, 
     paths = ['feat', 'heat', 'pose'], pretrain_pth = '', output_before_fc = False, output_both = False):
        super().__init__()
        c = 1
        self.feat_fc_size, self.heat_fc_size, self.pose_fc_size = fc_size
        self.output_before_fc = output_before_fc
        self.output_both = output_both
        self.paths = paths
        self.fc_dict = OrderedDict()
        self.fc_dict['feat'] = self.feat_fc_size
        self.fc_dict['heat'] = self.heat_fc_size
        self.fc_dict['pose'] = self.pose_fc_size

        self.final_fc_size = np.sum([self.fc_dict[_] for _ in self.paths])

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        # encoder part
        self.conv1 = nn.Conv2d(c, 16, 3, padding = 1, bias = False)
        self.activation_1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2a = nn.Conv2d(16, 32, 3, padding = 1, bias = False)
        self.activation_2 = nn.ReLU()
        self.conv2b = nn.Conv2d(32, 32, 3, padding = 1, bias = False)
        self.activation_3 = nn.ReLU()
        self.conv2d = nn.Conv2d(32, 32, 3, padding = 1, bias = False)
        self.activation_4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3a = nn.Conv2d(32, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_5 = nn.ReLU()
        self.conv3b = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_6 = nn.ReLU()
        self.conv3c = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_7 = nn.ReLU()
        self.conv3d = nn.Conv2d(64, 64, 3, padding = 2, dilation = 2, bias = False)
        self.activation_8 = nn.ReLU()

        self.encoder = nn.Sequential(
            self.conv1,
            self.activation_1,
            self.pool1,
            self.conv2a,
            self.activation_2,
            self.conv2b,
            self.activation_3,
            self.conv2d,
            self.activation_4,
            self.pool2,
            self.conv3a,
            self.activation_5,
            self.conv3b,
            self.activation_6,
            self.conv3c,
            self.activation_7,
            self.conv3d,
            self.activation_8
        )

        self.conv3_up = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.activation_9 = nn.ReLU()
        self.conv4a = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_10 = nn.ReLU()
        self.conv4b = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_11 = nn.ReLU()
        self.conv4c = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_12 = nn.ReLU()
        self.conv4d = nn.Conv2d(32, 32, 3, padding = 2, dilation = 2, bias = False)
        self.activation_13 = nn.ReLU()
        self.conv4_up = nn.ConvTranspose2d(32, 16, 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.activation_14 = nn.ReLU()
        self.conv5a = nn.Conv2d(16, 16, 3, padding = 1, bias = False)
        self.activation_15 = nn.ReLU()
        self.conv5d = nn.Conv2d(16, 16, 3, padding = 1, bias = False)
        self.activation_16 = nn.ReLU()

        self.decoder = nn.Sequential(
            self.conv3_up,
            self.activation_9,
            self.conv4a,
            self.activation_10,
            self.conv4b,
            self.activation_11,
            self.conv4c,
            self.activation_12,
            self.conv4d,
            self.activation_13,
            self.conv4_up,
            self.activation_14,
            self.conv5a,
            self.activation_15,
            self.conv5d,
            self.activation_16
        )

        self.pred_cube = nn.Conv2d(16, 13, 3, padding = 1, bias = False)
        self.activation_17 = nn.ReLU()

        # after activation 8
        self.feat_conv1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.feat_conv2 = nn.Conv2d(128, 256, 3, padding = 1)
        self.feat_conv3 = nn.Conv2d(256, 512, 3, padding = 1)
        self.feat_fc = nn.Linear(512 * 8 * 10, self.fc_dict['feat'])

        self.feat_to_fc = nn.Sequential(
            self.feat_conv1,
            self.relu,
            self.maxpool,
            self.feat_conv2,
            self.relu,
            self.maxpool,
            self.feat_conv3,
            self.relu,
            self.maxpool
        )
        self.heat_conv1 = nn.Conv2d(16, 32, 3, padding = 1)
        self.heat_conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.heat_conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.heat_conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.heat_conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.heat_fc = nn.Linear(512 * 8 * 10, self.fc_dict['heat'])

        self.heat_to_fc = nn.Sequential(
            self.heat_conv1,
            self.relu,
            self.maxpool,
            self.heat_conv2,
            self.relu,
            self.maxpool,
            self.heat_conv3,
            self.relu,
            self.maxpool,
            self.heat_conv4,
            self.relu,
            self.maxpool,
            self.heat_conv5,
            self.relu,
            self.maxpool,
        )
        self.gen_pose = Heatmap2Pose()

        # self.merge_fc = MergeFC(fc_size, num_classes)
        self.final_fc = nn.Linear(self.final_fc_size, num_classes)

        if pretrain_pth!='':
            self.load_pretrain(pretrain_pth)
    def forward(self, x):
        if not self.output_both:
            x = self.forward_to_fc(x)
            if not self.output_before_fc:
                x = self.final_fc(x)
            return x
        else:
            x = self.forward_to_fc(x)
            y = self.final_fc(x)
            return x,y

    def forward_to_fc(self, x):
        ret_list = []
        x = self.encoder(x)

        if 'feat' in self.paths:
            feat = self.feat_to_fc(x)
            feat = feat.view(feat.size(0), -1)
            feat = self.feat_fc(feat)
            ret_list.append(feat)

        if 'heat' in self.paths or 'pose' in self.paths:
            x = self.decoder(x)

        if 'heat' in self.paths:
            heat = self.heat_to_fc(x)
            heat = heat.view(heat.size(0), -1)
            heat = self.heat_fc(heat)
            ret_list.append(heat)

        if 'pose' in self.paths:
            x = self.pred_cube(x)
            x = self.activation_17(x)
            pose = self.gen_pose(x)
            ret_list.append(pose)
        return torch.cat(ret_list, dim = 1)
    
    def load_pretrain(self, model_pth):
        print("load model from " + model_pth)
        input_state = torch.load(model_pth)
        to_load = OrderedDict()

        state = self.state_dict()

        for k, v in input_state.items():
            name = k.replace('module.', '')
            if name in state:
                to_load[name] = v
        state.update(to_load)
        self.load_state_dict(state)
    
    def fix_pretrain(self):
        print("fix_pretrain")
        for param in self.encoder.parameters():
            param.require_grad = False
        for param in self.decoder.parameters():
            param.require_grad = False
        for param in self.pred_cube.parameters():
            param.require_grad = False
    
    def unfix_pretrain(self):
        print("unfix_pretrain")
        for param in self.parameters():
            param.require_grad = True

    def fix_paths(self, paths):
        if 'feat' in paths:
            for param in self.feat_to_fc.parameters():
                param.require_grad = False
        if 'heat' in paths:
            for param in self.heat_to_fc.parameters():
                param.require_grad = False 
        if 'pose' in paths:
            for param in self.pred_cube.parameters():
                param.require_grad = False
    
    def unfix_paths(self, paths):
        if 'feat' in paths:
            for param in self.feat_to_fc.parameters():
                param.require_grad = True
        if 'heat' in paths:
            for param in self.heat_to_fc.parameters():
                param.require_grad = True 
        if 'pose' in paths:
            for param in self.pred_cube.parameters():
                param.require_grad = True


def test_paths():
    input_size = (1, 260, 344)
    a = torch.zeros((16,) + input_size)
    import itertools 
    def findsubsets(s, n): 
        return list(itertools.combinations(s, n)) 
    paths = ['feat','heat','pose']
    for i in range(1, 4):
        allsubsets = findsubsets(paths, i)
        if i==3:
            allsubsets = (paths,)
        for subset in allsubsets:
            print(subset)
            m = MM_CNN(paths = subset)
            b = m(a)
            m1 = MM_CNN(paths = ['heat'])

def test_sequential():
    m = MM_CNN(paths = ['feat', 'heat', 'pose'])
    m.load_pretrain('../../checkpoints/pose7500/1125_15:05:54_model.pt')
    print(m.conv1.weight.data[0])
    print(m.encoder[0].weight.data[0])

def test_fix():
    m = MM_CNN(paths = ['feat', 'heat', 'pose'])
    m.fix_pretrain()
    m.unfix_pretrain()

def count_params_for_model(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
def get_parameter_num():
    m = MM_CNN(paths = ['feat', 'heat', 'pose'])
    m.unfix_pretrain()
    print(count_params_for_model(m))
    print(count_params_for_model(m.encoder))
    print(count_params_for_model(m.feat_to_fc))
    print(count_params_for_model(m.feat_fc))
    print(count_params_for_model(m.decoder))
    print(count_params_for_model(m.heat_to_fc))
    print(count_params_for_model(m.heat_fc))
    print(count_params_for_model(m.pred_cube))

def get_operation_num():
    model = MM_CNN(paths = ['feat', 'heat', 'pose'])
    from profiler import profile
    input_size = (1, 1, 260, 344)
    num_ops, num_params = profile(model, input_size)
    # custom_ops = {'ConvTranspose2d', ''}
    print(num_ops)
    print(num_params)
if __name__ == '__main__':    
    get_operation_num()
    