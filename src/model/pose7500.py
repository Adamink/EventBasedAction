import torch
import torch.nn as nn
import numpy as np
class DHP_CNN(nn.Module):
    def __init__(self, input_size = (1, 260, 344)):
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
        self.fc = nn.Linear(np.sum(input_size), output_size)
    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3),dim = 1)
        x = self.fc(x)
        return x

class MM_CNN(nn.Module):
    def __init__(self, fc_size = (1024, 1024, 26), num_classes = 33):
        super().__init__()
        c = 1
        self.feat_fc_size, self.heat_fc_size, self.pose_fc_size = fc_size
        
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
        self.feat_fc = nn.Linear(512 * 8 * 10, self.feat_fc_size)

        # self.feature_path = nn.Sequential(
        #     self.feat_conv1,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), #(32, 44)
        #     self.feat_conv2,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), #(16, 22)
        #     self.feat_conv3,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), #(8, 11)
        #     self.feat_fc
        # )

        self.heat_conv1 = nn.Conv2d(16, 32, 3, padding = 1)
        self.heat_conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.heat_conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.heat_conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.heat_conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.heat_fc = nn.Linear(512 * 8 * 10, self.heat_fc_size)

        # self.heatmap_path = nn.Sequential(
        #     self.heat_conv1,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     self.heat_conv2,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     self.heat_conv3,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     self.heat_conv4,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     self.heat_conv5,
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     self.heat_fc
        # )

        # self.pose_path = nn.Sequential(
        #     self.pred_cube,
        #     self.activation_17,
        #     Heatmap2Pose(),
        # )
        self.gen_pose = Heatmap2Pose()


        self.merge_fc = MergeFC(fc_size, num_classes)
    def forward(self, x):
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
        
        feat = self.feat_conv1(x)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.feat_conv2(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.feat_conv3(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.feat_fc(feat)

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

        heat = self.heat_conv1(x)
        heat = self.relu(heat)
        heat = self.maxpool(heat)
        heat = self.heat_conv2(heat)
        heat = self.relu(heat)
        heat = self.maxpool(heat)
        heat = self.heat_conv3(heat)
        heat = self.relu(heat)
        heat = self.maxpool(heat)
        heat = self.heat_conv4(heat)
        heat = self.relu(heat)
        heat = self.maxpool(heat)
        heat = self.heat_conv5(heat)
        heat = self.relu(heat)
        heat = self.maxpool(heat)
        heat = heat.view(heat.size(0), -1)
        heat = self.heat_fc(heat)

        x = self.pred_cube(x)
        x = self.activation_17(x)
        pose = self.gen_pose(x)

        # x = self.encoder(x)
        # feat = self.feature_path(x)
        # x = self.decoder(x)
        # heat = self.heatmap_path(x)
        # pose = self.pose_path(x)

        output = self.merge_fc(feat, heat, pose)
        return output
    def load_pretrain(self, model_pth):
        state_dict = torch.load(model_pth)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        state = self.state_dict()
        state.update(new_state_dict)
        self.load_state_dict(state)
if __name__ == '__main__':    
    input_size = (1, 260, 344)
    a = torch.zeros((16,) + input_size)
    m = MM_CNN()
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    state_dict_pth = '../../checkpoints/pose_estimation_103.pth'
    m.load_pretrain(state_dict_pth)

    