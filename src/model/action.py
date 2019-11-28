import torch
import torch.nn as nn
from .pose7500 import MM_CNN
from .utils import TimeDistributed
class DeepLSTM(nn.Module):
    def __init__(self, arch = 'GRU', hidden_size = 100, num_layers = 3, dropout = 0.2, bidirectional = True,
     fc_size = (1024, 1024, 26), num_classes = 33, paths = ['feat', 'heat', 'pose'], 
     pretrain_pth = '', arch_option = 1, mean_after_fc = False):
        super(DeepLSTM, self).__init__()
        assert arch in ['GRU','LSTM']
        self.arch_option = arch_option
        self.mean_after_fc = mean_after_fc
        self.mm_cnn = MM_CNN(fc_size, num_classes, paths, pretrain_pth, output_before_fc = True)
        self.mm_feature_size = (int)(self.mm_cnn.final_fc_size)
        self.mm_cnn = TimeDistributed(self.mm_cnn)
        if arch_option==0:
            self.tnn = eval('nn.' + arch)(self.mm_feature_size, hidden_size, num_layers, dropout,
                batch_first=True, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size, num_classes)
        elif arch_option==1:
            self.gru1 = nn.GRU(self.mm_feature_size, 512, 2, batch_first=True)
            self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
            self.gru3 = nn.GRU(256, 128, 1, batch_first=True)
            # self.tnn = nn.Sequential(self.gru1, self.gru2, self.gru3)
            self.fc = nn.Linear(128, num_classes)
        if mean_after_fc:
            self.fc = TimeDistributed(self.fc)
            
    def forward(self, x):
        
        # x: (batch, seq_len, 1, 260, 344)
        x = self.mm_cnn(x)
        # x: (batch, seq_len, mm_feature_size)
        if self.arch_option==0:
            self.tnn.flatten_parameters()
            x, _ = self.tnn(x)
        elif self.arch_option==1:
            self.gru1.flatten_parameters()
            self.gru2.flatten_parameters()
            self.gru3.flatten_parameters()
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
            x, _ = self.gru3(x)
        # x: (batch, seq_len, mm_feature_size)
        if self.mean_after_fc:
            x = self.fc(x)
            x = torch.mean(x, dim = 1)
        else:
            x = torch.mean(x, dim = 1) 
            x = self.fc(x)
        return x
    def fix_cnn(self):
        for param in self.mm_cnn.parameters():
            param.require_grad = False
    
    def unfix_cnn(self):
        for param in self.mm_cnn.parameters():
            param.require_grad = True



            

