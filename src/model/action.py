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
        self.num_classes = num_classes
        self.stage = 0
        if arch_option==0 or arch_option==1 or arch_option==5:
            self.mm_cnn = MM_CNN(fc_size, num_classes, paths, pretrain_pth, output_before_fc = True)
        elif arch_option==2 or arch_option==3 or arch_option==4:
            self.mm_cnn = MM_CNN(fc_size, num_classes, paths, pretrain_pth, output_before_fc = False)
        elif arch_option==6:
            self.mm_cnn = MM_CNN(fc_size, num_classes, paths, pretrain_pth, output_both = True)
        self.mm_feature_size = (int)(self.mm_cnn.final_fc_size)
        self.mm_cnn = TimeDistributed(self.mm_cnn)

        if arch_option==0:
            self.tnn = eval('nn.' + arch)(input_size = self.mm_feature_size, 
             hidden_size = hidden_size, 
             num_layers = num_layers,
             batch_first=True, 
             bidirectional=bidirectional)
            if not bidirectional:
                self.fc = nn.Linear(hidden_size, num_classes)
            else:
                self.fc = nn.Linear(hidden_size * 2, num_classes)
        elif arch_option==1:
            self.gru1 = nn.GRU(self.mm_feature_size, 512, 2, batch_first=True)
            self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
            self.gru3 = nn.GRU(256, 128, 1, batch_first=True)
            # self.tnn = nn.Sequential(self.gru1, self.gru2, self.gru3)
            self.fc = nn.Linear(128, num_classes)
        elif arch_option==2 or arch_option==3 or arch_option==4:
            self.fc = nn.Identity()
        elif arch_option==5:
            self.fc = self.mm_cnn.module.final_fc
            self.tnn = nn.GRU(self.mm_feature_size, self.mm_feature_size, num_layers = 2, batch_first=True)
        elif arch_option==6:
            self.mm_cnn = self.mm_cnn.module
            self.gru1 = nn.GRU(26, 110, batch_first=True)
            self.gru2 = nn.GRU(110, 110, batch_first=True)
            self.gru3 = nn.GRU(110, 110, batch_first=True)
            self.fc = nn.Linear(110, num_classes)
        if mean_after_fc:
            self.fc = TimeDistributed(self.fc)
            
    def forward(self, x):
        if self.arch_option==6:
            batch_size = x.size(0)
            seq_len = x.size(1)
            x = x.view((-1,)+ tuple(x.size()[2:]))
            x, y= self.mm_cnn(x)
            x = x.view((batch_size, seq_len)+tuple(x.size()[1:]))
            y = y.view((batch_size, seq_len) + tuple(y.size()[1:]))
            x = x[:,:,-26:]
            self.gru1.flatten_parameters()
            self.gru2.flatten_parameters()
            self.gru3.flatten_parameters()
            x,_ = self.gru1(x)
            x,_ = self.gru2(x)
            x,_ = self.gru3(x)
            if self.mean_after_fc:
                x = self.fc(x)
                x = torch.mean(x, dim = 1) 
            else:
                x = torch.mean(x, dim = 1)
                x = self.fc(x)
            if self.stage==0:
                return x
            else:
                y = torch.mean(y, dim = 1)
                return x + y
        # x: (batch, seq_len, 1, 260, 344)
        x = self.mm_cnn(x)
        # x: (batch, seq_len, mm_feature_size)
        if self.arch_option==0 or self.arch_option==5:
            self.tnn.flatten_parameters()
            x, _ = self.tnn(x)
        elif self.arch_option==1:
            self.gru1.flatten_parameters()
            self.gru2.flatten_parameters()
            self.gru3.flatten_parameters()
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
            x, _ = self.gru3(x)
        elif self.arch_option==2:
            x = torch.mean(x, dim = 1) #(batch, time, 33)
            return x
        elif self.arch_option==3: # max response
            y, y_where = x.max(dim = 2) #(batch, time)
            t, t_where = y.max(dim = 1) #(batch, )
            t_where = t_where.view((t_where.size() + (1,1)))
            t_where = t_where.repeat(1,1,self.num_classes)
            x = torch.gather(x, dim = 1, index = t_where)
            x = x.view((-1,) + (x.size(-1),))
            return x
        elif self.arch_option==4:
            x = x.view((-1,) + (x.size(2),))
            return x

                
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

    def update_stage(self):
        self.stage += 1