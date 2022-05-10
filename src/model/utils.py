import torch
import torch.nn as nn
class TimeDistributed(nn.Module):
    """
    A layer that could be nested to apply sub operation to every timestep of sequence input.
    """ 
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        if self.batch_first:
            batch = x.size(0)
            timesteps = x.size(1)
        else:
            batch = x.size(1)
            timesteps = x.size(0)

        x_reshape = x.contiguous().view((-1,) + tuple(x.size()[2:]))  

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view((batch, timesteps) + tuple(y.size()[1:]))  
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view((timesteps, batch) + tuple(y.size()[1:]))  

        return y


if __name__=='__main__':
    fc = nn.Linear(10,20)
    fc = TimeDistributed(fc)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    a = torch.zeros((32, 10, 10))
    fc.cuda()
    a = a.cuda()
    b = fc(a)