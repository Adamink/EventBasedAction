import h5py
import numpy as np
from dataset.utils import visualize_event
pth_1 = '/mnt/hdd/wuxiao/DHP19/matlab_output/h5_dataset_7500_events/344x260/S13_session3_mov4_7500events.h5'
pth_2 = '/mnt/hdd/wuxiao/DHP19/matlab_official/h5_dataset_7500_events/344x260/S13_session3_mov4_7500events.h5'

f1 = np.array(h5py.File(pth_1, 'r')['DVS'][5,:,:,3], dtype=np.uint8)
f2 = np.array(h5py.File(pth_2, 'r')['DVS'][5,:,:,3], dtype=np.uint8)
print(f1)
print(f1.dtype)
print(f1.max())
print(f1.min())
print('*'*10)
print(f2.dtype)
print(f2.max())
print(f2.min())
print((f1-f2).sum())
diff = f1 - f2

x,y = diff.nonzero()
print(x.shape)
print(y.shape)
print(f1[x[:100],y[:100]])
print(f2[x[:100],y[:100]])

visualize_event(f1, 'tmp.png')
visualize_event(f2, 'tmp_official.png')
