import numpy as np
import pickle

filename = '/mnt/disks/nlioz/DCASE2021/dcase2020/output_3class_mobilenet_v2/score_val_2021_2_24_7_59.pkl'
with open(filename, 'rb') as f:
    x = np.array(pickle.load(f))

print(x.max())
print(x.argmax())
