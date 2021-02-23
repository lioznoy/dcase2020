import numpy as np
import pickle

filename = '/mnt/disks/nlioz/DCASE2021/dcase2020/output_two_path_all_aug/score_val_2021_2_23_5_45.pkl'
with open(filename, 'rb') as f:
    x = np.array(pickle.load(f))

print(x.max())
print(x.argmax())
