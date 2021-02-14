from torch.utils.data import Dataset
import os.path as osp
import torchaudio
from utils import LABELS_10, LABELS_3
import torch
import librosa
import random


class BasicDataset(Dataset):
    def __init__(self, data_dir, features_dir, train_df, n_classes):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.file_names = train_df['filename']
        self.labels = train_df['scene_label']
        self.n_classes = n_classes

    def __len__(self):
        return self.file_names.shape[0]

    def __getitem__(self, i):
        audio_file = osp.basename(self.file_names.iloc[i])
        assert osp.exists(osp.join(self.data_dir, audio_file)) == 1, \
            f"audio file {osp.join(self.data_dir, audio_file)} doesn't exist"
        if self.n_classes == 10:
            aug_num = random.randint(0, 4)
        elif self.n_classes == 3:
            aug_num = random.randint(0, 4)
        mel = torch.load(osp.join(self.features_dir, f'{audio_file.replace(".wav", "")}_mel_{aug_num}.pkl'))
        if self.n_classes == 3:
            label = LABELS_3[self.labels.iloc[i]]
        if self.n_classes == 10:
            label = LABELS_10[self.labels.iloc[i]]

        return {
            'mels': mel.type(torch.FloatTensor),
            'label': torch.tensor(label)
        }
