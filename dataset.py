from torch.utils.data import Dataset
import os.path as osp
import torchaudio
from utils import LABELS_10
import torch


class BasicDataset(Dataset):
    def __init__(self, data_dir, train_df):
        self.data_dir = data_dir
        self.file_names = train_df['filename']
        self.labels = train_df['scene_label']

    def __len__(self):
        return self.file_names.shape[0]

    def __getitem__(self, i):
        audio_file = osp.basename(self.file_names.iloc[i])
        assert osp.exists(osp.join(self.data_dir, audio_file)) == 1, \
            f"audio file {osp.join(self.data_dir, audio_file)} doesn't exist"
        waveform, sample_rate = torchaudio.load(osp.join(self.data_dir, audio_file))
        one_mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=1766, win_length=1766, hop_length=883,
                                                       n_mels=40)(waveform)
        mel = torch.cat([one_mel, one_mel, one_mel], dim=0)
        label = LABELS_10[self.labels.iloc[i]]

        return {
            'mels': mel.type(torch.FloatTensor),
            'label': torch.tensor(label)
        }
