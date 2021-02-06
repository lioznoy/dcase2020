from torch.utils.data import Dataset
import os.path as osp
import torchaudio
from utils import LABELS_10
import torch
import librosa


class BasicDataset(Dataset):
    def __init__(self, data_dir, features_dir, train_df):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.file_names = train_df['filename']
        self.labels = train_df['scene_label']

    def __len__(self):
        return self.file_names.shape[0]

    def __getitem__(self, i):
        audio_file = osp.basename(self.file_names.iloc[i])
        assert osp.exists(osp.join(self.data_dir, audio_file)) == 1, \
            f"audio file {osp.join(self.data_dir, audio_file)} doesn't exist"
        # waveform, sample_rate = torchaudio.load(osp.join(self.data_dir, audio_file))
        # melkwargs = {'n_fft': 1766, 'f_min': 0}
        # one_mel = torchaudio.transforms.MFCC(sample_rate, n_mfcc=40, log_mels=False, melkwargs=melkwargs)(waveform)

        # one_mel = one_mel / one_mel.max()
        mel = torch.load(osp.join(self.features_dir, f'{audio_file.replace(".wav", "")}_mel.pkl'))
        mel[:, 0:32, :] = 0
        # mel = torch.cat([one_mel, one_mel, one_mel], dim=0)
        label = LABELS_10[self.labels.iloc[i]]

        return {
            'mels': mel.type(torch.FloatTensor),
            'label': torch.tensor(label)
        }
