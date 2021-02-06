import pandas as pd
import torchaudio
import os.path as osp
import torch
import os
from tqdm import tqdm
from librosa.feature import melspectrogram, delta
import numpy as np


def create_fetures(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    one_mel = melspectrogram(waveform.squeeze(0).numpy(), sr=sample_rate, n_fft=2048, hop_length=1024,
                             n_mels=128, fmin=0.0, fmax=sample_rate / 2, htk=True, norm=None)
    one_mel = np.log(one_mel + 1e-8)
    one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
    one_mel_delta = delta(one_mel)
    one_mel_delta = (one_mel_delta - np.min(one_mel_delta)) / (np.max(one_mel_delta) - np.min(one_mel_delta))
    one_mel_delta_delta = delta(one_mel, order=2)
    one_mel_delta_delta = (one_mel_delta_delta - np.min(one_mel_delta_delta)) / (
                np.max(one_mel_delta_delta) - np.min(one_mel_delta_delta))
    mel_3d = torch.cat([torch.tensor(one_mel).unsqueeze(0), torch.tensor(one_mel_delta).unsqueeze(0),
                        torch.tensor(one_mel_delta_delta).unsqueeze(0)], dim=0)
    return mel_3d


force = True
fix = False
data_dir = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
feature_dir = osp.join(data_dir, 'mel_features_3d')
if not osp.exists(feature_dir):
    os.mkdir(feature_dir)
meta_csv_file = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv'
meta_csv_df = pd.read_csv(meta_csv_file, sep='\t')
for audio_file in tqdm(meta_csv_df['filename']):
    feature_pkl = osp.join(feature_dir, f'{osp.basename(audio_file).replace(".wav", "")}_mel.pkl')
    if osp.exists(feature_pkl) and fix:
        try:
            torch.load(feature_pkl)
        except RuntimeError:
            audio_path = osp.join(data_dir, audio_file)
            mel_3d = create_fetures(audio_path)
            torch.save(mel_3d, feature_pkl)
            continue
    if osp.exists(feature_pkl) and not force:
        continue
    audio_path = osp.join(data_dir, audio_file)
    mel_3d = create_fetures(audio_path)
    torch.save(mel_3d, feature_pkl)
