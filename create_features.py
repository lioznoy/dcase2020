import pandas as pd
import torchaudio
import os.path as osp
import torch
import os
from tqdm import tqdm
from librosa.feature import melspectrogram, delta
import numpy as np
from librosa.effects import pitch_shift
from utils import drc
import pickle
import random
import argparse

FILTER_RESPONSES = f'/mnt/disks/nlioz/DCASE2021/dcase2020/filter_responses/s.pickle'


def create_mels_deltas(waveform, sample_rate):
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


def create_fetures(audio_path, aug_index):
    waveform, sample_rate = torchaudio.load(audio_path)
    if aug_index == 1:
        waveform = waveform + torch.randn(waveform.shape) * 0.001
    if aug_index == 2:
        waveform = torchaudio.functional.contrast(waveform)
    if aug_index == 3:
        if waveform.shape[0] == 2:
            waveform = torch.cat([torch.tensor(pitch_shift(waveform.squeeze(0).numpy()[0, :],
                                                           sample_rate, np.random.random(1)[0])).unsqueeze(0),
                                  torch.tensor(pitch_shift(waveform.squeeze(0).numpy()[1, :],
                                                           sample_rate, np.random.random(1)[0])).unsqueeze(0)], dim=0)
        else:
            waveform = torch.tensor(pitch_shift(waveform.squeeze(0).numpy(),
                                                sample_rate, np.random.random(1)[0])).unsqueeze(0)
    if aug_index == 4:
        waveform = drc(waveform, bitdepth=6)
    if aug_index == 5 and '-a' in audio_path:
        with open(FILTER_RESPONSES.replace('s.', f's{random.randint(1, 6)}.'), 'rb') as handle:
            h = pickle.load(handle)
        waveform = torch.tensor(np.convolve(waveform.squeeze(0).numpy(), h, 'same')).unsqueeze(0)
    if waveform.shape[0] == 2:
        full_mel_3d = torch.cat(
            [create_mels_deltas(waveform[0], sample_rate), create_mels_deltas(waveform[1], sample_rate)], dim=0)
    else:
        full_mel_3d = create_mels_deltas(waveform, sample_rate)
    return full_mel_3d


def fix_feature(feature_pkl, data_dir, audio_file, i):
    try:
        torch.load(feature_pkl)
    except:
        audio_path = osp.join(data_dir, audio_file)
        mel_3d = create_fetures(audio_path, i)
        torch.save(mel_3d, feature_pkl)


force = False
fix = False
parser = argparse.ArgumentParser(description='3 or 10')
parser.add_argument('-n', '--n_classes', type=int, default=10, help='Number of epochs')
args = parser.parse_args()
if args.n_classes == 3:
    data_dir = '../datasets/TAU-urban-acoustic-scenes-2020-3class-development/'
else:
    data_dir = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'

feature_dir = osp.join(data_dir, 'mel_features_3d')
if not osp.exists(feature_dir):
    os.mkdir(feature_dir)
if args.n_classes == 3:
    meta_csv_file = '../datasets/TAU-urban-acoustic-scenes-2020-3class-development/meta.csv'
else:
    meta_csv_file = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv'
meta_csv_df = pd.read_csv(meta_csv_file, sep='\t')
aug_num = 5 if args.n_classes == 3 else 6
for audio_file in tqdm(meta_csv_df['filename']):
    for i in range(aug_num):
        feature_pkl = osp.join(feature_dir, f'{osp.basename(audio_file).replace(".wav", "")}_mel_{i}.pkl')
        if osp.exists(feature_pkl) and fix:
            fix_feature(feature_pkl, data_dir, audio_file, i)
            continue
        if osp.exists(feature_pkl) and not force:
            continue
        audio_path = osp.join(data_dir, audio_file)
        mel_3d = create_fetures(audio_path, i)
        torch.save(mel_3d, feature_pkl)
