import numpy as np
import librosa
import os
import os.path as osp
from numpy.linalg import lstsq
from tqdm import tqdm
import wavio
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.optimize import lsq_linear
from librosa.feature import melspectrogram, delta
import pickle

plot_and_save_audio = True

for s in range(1, 7):
    dir_path_s = f'/mnt/disks/nlioz/DCASE2021/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/audio_for_filters/s{s}/'

    all_audio_a = np.array([])
    all_audio_s = np.array([])

    for audio_file in os.listdir(dir_path_s):
        if '-a' in audio_file:
            sig_a, rate_a = librosa.load(osp.join(dir_path_s, audio_file), sr=None)
            sig_s, rate_s = librosa.load(osp.join(dir_path_s, audio_file.replace('-a', f'-s{s}')), sr=None)
        else:
            continue
        sig_a_all = sig_a[:int(sig_a.shape[0])]
        for i in tqdm(range(1, 24)):
            sig_a_all = np.vstack([sig_a_all, np.roll(sig_a[:int(sig_a.shape[0])], i)])
        sig_a_all = sig_a_all.T
        if all_audio_a.shape[0] == 0:
            all_audio_a = sig_a_all.copy()
        else:
            all_audio_a = np.concatenate([all_audio_a, sig_a_all], axis=0)
        if all_audio_s.shape[0] == 0:
            all_audio_s = sig_s[:int(sig_s.shape[0])]
        else:
            all_audio_s = np.concatenate([all_audio_s, sig_s[:int(sig_s.shape[0])]], axis=0)
        print(audio_file)
    # h = lsq_linear(all_audio_a, all_audio_s1)
    h = np.dot(np.dot(np.linalg.inv(np.dot(all_audio_a.T, all_audio_a)), all_audio_a.T), all_audio_s)
    with open(f'/mnt/disks/nlioz/DCASE2021/dcase2020/filter_responses/s{s}.pickle', 'wb') as handle:
        pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if plot_and_save_audio:
        s_recon = np.convolve(sig_a, h, 'same')
        wavio.write(f"/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/s{s}_recon.wav", s_recon, rate_s,
                    sampwidth=2)
        wavio.write(f"/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/s{s}_original.wav", sig_s, rate_s,
                    sampwidth=2)
        wavio.write(f"/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/a{s}_original.wav", sig_a, rate_a,
                    sampwidth=2)
        plt.stem(h)
        plt.savefig(f'/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/h{s}_stem.jpg')
        plt.close()
        w, p = freqz(h)
        plt.plot(w, 20 * np.log10(abs(p)), 'b')
        plt.savefig(f'/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/h{s}_freqz.jpg')
        plt.close()
        print(np.linalg.norm(abs(s_recon - sig_s), 2))
        print(np.linalg.norm(abs(s_recon - sig_a), 2))

        one_mel = melspectrogram(sig_s, sr=rate_a, n_fft=2048, hop_length=1024,
                                 n_mels=128, fmin=0.0, fmax=rate_a / 2, htk=True, norm=None)
        one_mel = np.log(one_mel + 1e-8)
        one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
        plt.imshow(one_mel)
        plt.savefig(f'/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/mel_orig_s{s}.jpg')
        plt.close()

        one_mel = melspectrogram(s_recon, sr=rate_a, n_fft=2048, hop_length=1024,
                                 n_mels=128, fmin=0.0, fmax=rate_a / 2, htk=True, norm=None)
        one_mel = np.log(one_mel + 1e-8)
        one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
        plt.imshow(one_mel)
        plt.savefig(f'/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/mel_recon_s{s}.jpg')
        plt.close()

        one_mel = melspectrogram(sig_a, sr=rate_a, n_fft=2048, hop_length=1024,
                                 n_mels=128, fmin=0.0, fmax=rate_a / 2, htk=True, norm=None)
        one_mel = np.log(one_mel + 1e-8)
        one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
        plt.imshow(one_mel)
        plt.savefig(f'/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/mel_recon_a{s}.jpg')
        plt.close()


