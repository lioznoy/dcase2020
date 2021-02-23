import soundfile as sf
import torchaudio
import wavio
import numpy as np
import torch
from librosa.feature import melspectrogram, delta
import matplotlib.pyplot as plt
from librosa.effects import pitch_shift

# Extract audio data and sampling rate from file
audio_file = 'airport-barcelona-0-10-a.wav'
audio_path = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/audio/airport-barcelona-0-10-a.wav'
waveform, sample_rate = torchaudio.load(audio_path)
wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/original.wav", waveform[0].numpy(), sample_rate, sampwidth=2)
waveform_new = torchaudio.functional.contrast(waveform)
print(10 * torch.log10((torch.abs(waveform.std() - waveform_new.std())) / waveform.std()))
wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/contrast.wav", waveform_new[0].numpy(), sample_rate,
            sampwidth=2)
waveform_new = waveform + torch.randn(waveform.shape) * 0.001
print(10 * torch.log10((torch.abs(waveform.std() - waveform_new.std())) / waveform.std()))
wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/noise.wav", waveform_new[0].numpy(), sample_rate,
            sampwidth=2)
# waveform_new = torchaudio.functional.flanger(waveform, sample_rate)
# print(10 * torch.log10((torch.abs(waveform.std() - waveform_new.std())) / waveform.std()))
# wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/flanger.wav", waveform_new[0].numpy(), sample_rate, sampwidth=2)

one_mel = melspectrogram(waveform.squeeze(0).numpy(), sr=sample_rate, n_fft=2048, hop_length=1024,
                         n_mels=128, fmin=0.0, fmax=sample_rate / 2, htk=True, norm=None)
one_mel = np.log(one_mel + 1e-8)
one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))

plt.imshow(one_mel)
plt.savefig('/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/original_mel.jpg')
plt.close()

waveform, sample_rate = torchaudio.load(audio_path)
bitdepth = 6
waveform_drc = np.sign(waveform) * np.log(1 + (2 ** bitdepth - 1) * np.abs(waveform)) / np.log(2 ** bitdepth)
waveform_drc = np.round(waveform_drc * (2 ** (bitdepth - 1))) / (2 ** (bitdepth - 1))
waveform_drc = np.sign(waveform_drc) * ((2 ** bitdepth) ** np.abs(waveform_drc) - 1) / (2 ** bitdepth - 1)
wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/drc.wav", waveform_drc[0].numpy(), sample_rate,
            sampwidth=2)
one_mel = melspectrogram(waveform_drc.squeeze(0).numpy(), sr=sample_rate, n_fft=2048, hop_length=1024,
                         n_mels=128, fmin=0.0, fmax=sample_rate / 2, htk=True, norm=None)
one_mel = np.log(one_mel + 1e-8)
one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
print(np.linalg.norm(abs(waveform_drc - waveform), 2))
plt.imshow(one_mel)
plt.savefig('/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/drc_mel.jpg')
plt.close()

waveform_pitch_shift = pitch_shift(waveform.squeeze(0).numpy(), sample_rate, 0.5)
wavio.write("/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/pitch_shift.wav", waveform_pitch_shift, sample_rate,
            sampwidth=2)
one_mel = melspectrogram(waveform_pitch_shift, sr=sample_rate, n_fft=2048, hop_length=1024,
                         n_mels=128, fmin=0.0, fmax=sample_rate / 2, htk=True, norm=None)
one_mel = np.log(one_mel + 1e-8)
one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
print(np.linalg.norm(abs(waveform_pitch_shift - waveform.squeeze(0).numpy()), 2))
plt.imshow(one_mel)
plt.savefig('/mnt/disks/nlioz/DCASE2021/dcase2020/output_test/waveform_pitch_shift.jpg')
plt.close()
