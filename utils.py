import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import torch
import random
from spec_augmentation_functions import get_flat_grid_locations, interpolate_spline, create_dense_flows, \
    dense_image_warp

LABELS_10 = {'airport': 0, 'shopping_mall': 1, 'metro_station': 2, 'street_pedestrian': 3, 'public_square': 4,
             'street_traffic': 5, 'tram': 6, 'bus': 7, 'metro': 8, 'park': 9}

LABELS_3 = {'indoor': 0, 'outdoor': 1, 'transportation': 2}


def plot_loss_score(epochs, train, val, timestamp, loss_score):
    plt.plot(epochs, train, 'r')
    plt.plot(epochs, val, 'g')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel(f'{loss_score}', fontsize=16)
    plt.legend([f"train {loss_score}", f"validation {loss_score}"])
    plt.grid()
    plt.savefig(osp.join('outputs', f'{loss_score}_{timestamp}.jpg'))
    plt.close()


def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ):
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ):
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)


def drc(waveform, bitdepth):
    waveform_drc = np.sign(waveform) * np.log(1 + (2 ** bitdepth - 1) * np.abs(waveform)) / np.log(2 ** bitdepth)
    waveform_drc = np.round(waveform_drc * (2 ** (bitdepth - 1))) / (2 ** (bitdepth - 1))
    waveform_drc = np.sign(waveform_drc) * ((2 ** bitdepth) ** np.abs(waveform_drc) - 1) / (2 ** bitdepth - 1)
    return waveform_drc
