import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

LABELS_10 = {'airport': 0, 'shopping_mall': 1, 'metro_station': 2, 'street_pedestrian': 3, 'public_square': 4,
             'street_traffic': 5, 'tram': 6, 'bus': 7, 'metro': 8, 'park': 9}

LABELS_3 = {'indoor': 0, 'outdoor': 1, 'transportation': 2}


def plot_loss(epochs, loss_train, loss_val, timestamp):
    plt.plot(epochs, loss_train, 'r')
    plt.plot(epochs, loss_val, 'g')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(["train loss", "validation loss"])
    plt.savefig(osp.join('outputs', f'losses_{timestamp}.jpg'))