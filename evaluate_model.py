import argparse
import os
import os.path as osp
import pandas as pd
from datetime import datetime
import torch
from model import ClassifierModule10, ClassifierModule3, FCNNModel, ClassifierModule10_2path
from dataset import BasicDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from utils import LABELS_10, LABELS_3
import matplotlib.pyplot as plt
from quantize_weights import quantize_weights, pruning
import torch.nn.utils.prune as prune
from utils import vec3_to_vec10


def get_prediction(test_loader, net, device, setup):
    all_labels = np.array([])
    all_predictions = np.array([])
    for batch in tqdm(test_loader, total=len(test_loader)):
        mels = batch['mels']
        label = batch['label']
        mels = mels.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        with torch.no_grad():
            if setup == 'two_path':
                pred_vec10, pred_vec3 = net(mels)
                all_predictions = np.concatenate(
                    [all_predictions, (pred_vec10 * vec3_to_vec10(pred_vec3, device)).argmax(dim=1).cpu().numpy()])
            else:
                pred_vec = net(mels)
                all_predictions = np.concatenate([all_predictions, pred_vec.argmax(dim=1).cpu().numpy()])
            all_labels = np.concatenate([all_labels, label.cpu().numpy()])
    return all_labels, all_predictions


def plot_results(prediction, labels, output_dir, rec_device, n_classes):
    acc = round(accuracy_score(prediction, labels) * 100, 2)
    cm = confusion_matrix(prediction, labels, normalize='true')
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.rcParams.update({'font.size': 18})
    if n_classes == 10:
        cmd = ConfusionMatrixDisplay(cm, display_labels=list(LABELS_10.keys()))
    else:
        cmd = ConfusionMatrixDisplay(cm, display_labels=list(LABELS_3.keys()))
    cmd.plot(cmap='Blues', xticks_rotation=60, values_format='.2f', ax=ax)
    plt.title(f'Acoustic Scene Clasiffication {rec_device}- Confusion Matrix - Accuracy = {acc}%', fontsize=18,
              fontweight='bold')
    plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
    plt.ylabel('True label', fontsize=18, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, f'confusion_matrix_{rec_device}.jpg'))


def get_args():
    parser = argparse.ArgumentParser(description='Run test on dcase 2020 challenge task 1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', '--weights', type=str, help='Load model weights from a .pth file', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size for test set', default=32)
    parser.add_argument('--data_dir', '--data_dir', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/audio',
                        help='dir with audio files')
    parser.add_argument('--features_dir', '--features_dir', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/mel_features_3d',
                        help='dir with audio files')
    parser.add_argument('--test_csv', '--test_csv', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/fold1_evaluate.csv',
                        help='test csv file')
    parser.add_argument('--output_dir', '--output_dir', type=str, default='output_test/', help='output dir')
    parser.add_argument('--n_classes', '--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--gpu', '--gpu', type=str, default=0, help='gpu to use')
    parser.add_argument('--backbone_10', '--backbone_10', type=str, default='resnet18',
                        help='backbone for CNN classifier 10 classes')
    parser.add_argument('--backbone_3', '--backbone_3', type=str, default='mobilenet_v2',
                        help='backbone for CNN classifier 3 classes')
    parser.add_argument('--setup', '--setup', type=str, default='resnet',
                        help='setup to run fcnn / two_path / single_path')
    return parser.parse_args()


if __name__ == '__main__':
    # load args
    args = get_args()
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    test_df = pd.read_csv(args.test_csv, sep='\t')
    if 'scene_label' not in test_df.columns:
        test_df['scene_label'] = test_df.apply(lambda x: x['filename'].split('/')[1].split('-')[0], axis=1)
    print(f'{test_df.shape[0]} audio files to evaluate')
    n_classes = int(args.n_classes)
    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp = f'{dt_object.year}_{dt_object.month}_{dt_object.day}_{dt_object.hour}_{dt_object.minute}'
    # choose device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)
    # init net
    if args.n_classes == 3:
        if args.setup == 'fcnn':
            net = FCNNModel(channels=6, output_features=args.n_classes)
        else:
            net = ClassifierModule3(n_class=args.n_classes)
    elif args.n_classes == 10:
        if args.setup == 'fcnn':
            net = FCNNModel(channels=3, output_features=args.n_classes)
        if args.setup == 'two_path':
            net = ClassifierModule10_2path(backbone10=args.backbone_10, backbone3=args.backbone_10, fcnn=False)
        if args.setup == 'two_path_fcnn':
            net = ClassifierModule10_2path(backbone10=args.backbone_10, backbone3=args.backbone_10, fcnn=True)
        if args.setup == 'single_path':
            net = ClassifierModule10(backbone=args.backbone_10)
    # net = BaseLine()
    net.to(device=device)
    weigths = torch.load(args.weights, map_location=device)
    if args.n_classes == 3:
        weigths, total_size_new = quantize_weights(weigths, 4)
        # weigths = pruning(weigths, 0.009, total_size_new)
    net.load_state_dict(weigths)
    # torch.quantization.quantize_dynamic(
    #     net,  # the original model
    #     {torch.nn.Linear},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8)  # the target dtype for quantized weights
    net.eval()
    # test data loader
    dataset_test = BasicDataset(args.data_dir, args.features_dir, test_df, args.n_classes, test=True,
                                augmentations='without')
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             drop_last=False)
    prediction, labels = get_prediction(test_loader, net, device, args.setup)
    plot_results(prediction, labels, args.output_dir, rec_device='all', n_classes=args.n_classes)
    rec_devices = test_df.apply(lambda x: x['filename'].split('/')[1].split('-')[-1].replace('.wav', ''), axis=1)
    for rec_device in rec_devices.unique():
        prediction_partial = prediction[rec_devices == rec_device]
        labels_partial = labels[rec_devices == rec_device]
        plot_results(prediction_partial, labels_partial, args.output_dir, rec_device=rec_device,
                     n_classes=args.n_classes)
