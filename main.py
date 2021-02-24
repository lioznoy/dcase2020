import argparse
import torch
from datetime import datetime
import sys
import os
import os.path as osp
from train import train_net
from model import ClassifierModule10, ClassifierModule3, BaseLine, FCNNModel, ClassifierModule10_2path
from model_resnet_with_att import ResNetk


def get_args():
    parser = argparse.ArgumentParser(description='Run train on dcase 2020 challenge task 1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-w', '--weights', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--data_dir_10', '--data_dir_10', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/audio',
                        help='dir with audio files')
    parser.add_argument('--data_dir_3', '--data_dir_3', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/audio',
                        help='dir with audio files')

    parser.add_argument('--features_dir_10', '--features_dir_10', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/mel_features_3d',
                        help='dir with audio files')
    parser.add_argument('--features_dir_3', '--features_dir_3', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/mel_features_3d',
                        help='dir with audio files')

    parser.add_argument('--folds_dir_10', '--folds_dir_10', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/',
                        help='dir with folds csv files')
    parser.add_argument('--folds_dir_3', '--folds_dir_3', type=str,
                        default='../datasets/TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/',
                        help='dir with folds csv files')

    parser.add_argument('--output_dir', '--output_dir', type=str,
                        default='outputs', help='dir to save figures and logs')
    parser.add_argument('--dir_checkpoint', '--dir_checkpoint', type=str,
                        default='checkpoints', help='dir to save best nets during training')
    parser.add_argument('--n_classes', '--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--gpu', '--gpu', type=str, default=0, help='gpu to use')
    parser.add_argument('--backbone_10', '--backbone_10', type=str, default='resnet18',
                        help='backbone for CNN classifier 10 classes')
    parser.add_argument('--backbone_3', '--backbone_3', type=str, default='mobilenet_v2',
                        help='backbone for CNN classifier 3 classes')
    parser.add_argument('--setup', '--setup', type=str, default='two_path',
                        help='setup to run single_path / fcnn / two_path / two_path_fcnn /')
    parser.add_argument('--augmentations', '--augmentations', type=str, default='all',
                        help='without / no_impulse / all')
    return parser.parse_args()


if __name__ == '__main__':
    # load args
    args = get_args()
    assert args.n_classes == 3 or args.n_classes == 10, 'n_classes must be 3 or 10!'
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not osp.exists(args.dir_checkpoint):
        os.mkdir(args.dir_checkpoint)
    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp = f'{dt_object.year}_{dt_object.month}_{dt_object.day}_{dt_object.hour}_{dt_object.minute}'
    # print args to log
    print(args)
    print(args, file=open(osp.join(args.output_dir, f'log_{timestamp}.txt'), 'w+'))
    # choose device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # print device to log
    print(f'Using device {device}')
    print(f'Using device {device}', file=open(osp.join(args.output_dir, f'log_{timestamp}.txt'), 'a'))
    # init net
    # net = ResNetk(k=18, use_cbam_block=True, use_cbam_class=True)
    if args.n_classes == 3:
        if args.setup == 'fcnn':
            net = FCNNModel(channels=6, output_features=3)
        else:
            net = ClassifierModule3(n_class=3)
    elif args.n_classes == 10:
        if args.setup == 'fcnn':
            net = FCNNModel(channels=3, output_features=10)
        if args.setup == 'two_path':
            net = ClassifierModule10_2path(backbone10=args.backbone_10, backbone3=args.backbone_10, fcnn=False)
        if args.setup == 'two_path_fcnn':
            net = ClassifierModule10_2path(backbone10=args.backbone_10, backbone3=args.backbone_10, fcnn=True)
        if args.setup == 'single_path':
            net = ClassifierModule10(backbone=args.backbone_10)
    # net = BaseLine()
    print(net, file=open(osp.join(args.output_dir, f'log_{timestamp}.txt'), 'a'))
    net.to(device=device)
    # load checkpoint weights
    if args.weights:
        net.load_state_dict(torch.load(args.weights, map_location=device))
    # start training
    try:
        if args.n_classes == 3:
            train_net(net=net,
                      epochs=args.epochs,
                      data_dir=args.data_dir_3,
                      output_dir=args.output_dir,
                      features_dir=args.features_dir_3,
                      folds_dir=args.folds_dir_3,
                      dir_checkpoint=args.dir_checkpoint,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      n_classes=args.n_classes,
                      timestamp=timestamp,
                      augmentations=args.augmentations)
        elif args.n_classes == 10:
            train_net(net=net,
                      epochs=args.epochs,
                      data_dir=args.data_dir_10,
                      output_dir=args.output_dir,
                      features_dir=args.features_dir_10,
                      folds_dir=args.folds_dir_10,
                      dir_checkpoint=args.dir_checkpoint,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      n_classes=args.n_classes,
                      timestamp=timestamp,
                      setup=args.setup,
                      augmentations=args.augmentations)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), osp.join(args.dir_checkpoint, 'INTERRUPTED.pth'))
        print()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print('training finished')
    print('training finished', file=open(osp.join(args.output_dir, f'log_{timestamp}.txt'), 'a'))
