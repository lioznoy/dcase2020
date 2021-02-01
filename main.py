import argparse
import torch
from datetime import datetime
import sys
import os
import os.path as osp
from train import train_net
from model import ClassifierModule, MLP

def get_args():
    parser = argparse.ArgumentParser(description='Run train on dcase 2020 challenge task 1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-w', '--weights', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--data_dir', '--data_dir', type=str,
                        default='../dataset/TAU-urban-acoustic-scenes-2020-mobile-development/audio',
                        help='dir with audio files')
    parser.add_argument('--folds_dir', '--folds_dir', type=str,
                        default='../dataset/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/',
                        help='dir with folds csv files')
    parser.add_argument('--dir_checkpoint', '--dir_checkpoint', type=str,
                        default='checkpoints', help='dir to save best nets during training')
    parser.add_argument('--n_classes', '--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--gpu', '--gpu', type=str, default=0, help='gpu to use')
    parser.add_argument('--backbone', '--backbone', type=str, default='resnet50', help='backbone for CNN classifier')
    return parser.parse_args()


if __name__ == '__main__':
    # load args
    args = get_args()
    if not osp.exists('outputs'):
        os.mkdir('outputs')
    if not osp.exists(args.dir_checkpoint):
        os.mkdir(args.dir_checkpoint)
    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp = f'{dt_object.year}_{dt_object.month}_{dt_object.day}_{dt_object.hour}_{dt_object.minute}'
    # print args to log
    print(args)
    print(args, file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'w+'))
    # choose device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # print device to log
    print(f'Using device {device}')
    print(f'Using device {device}', file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))
    # init net
    # net = ClassifierModule(args.backbone)
    net = MLP()
    print(net, file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))
    net.to(device=device)
    # load checkpoint weights
    if args.weights:
        net.load_state_dict(torch.load(args.weights, map_location=device))
    # start training
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  data_dir=args.data_dir,
                  folds_dir=args.folds_dir,
                  dir_checkpoint=args.dir_checkpoint,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  n_classes=args.n_classes,
                  timestamp=timestamp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), osp.join(args.dir_checkpoint, 'INTERRUPTED.pth'))
        print()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print('training finished')
    print('training finished', file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))
