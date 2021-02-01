from utils import plot_loss
import pandas as pd
import os.path as osp
from dataset import BasicDataset
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from eval import eval_net


def train_net(net, epochs, data_dir, folds_dir, dir_checkpoint, batch_size, lr, device, n_classes, timestamp):
    # load folds
    train_csv = osp.join(folds_dir, 'fold1_train.csv')
    val_csv = osp.join(folds_dir, 'fold1_evaluate.csv')
    train_df = pd.read_csv(train_csv, sep='\t')
    val_df = pd.read_csv(val_csv, sep='\t')

    # initialize data loaders
    dataset_train = BasicDataset(data_dir, train_df)
    dataset_val = BasicDataset(data_dir, val_df)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            drop_last=True)

    print(f'''\nStarting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {round(train_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
        Validation size: {round(val_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
        Checkpoints dir: {dir_checkpoint}
        Device:          {device.type}
    ''')
    print(f'''\nStarting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {round(train_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
        Validation size: {round(val_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
        Checkpoints dir: {dir_checkpoint}
        Device:          {device.type}
    ''', file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))

    # set optimizer
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr) #weight_decay=1e-5)
    # optimizer = torch.optim.AdamW(net.parameters(), lr = lr , betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    criterion = nn.CrossEntropyLoss()
    loss_val = []
    loss_train = []

    # start epochs
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        with tqdm(total=train_df.shape[0],
                  desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                mels = batch['mels']
                label = batch['label']
                mels = mels.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                pred_vec = net(mels)
                loss = criterion(pred_vec, label)
                epoch_loss.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                # nn.utils2.clip_grad_value_(net.parameters(), 0.25)
                optimizer.step()
                pbar.update(mels.shape[0])
            val_loss = eval_net(net, val_loader, device, criterion)
            loss_val.append((val_loss.item()))
            loss_train.append(sum(epoch_loss) / len(epoch_loss))
            torch.save(net.state_dict(),
                       osp.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            print(f'epoch = {epoch + 1}, train loss = {sum(epoch_loss) / len(epoch_loss)},'
                  f' validation loss = {val_loss}',
                  file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))
            print(f'saved weights to {osp.join(dir_checkpoint, f"CP_epoch_{epoch + 1}.pth")}',
                  file=open(osp.join('outputs', f'log_{timestamp}.txt'), 'a'))

    plot_loss(np.arange(1, epochs + 1).astype(int), loss_val, loss_train, timestamp)







