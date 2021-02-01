from tqdm import tqdm
import torch

def eval_net(net, val_loader, device, criterion):
    net.eval()
    n_val = len(val_loader)
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in val_loader:
            mels = batch['mels']
            label = batch['label']
            mels = mels.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_vec = net(mels)

            tot += criterion(pred_vec, label)
            pbar.update()

        net.train()
        return tot / n_val