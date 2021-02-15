from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils import vec3_to_vec10

def eval_net(net, val_loader, device, criterion, n_classes):
    net.eval()
    n_val = len(val_loader)
    tot_loss = 0
    tot_score = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in val_loader:
            mels = batch['mels']
            label = batch['label']
            mels = mels.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            _label = label
            label = F.one_hot(label, n_classes)
            with torch.no_grad():
                if n_classes == 10:
                    pred_vec10, pred_vec3 = net(mels)
                    pred_vec = pred_vec10 * vec3_to_vec10(pred_vec3, device)
                else:
                    pred_vec = net(mels)
            tot_loss += criterion(pred_vec, label)
            tot_score += accuracy_score(pred_vec.argmax(dim=1).flatten().cpu().numpy(), _label.flatten().cpu().numpy())
            pbar.update()

        net.train()
        return tot_loss / n_val, tot_score / n_val