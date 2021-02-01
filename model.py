import torch.nn as nn
import torchvision.models as models


class ClassifierModule(nn.Module):
    def __init__(self, backbone):
        super(ClassifierModule, self).__init__()
        self.net = models.__dict__[backbone](pretrained=True)
        self.fc_class_vec = nn.Linear(1000, 10)
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        # pass through net
        x1 = self.net(x)
        # fc to 10 labels
        y = self.fc_class_vec(x1)
        return y
