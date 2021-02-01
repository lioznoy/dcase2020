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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(500 * 40,  100 * 40),
            nn.ReLU(),
            nn.Linear(100 * 40, 50 * 40),
            nn.ReLU(),
            nn.Linear(50 * 40, 10 * 40),
            nn.ReLU(),
            nn.Linear(10 * 40, 10)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x