import torch.nn as nn
import torchvision.models as models


class ClassifierModule10(nn.Module):
    def __init__(self, backbone):
        super(ClassifierModule10, self).__init__()
        self.net_low_freq = models.__dict__[backbone](pretrained=False)
        self.net_high_freq = models.__dict__[backbone](pretrained=False)
        self.fc_class_vec = nn.Linear(1000, 10)

    def forward(self, x):
        # pass through net
        n_freq = x.shape[2]
        x_low_freq = self.net_low_freq(x[:, :, :int(n_freq / 2), :])
        x_high_freq = self.net_high_freq(x[:, :, int(n_freq / 2):, :])
        # fc to 10 labels
        y = self.fc_class_vec((x_low_freq + x_high_freq) / 2)
        return y


class ClassifierModule3(nn.Module):
    def __init__(self, backbone):
        super(ClassifierModule3, self).__init__()
        self.net = models.__dict__[backbone](pretrained=False)
        self.fc_class_vec = nn.Linear(1000, 3)

    def forward(self, x):
        # pass through net
        x = (self.net(x[:, :3, :, :]) + self.net(x[:, 3:, :, :])) / 2
        # fc to 10 labels
        y = self.fc_class_vec(x)
        return y


class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Dropout2d(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 100)),
            nn.Dropout2d(0.3)
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=128, out_features=100),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(in_features=100, out_features=10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

# class ResNet_17_ASC(nn.Module):
#     def __init__(self, use_relu):
#         super(ResNet_17_ASC, self).__init__()
#         self.resnet_layer1 = nn.Sequential(
#             nn.BatchNorm2d(),
#             if use_relu:
#                 nn.ReLU()
#             nn.Conv2d(in_channels=num_filters,out_channels=, kernel_size=kernel_size,stride=strides,padding='same',bias=False)
#         )
#
#         self.resnet_layer2 = nn.Sequential(
#             nn.BatchNorm2d(),
#         if use_relu:
#             nn.ReLU()
#         nn.Conv2d(in_channels=num_filters, out_channels=, kernel_size=kernel_size, stride=strides, padding='same',
#                   bias=False)
#         )

#
#
# def forward(self, x):
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = x.view(x.size(0), -1)
#     x = self.dense(x)
#     return x
