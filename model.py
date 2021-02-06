import torch.nn as nn
import torchvision.models as models


class ClassifierModule(nn.Module):
    def __init__(self, backbone):
        super(ClassifierModule, self).__init__()
        self.net = models.__dict__[backbone](pretrained=False)
        self.fc_class_vec = nn.Linear(1000, 10)

    def forward(self, x):
        # pass through net
        x1 = self.net(x)
        # fc to 10 labels
        y = self.fc_class_vec(x1)
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
