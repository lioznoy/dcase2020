import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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
        y = self.fc_class_vec(x_low_freq * 0.25 + x_high_freq * 0.75)
        return y


class ClassifierModule10_2path(nn.Module):
    def __init__(self, backbone10, backbone3):
        super(ClassifierModule10_2path, self).__init__()
        self.net_low_freq = models.__dict__[backbone10](pretrained=False)
        self.net_high_freq = models.__dict__[backbone10](pretrained=False)
        self.class3 = models.__dict__[backbone3](pretrained=False)
        self.fc_class_vec10 = nn.Linear(1000, 10)
        self.fc_class_vec3 = nn.Linear(1000, 3)

    def forward(self, x):
        # pass through net
        n_freq = x.shape[2]
        x_low_freq = self.net_low_freq(x[:, :, :int(n_freq / 2), :])
        x_high_freq = self.net_high_freq(x[:, :, int(n_freq / 2):, :])
        x_3class = self.class3(x)
        y10 = self.fc_class_vec10((x_low_freq + x_high_freq) / 2)
        y3 = self.fc_class_vec3(x_3class)
        return y10, y3


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

class FCNNModel(nn.Module):
    def __init__(self, channels, num_filters=14):
        super(FCNNModel, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(in_channels=channels, out_channels=num_filters * channels,
                      kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * channels),
            nn.ReLU(),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * channels, out_channels=num_filters * channels, kernel_size=3, stride=1,
                      padding=0, bias=True),
            nn.BatchNorm2d(num_filters * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        self.conv_layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * channels, out_channels=num_filters * 2 * channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * 2 * channels),
            nn.ReLU(),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 2 * channels, out_channels=num_filters * 2 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 2 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        self.conv_layer3 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 2 * channels, out_channels=num_filters * 4 * channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=num_filters * 4 * channels, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_filters * 4 * channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )

        self.resnet_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=10, out_features=5, bias=True),
            nn.ReLU())

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=5, out_features=10, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.resnet_layer(x)
        avg_pool = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        avg_pool = torch.reshape(avg_pool, (x.shape[0], 1, 1, x.shape[1]))
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        max_pool = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        max_pool = torch.reshape(max_pool, (x.shape[0], 1, 1, x.shape[1]))
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)
        cbam_feature = torch.add(avg_pool, max_pool)
        cbam_feature = torch.sigmoid(cbam_feature)
        cbam_feature = torch.reshape(cbam_feature, (cbam_feature.shape[0], cbam_feature.shape[3], 1, 1))
        cbam_feature = torch.multiply(x, cbam_feature)
        output = F.avg_pool2d(cbam_feature, kernel_size=(x.shape[2], x.shape[3]))
        # output = F.softmax(output, dim=1)
        return output
