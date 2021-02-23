import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


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
        x_low_freq = self.fc_class_vec(x_low_freq)
        x_high_freq = self.fc_class_vec(x_high_freq)
        # fc to 10 labels
        y = x_low_freq * 0.25 + x_high_freq * 0.75
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
        x_low_freq = self.fc_class_vec10(x_low_freq)
        x_high_freq = self.fc_class_vec10(x_high_freq)
        y10 = x_low_freq * 0.25 + x_high_freq * 0.75
        # y10 = (x_low_freq + x_high_freq)
        y3 = self.fc_class_vec3(x_3class)
        # y3 = x_3class
        return y10, y3


# class ClassifierModule3(nn.Module):
#     def __init__(self, backbone):
#         super(ClassifierModule3, self).__init__()
#         self.net = models.__dict__[backbone](pretrained=False)
#         self.fc_class_vec = nn.Linear(1000, 3)
#
#     def forward(self, x):
#         # pass through net
#         x = (self.net(x[:, :3, :, :]) + self.net(x[:, 3:, :, :])) / 2
#         # fc to 10 labels
#         y = self.fc_class_vec(x)
#         return y

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 6, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def resnet_layer(in_channels, out_channels, use_relu):
    if use_relu:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )


class ClassifierModule3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ClassifierModule3, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=2),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU())
        self.inv_res1 = InvertedResidual(32, 32, 2, 3)
        self.inv_res2 = InvertedResidual(32, 40, 2, 3)
        self.inv_res3 = InvertedResidual(40, 48, 2, 3)
        self.resnet1 = resnet_layer(96, 48, use_relu=True)
        self.resnet2 = resnet_layer(48, n_class, use_relu=False)
        self.dropout = nn.Dropout2d(0.3)
        self.bn = nn.BatchNorm2d(n_class)
        # building classifier

    def forward(self, x):
        n_freq = x.shape[2]
        split1 = x[:, :, 0:int(n_freq / 2), :]
        split2 = x[:, :, int(n_freq / 2):, :]
        split1 = self.conv_block1(split1)
        split1 = self.inv_res1(split1)
        split1 = self.inv_res2(split1)
        split1 = self.inv_res3(split1)
        split2 = self.conv_block1(split2)
        split2 = self.inv_res1(split2)
        split2 = self.inv_res2(split2)
        split2 = self.inv_res3(split2)
        MobilePath = torch.cat([split1, split2], dim=1)
        OutputPath = self.resnet1(MobilePath)
        OutputPath = self.dropout(OutputPath)
        OutputPath = self.resnet2(OutputPath)
        OutputPath = self.bn(OutputPath)
        OutputPath = F.avg_pool2d(OutputPath, kernel_size=(OutputPath.shape[2], OutputPath.shape[3]))
        return OutputPath.squeeze(3).squeeze(2)


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


class FCNNModel(nn.Module):
    def __init__(self, channels, num_filters=8, output_features=10):
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
            nn.Conv2d(in_channels=num_filters * 4 * channels, out_channels=output_features, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(output_features),
            nn.ReLU(),
            nn.BatchNorm2d(output_features)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=output_features, out_features=int(np.ceil(output_features / 2)), bias=True),
            nn.ReLU())

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=int(np.ceil(output_features / 2)), out_features=output_features, bias=True),
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
        return output.squeeze(3).squeeze(2)
