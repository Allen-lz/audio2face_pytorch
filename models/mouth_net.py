import torch.nn as nn
import torch.nn.functional as F
import torch

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class MouthNet(nn.Module):
    def __init__(self, class_num=16):
        super(MouthNet, self).__init__()

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        encoder1 = []
        layer1 = []
        layer1.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=0))
        layer1.append(nn.BatchNorm2d(32))
        layer1.append(Swish())
        layer1 = nn.Sequential(*layer1)
        encoder1.append(layer1)

        layer2 = []
        layer2.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), stride=(4, 1), padding=0))
        layer2.append(nn.BatchNorm2d(64))
        layer2.append(Swish())
        layer2 = nn.Sequential(*layer2)
        encoder1.append(layer2)

        layer3 = []
        layer3.append(nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 1), stride=(4, 1), padding=0))
        layer3.append(nn.BatchNorm2d(96))
        layer3.append(Swish())
        layer3 = nn.Sequential(*layer3)
        encoder1.append(layer3)

        self.encoder1 = nn.Sequential(*encoder1)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        encoder2 = []
        layer1 = []
        layer1.append(nn.Conv2d(in_channels=97, out_channels=128, kernel_size=(1, 3), stride=(1, 3), padding=0))
        layer1.append(nn.BatchNorm2d(128))
        layer1.append(Swish())
        layer1 = nn.Sequential(*layer1)
        encoder2.append(layer1)

        layer2 = []
        layer2.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 2), padding=0))
        layer2.append(nn.BatchNorm2d(128))
        layer2.append(Swish())
        layer2 = nn.Sequential(*layer2)
        encoder2.append(layer2)

        layer3 = []
        layer3.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 2), padding=0))
        layer3.append(nn.BatchNorm2d(128))
        layer3.append(Swish())
        layer3 = nn.Sequential(*layer3)
        encoder2.append(layer3)

        self.encoder2 = nn.Sequential(*encoder2)

        regression = []
        regression.append(nn.Linear(128, 64))
        regression.append(nn.Dropout(0.5))
        regression.append(nn.BatchNorm1d(64))
        regression.append(Swish())
        regression.append(nn.Linear(64, class_num))
        self.regression = nn.Sequential(*regression)


    def forward(self, x):
        # 前向传播
        feat_zc = x[:, :, 0, :].unsqueeze(-2)
        feat = x[:, :, 1:, :]
        encoder1 = self.encoder1(feat)
        x2 = torch.cat([encoder1, feat_zc], 1)
        encoder2 = self.encoder2(x2)
        encoder2 = torch.flatten(encoder2, 1)

        bs = self.regression(encoder2)

        return bs


if __name__ == "__main__":
    print('##############PyTorch################')
    net = MouthNet(class_num=16)
    x = torch.randn((8, 1, 33, 13))
    y = net(x)