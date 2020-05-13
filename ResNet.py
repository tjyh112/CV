import torch.nn as nn
import torch.nn.functional as F
import torch
import Bottleneck


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=7, stride=2) #in_channal=1, out_channal=6, kernal_size=5, stride=1
        self.pool_init = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bottleneck1 = Bottleneck(64, 256)

        self.pool_end = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256 * 1 * 1, 6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool_init(self.conv1(x))

        x = self.bottleneck1(x)

        x = self.pool_end(x)

        x = self.relu(self.fc(x))


        return x
