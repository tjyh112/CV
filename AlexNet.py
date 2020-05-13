import torch.nn as nn
import torch.nn.functional as F
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #in_channal=1, out_channal=6, kernal_size=5, stride=1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 87 * 57, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 87 * 57)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5) #in_channal=1, out_channal=6, kernal_size=5, stride=1
#         # self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(6 * 178*2 * 118*2, 6)
#
#
#     def forward(self, x):
#         # x = self.pool(F.relu(self.conv1(x)))
#         x = F.relu(self.conv1(x))
#         x = x.view(-1, 6 * 178*2 * 118*2)
#         x = self.fc1(x)
#
#         return x




# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5) #in_channal=1, out_channal=6, kernal_size=5, stride=1
#         self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(6 * 178 * 118, 120)
#         self.fc2 = nn.Linear(120, 6)
#
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 6 * 178 * 118)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x