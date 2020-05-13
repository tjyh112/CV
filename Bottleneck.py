import torch.nn as nn

class Bottleneck(nn.Module):

    def __init__(self, in_d, out_d, stride=1):

        super(Bottleneck, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_d, in_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d, in_d, kernel_size=3, stride=stride, padding=1, bias=False),
            nn. BatchNorm2d(in_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d, out_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_d)
        )

        self.downsample = nn.Sequential(nn.Conv2d(in_d, out_d, 1, 1),
                                        nn.BatchNorm2d(out_d)
                                        )

        self.relu=nn.ReLU(inplace=True)


    def forward(self, x):

        temp = x
        x = self.downsample(x)
        output = self.relu(self.net(temp) + x)

        return output

