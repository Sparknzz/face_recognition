import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c))

    def forward(self, x):
        return self.net(x)


class ConvBnPrelu(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride,
                   padding, groups=groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)

class DWResidual(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.net = self.net = nn.Sequential(
            ConvBnPrelu(in_c, out_c=groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(in_c=groups, out_c=groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn(in_c=groups, out_c=out_c, kernel=(1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x) + x


class MultiDwResidual(nn.Module):
    def __init__(self, num_blocks, channels, kernel=(3,3), stride=1, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(*[DWResidual(channels, channels, kernel, stride, padding, groups) for _ in range(num_blocks)])

    def forward(self, x):
        return self.net(x)



class FaceMobileNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.conv1 = ConvBnPrelu(in_c=3, out_c=64, kernel=(3,3), stride=2, padding=1)
        self.conv2 = ConvBn(in_c=64, out_c=64, kernel=(3,3), stride=1, padding=1, groups=64) # 1 * 3 * 3 * 64
        self.conv3 = DepthWise(in_c=64, out_c=64, kernel=(3,3), stride=2, padding=1, groups=128) # note here 64 to 128 channels then do depthwise conv

        # block 1
        self.conv4 = MultiDwResidual(num_blocks=4, channels=64, kernel=(3,3), stride=1, padding=1, groups=128) # depthwise residual
        self.conv5 = DepthWise(in_c=64, out_c=128, kernel=(3,3), stride=2, padding=1, groups=256)

        # block 2
        self.conv6 = MultiDwResidual(num_blocks=6, channels=128, kernel=(3,3), stride=1, padding=1, groups=256) # depthwise residual
        self.conv7 = DepthWise(in_c=128, out_c=128, kernel=(3,3), stride=2, padding=1, groups=512)

        # block 3
        self.conv8 = MultiDwResidual(num_blocks=2, channels=128, kernel=(3,3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128,512,kernel=(1,1))

        self.conv10 = ConvBn(512,512,groups=512,kernel=(7,7))
        
        self.flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        # print(out.shape)
        out = self.flatten(out)
        # print(out.shape)
        out = self.linear(out)
        out = self.bn(out)
        return out


if __name__ == "__main__":
    import numpy as np
    x = np.zeros((1, 3, 112, 112))
    x = torch.from_numpy(x).float()
    net = FaceMobileNet(512)
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)