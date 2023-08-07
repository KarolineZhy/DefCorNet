import torch
import torch.nn as nn
import torch.nn.functional as F

def convrelu(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, normalize=False):
    if normalize:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.InstanceNorm2d(in_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )

class FeatureExtractor(nn.Module):
    """Feature Extractor for different level of image"""
    def __init__(self, channels=[64,128], normalize=False):
        super(FeatureExtractor).__init__()
        self.pconv1_1 = convrelu(3, channels[0], 3, 2, normalize=normalize)
        self.pconv1_2 = convrelu(channels[0], channels[0], 3, 1, normalize=normalize)
        self.pconv2_1 = convrelu(channels[0], channels[1], 3, 2, normalize=normalize)
        self.pconv2_2 = convrelu(channels[1], channels[1], 3, 1, normalize=normalize)
        self.pconv2_3 = convrelu(channels[1], channels[1], 3, 1, normalize=normalize)
        self.pconv3_1 = convrelu(channels[1], channels[2], 3, 2, normalize=normalize)
        self.pconv3_2 = convrelu(channels[2], channels[2], 3, 1, normalize=normalize)
        self.pconv3_3 = convrelu(channels[2], channels[2], 3, 1, normalize=normalize)

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
     
        f1 = self.pconv1_2(self.pconv1_1(x)) 
        f2 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f1))) 
        f3 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f2)))
        f4 = self.pool(f3)
        f5 = self.pool(f4)
        f6 = self.pool(f5)

        return f1, f2, f3, f4, f5, f6 

class DownSampler(nn.Module):
    def __init__(self):
        super(DownSampler).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))

    def forward(self, x):
        
        f1 = self.pool(x)
        f2 = self.pool(f1)
        
        return f1, f2

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, groups):
        super(Decoder, self).__init__()
        assert hidden_channels % groups == 0, 'hidden_channels must be divisible by groups'
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, hidden_channels, 3, 1)
        self.conv2 = convrelu(hidden_channels, hidden_channels, 3, 1, groups=groups)
        self.conv3 = convrelu(hidden_channels, hidden_channels, 3, 1, groups=groups)
        self.conv4 = convrelu(hidden_channels, hidden_channels, 3, 1, groups=groups)
        self.conv5 = convrelu(hidden_channels, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)

    def forward(self, x):
        if self.groups == 1:
            out = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        else:
            out = self.conv1(x)
            out = channel_shuffle(self.conv2(out), self.groups)
            out = channel_shuffle(self.conv3(out), self.groups)
            out = channel_shuffle(self.conv4(out), self.groups)
            out = self.conv6(self.conv5(out))
        return out

def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, 3, padding=1)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x