import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_network import UNet_orig, UNet, UNet_small
from utils.flow_util import warp
from utils.registry import NETWORK_REGISTRY

class DownSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        
    def forward(self, x):
        f1 = self.pool(x)
        f2 = self.pool(f1)
        return f1, f2

@NETWORK_REGISTRY.register()
class CorseToFineStiffSecondNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # get device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.downsampler = DownSampler()
        self.unet1 = UNet_small(in_channels, out_channels)
        # self.unet2 = UNet(in_channels, out_channels)
        # self.unet3 = UNet(in_channels, out_channels)

        # flow parameters
        self.x11 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.x12 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.x13 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y11 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y12 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y13 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))

        self.x21 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.x22 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        # self.x23 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y21 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y22 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        # self.y23 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        
        self.x31 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.x32 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        # self.x33 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y31 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.y32 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        # self.y33 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))

        # stiffness parameters
        self.stiff_x1 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_y1 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_z1 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))

        self.stiff_x2 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_y2 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_z2 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))

        self.stiff_x3 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_y3 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))
        self.stiff_z3 = nn.Parameter(torch.randn(1, requires_grad=True).to(self.device))

        # sigmoids
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()


    def forward(self, x):

        img = x[:,0:3,:,:]
        force = x[:,[3],:,:]
        stiff = x[:,[4],:,:]

        img_d2, img_d1 = self.downsampler(img)
        force_d2, force_d1 = self.downsampler(force)
        stiff_d2, stiff_d1 = self.downsampler(stiff)

        # stage one 
        mask_1 = self.unet1(img_d1)
        mask_1_unet = mask_1
        mask_1 = self.relu(mask_1)

        # stiff update
        stiff_1 = self.stiff_x1 * mask_1 + self.stiff_y1 * stiff_d1 + self.stiff_z1
        stiff_1_update = stiff_1
        stiff_1 = self.relu(stiff_1)

        # flow estimation
        dx =  force_d1 * (self.x11 * stiff_1 * stiff_1 + self.x12 * stiff_1 + self.x13)
        dy =  force_d1 * (self.y11 * stiff_1 * stiff_1 + self.y12 * stiff_1 + self.y13)
        flow_1 = torch.cat((dx, dy), dim=1)

        # stage two
        flow_1_up = F.interpolate(flow_1, scale_factor=2, mode='bilinear') * 2
        img_d2_warp = warp(img_d2, flow_1_up)
        mask_2 = self.unet1(img_d2_warp)
        mask_2_unet = mask_2
        mask_2 = self.relu(mask_2)
        
        # stiff update
        stiff_2 = self.stiff_x2 * mask_2 + self.stiff_y2 * stiff_d2 + self.stiff_z2
        stiff_2_update = stiff_2
        stiff_2 = self.relu(stiff_2)

        # flow estimation
        dx =  force_d2 * (self.x21 * stiff_2 + self.x22)
        dy =  force_d2 * (self.y21 * stiff_2 + self.y22)
        flow_2 = torch.cat((dx, dy), dim=1)
        flow_2 = flow_2 + flow_1_up

        # stage three
        flow_2_up = F.interpolate(flow_2, scale_factor=2, mode='bilinear') * 2
        img_d3_warp = warp(img, flow_2_up)
        mask_3 = self.unet1(img_d3_warp)
        mask_3_unet = mask_3
        mask_3 = self.relu(mask_3)
        
        # stiff update
        stiff_3 = self.stiff_x3 * mask_3 + self.stiff_y3 * stiff + self.stiff_z3
        stiff_3_update = stiff_3
        stiff_3 = self.relu(stiff_3)

        # flow estimation
        dx =  force * (self.x31 * stiff_3 + self.x32)
        dy =  force * (self.y31 * stiff_3 + self.y32)
        flow_3 = torch.cat((dx, dy), dim=1)
        flow_3 = flow_3 + flow_2_up

        return [flow_1, flow_2, flow_3], [stiff_1, stiff_2, stiff_3, mask_1_unet, mask_2_unet, mask_3_unet, stiff_1_update, stiff_2_update,stiff_3_update ]
        





