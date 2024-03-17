import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial



nonlinearity = partial(F.relu,inplace=True)


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        # self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        #self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=32, padding=32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        # dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        #dilate6_out = nonlinearity(self.dilate6(dilate5_out))
        out = dilate1_out + dilate2_out  + dilate3_out# + dilate4_out #+ dilate5_out + dilate6_out
        return out


class Mynet3(nn.Module):
    def __init__(self):
        super(Mynet3, self).__init__()

        self.down1 = self.conv_stage(3, 64)
        self.down2 = self.conv_stage(64, 128)
        self.down3 = self.conv_stage(128, 256)
        self.down4 = self.conv_stage(256, 512)

        self.dilate_center = Dblock(512)
        #self.center = self.conv_stage(512, 1024)
        # self.center_res = self.resblock(1024)

        self.up4 = self.conv_stage(1024, 512)
        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)

        self.trans4 = self.upsample(512, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):

        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )



    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):

        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))

        out = self.dilate_center(conv4_out)   #4,512,64,64  4,512,64,64
        # out = self.center_res(out)
        out = self.up4(torch.cat((out, conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))   #4,256,128,128    4,256,128,128
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))   #4,128,256,256   128,256,256
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))   #4,64,512,512    4,64,512,512

        out = self.conv_last(out)    #4,64,512,512   4,1,512,512


        return out

net = Mynet3()
print(net)