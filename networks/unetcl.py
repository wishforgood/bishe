
import torch
import torch.nn as nn
from torch.autograd import Variable as V

from functools import reduce




import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)










class Unet(nn.Module):
    def __init__(self,in_channels=3, initial_filter_size=32, kernel_size=3, do_instancenorm=True):
        super(Unet, self).__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=True)
        self.contr_1_2 = self.contract2(initial_filter_size, initial_filter_size, kernel_size,
                                       instancenorm=do_instancenorm, )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract2(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
                                       instancenorm=do_instancenorm)


        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract2(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
                                       instancenorm=do_instancenorm)



        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract2(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
                                       instancenorm=do_instancenorm)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upscale5 = nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, kernel_size=2,
                                           stride=2)
        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand2(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)

        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand2(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)

        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand2(initial_filter_size * 2, initial_filter_size * 2)

        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand2(initial_filter_size, initial_filter_size)

        self.head = nn.Sequential(
            nn.Conv2d(initial_filter_size, 1, kernel_size=1,
                      stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        # contr_1 = self.residual_block1(self.contr_1_1(x))
        pool1 = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool1))
        # contr_2 = self.residual_block2(self.contr_2_1(pool1))
        pool2 = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool2))
        # contr_3 = self.residual_block3(self.contr_3_1(pool2))
        pool3 = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool3))
        # contr_4 = self.residual_block4(self.contr_4_1(pool3))
        pool4 = self.pool1(contr_4)

        out = self.center(pool4)

        concat_weight =1
        upscale = self.upscale5(out)
        crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))

        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))

        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))

        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))


        out = self.head(expand)


        return out



    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True,dilation=1):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels),  # BatchNorm2d
                nn.LeakyReLU(inplace=True)  # LeakyReLU
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def contract2(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),

            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.LeakyReLU(inplace=True))
        return layer


    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return layer

    @staticmethod
    def expand2(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        return layer

net = Unet()
# print(net)

if __name__ == '__main__':
    aa=torch.rand(4,3,512,512)
    model = Unet(in_channels=3, initial_filter_size=32, kernel_size=3, do_instancenorm=True)
    bb= model(aa)