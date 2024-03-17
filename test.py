import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

# from networks.BANet.encoding.models.BANet import BANet
# from networks.GCN import GCN
# from networks.newnet2 import Unet
# from networks.DCSAUNet.DCSAU_Net import DCSAUNet
# from networks.unetcl import Unet
from networks.mynet3 import Mynet3
# from networks.UNet_2Plus.models.UNet_2Plus import UNet_2Plus
BATCHSIZE_PER_CARD = 8

device = ('cuda' if torch.cuda.is_available() else 'cpu')
class TTAFrame():
    def __init__(self, net):
        self.net = net().to(device)
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        self.net.eval()
        maska = self.net.forward(img1)[0].squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2)[0].squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3)[0].squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4)[0].squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).to(device))

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        # dict =torch.load(path,map_location='cpu')
        # save_model = dict
        # model_dict = self.net.state_dict()
        # # we only need to load the parameters of the encoder
        # load_key, no_load_key, temp_dict = [], [], {}
        # for k, v in save_model.items():
        #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        #         temp_dict[k] = v
        #         load_key.append(k)
        #     else:
        #         no_load_key.append(k)
        # model_dict.update(temp_dict)
        # self.net.load_state_dict(model_dict, strict=False)
        # # ------------------------------------------------------#
        # #   显示没有匹配上的Key
        # # ------------------------------------------------------#
        # print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        # print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        self.net.load_state_dict(torch.load(path))




source = '/home/pyw/jieduan2/dataset/test8/img/'  #测试图片位置-需要修改
val = os.listdir(source)
solver = TTAFrame(Mynet3)#测试时所用网络
# solver = TTAFrame(DCSAUNet)
solver.load('weights1/dunet1.th')#测试时所用模型-修改
tic = time()
target = 'submits4/dunet1-8/'#测试后图片保存位置
if not os.path.exists(target):
    os.mkdir(target)
for i, name in enumerate(val):
    if i % 10 == 0:
        print(i / 10, '    ', '%.2f' % (time() - tic))
    mask = solver.test_one_img_from_path(
        source + name)  # 这里好奇不？别激动，作者这里的意图应该是类似于归一化数据的，你们可以自己写一个不那么麻烦的，我还没时间试，先用这个吧，这种方式值得探究一下的
    mask[mask > 4] = 255
    mask[mask <= 4] = 0
    # mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    # cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))
    cv2.imwrite(target + name[:-4] + '.png', mask.astype(np.uint8))
