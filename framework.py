import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.optim import lr_scheduler

import cv2
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')
class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().to(device)

        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)

        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):  # 注释了一部分的
        self.net.eval()
        with torch.no_grad():
        # pred[pred>0.5] = 1
        # pred[pred<=0.5] = 0
            pred = self.net.forward(img)
            # mask = pred.squeeze().cpu().data.numpy()
            mask = pred[0].squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).to(device))

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.to(device), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.to(device), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        # loss1 = self.loss(self.mask, pred[1])
        # loss = loss + loss1
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        dict = torch.load(path)
        save_model = dict["net"]  # moco-cl-pix
        # save_model = dict["state_dict"]
        model_dict = self.net.state_dict()
        # we only need to load the parameters of the encoder
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in save_model.items():
            # if "encoder_q" in k:                #moco-cl
            #     temp_dict[k.replace('encoder_q.', '')] = v
            #     load_key.append(k)
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.net.load_state_dict(model_dict, strict=False)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        # self.net.load_state_dict(torch.load(path))

    def loads(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def lr_strategy(self):  # 新加的
        # scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # scheduler = lr_scheduler.MultiStepLR(self.optimizer, [30, 80], 0.1)
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        return scheduler

