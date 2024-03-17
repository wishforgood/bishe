import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from torchvision.models import vgg19


class FCN8s(nn.Module):
	def __init__(self):
		super(FCN8s, self).__init__()
        # num_classes 训练数据的类别 
        # 使用预训练好的vgg19网络作为基础网络
		model_vgg19 = vgg19(pretrained=True)
        # 不使用vgg19网络中的后面的adaptiveavgpool2d和linear层
		self.base_model = model_vgg19.features
        # 定义需要的额几个层操作
		self.relu = nn.ReLU(inplace=True)
		self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1 = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
		self.bn2 = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
		self.bn3 = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
		self.bn4 = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.conv_last = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())


		self.conv_last = nn.Sequential(
			nn.Conv2d(32, 1, 3, 1, 1),
			nn.Sigmoid()
		)
        # vgg19中maxpool2所在的层
		self.layers = {"4": "max_pool_1", "9": "maxpool_2",
                       "18": "maxpool_3", "27": "maxpool_4",
                       "36": "maxpool_5"}
 
	def forward(self, x):
		output = {}
		for name, layer in self.base_model._modules.items():
            # 从第一层开始获取图像的特征
			x = layer(x)
            # 如果是layer中指定的特征，那就保存到output中‘
			if name in self.layers:
				output[self.layers[name]] = x
		x5 = output["maxpool_5"]
		x4 = output["maxpool_4"]
		x3 = output["maxpool_3"]
 
        # 对图像进行相应转置卷积操作，逐渐将图像放大到原来大小
		score = self.relu(self.deconv1(x5))
		score = self.bn1(score + x4)
		score = self.relu(self.deconv2(score))
		score = self.bn2(score + x3)
		score = self.bn3(self.relu(self.deconv3(score)))
		score = self.bn4(self.relu(self.deconv4(score)))
		score = self.bn5(self.relu(self.deconv5(score)))
		score = self.conv_last(score)
		return score

