import torch
import torch.nn as nn
from utils.utils_network import *
from torchvision import transforms

class FaceMapNet(nn.Module):
	def __init__(self, in_channels = 3, num_blocks = 3, frame_len = 11):
		super(FaceMapNet, self).__init__()
		self.frame_len = frame_len
		self.Init_RsGB = Residual_Gradient_Conv(in_channels, 64)
		self.RsGBs = nn.Sequential(
			*[Residual_Gradient_Block(in_channels = 64, multiplier = 2, first = (i == 0)) for i in range(num_blocks)]
		)
		self.transform = transforms.Resize((32, 32))
		self.concat_RsGB = nn.Sequential(
			Residual_Gradient_Conv(64 * 6, 64 * 2),
			Residual_Gradient_Conv(64 * 2, 64 * 1)
		)
		self.conv3x3 = nn.Conv2d(64 * 1, 1, kernel_size = 3, padding = 3 // 2)

	def forward(self, input):
		batchsize, channels, height, weight = input.shape
		batchsize = batchsize // self.frame_len
		pool_list, pre_pool_list = [], []
		feature = self.Init_RsGB(input)
		for blocks in self.RsGBs:
			feature, pre_pool_feature = blocks(feature)
			pool_list.append(feature)
			pre_pool_list.append(pre_pool_feature)
		feature1 = self.transform(pool_list[0])
		feature2 = self.transform(pool_list[1])
		feature = self.transform(feature)
		pool_concat = torch.cat([feature1, feature2, feature], dim = 1)
		feature = self.concat_RsGB(pool_concat)
		feature = self.conv3x3(feature)
		feature = feature.view(batchsize, self.frame_len, -1)
		return feature, pre_pool_list

class PoolNet(nn.Module):
	def __init__(self, in_channels = 128, out_channels = 128, num_blocks = 3, frame_len = 11):
		super(PoolNet, self).__init__()
		self.frame_len = frame_len
		self.out_channels = out_channels
		self.transform_64 = transforms.Resize((64, 64))
		self.transform_32 = transforms.Resize((32, 32))
		self.Pool_Blocks = nn.Sequential(
			*[Pool_Block() for i in range(num_blocks)]
		)
		self.conv3x3_1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size = 3, stride = 1, padding = 3 // 2)
		self.bn_1 = nn.BatchNorm2d(out_channels)
		self.conv3x3_2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size = 3, stride = 1, padding = 3 // 2)
		self.bn_2 = nn.BatchNorm2d(out_channels)
		self.conv3x3_3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 3 // 2)
		self.bn_3 = nn.BatchNorm2d(out_channels)

	def forward(self, pre_pool_list):
		batchsize, channels, height, weight = pre_pool_list[0].shape
		batchsize = batchsize // self.frame_len
		features = []
		for i in range(len(self.Pool_Blocks)):
			feature = self.Pool_Blocks[i](pre_pool_list[i])
			features.append(feature)

		concat_feature_1_2 = torch.cat((features[0], self.transform_64(features[1])), dim = 1)
		concat_feature_1_2 = self.conv3x3_1(concat_feature_1_2)
		concat_feature_1_3 = torch.cat((self.transform_32(concat_feature_1_2), self.transform_32(features[2])), dim = 1)
		concat_feature_1_3 = self.conv3x3_2(concat_feature_1_3)
		concat_feature_1_3 = self.bn_2(concat_feature_1_3)
		concat_feature = self.conv3x3_3(concat_feature_1_3)
		concat_feature = self.bn_3(concat_feature)
		concat_feature = concat_feature.view(batchsize, -1, self.frame_len - 1, height, weight)
		concat_feature = concat_feature.view(batchsize, self.frame_len - 1, -1)
		return concat_feature

class ConvLSTMNet(nn.Module):
	def __init__(self, in_channels = 128 * 128 * 8, hidden_size = 128, out_channels = 1024):
		super(ConvLSTMNet, self).__init__()
		self.gru = nn.GRU(
			input_size = in_channels,
			hidden_size = hidden_size,
			num_layers = 10,
			batch_first = True,
			dropout = 0.2,
			bidirectional = False
		)
		self.transform = nn.Linear(hidden_size, out_channels)

	def forward(self, input):
		features, h_n = self.gru(input, None)
		features = self.transform(features)
		return features

class Classifier(nn.Module):
	def __init__(self, in_channels = 1024, out_channels = 2):
		super(Classifier, self).__init__()
		self.classify = nn.Linear(in_channels, out_channels)

	def forward(self, input):
		return self.classify(input)


