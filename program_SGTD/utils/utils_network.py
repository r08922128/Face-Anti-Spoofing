import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cal_gradient(input, gradient_type = 'x'):
	if gradient_type == 'x':
		sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = torch.float32).cuda()
	elif gradient_type == 'y':
		sobel_kernel = torch.tensor([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype = torch.float32).cuda()

	in_channels = input.shape[1]
	kernel_size = sobel_kernel.shape[0]

	sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size)
	# depthwise convolution needs sobel_kernel.shape[1] == in_channels / groups
	sobel_kernel = sobel_kernel.expand(in_channels, 1, kernel_size, kernel_size)
	# when groups == in_channels: conv2d equals to depthwise convolution
	gradient = F.conv2d(input, weight = sobel_kernel, padding = kernel_size // 2, groups = in_channels)
	return gradient

class Residual_Gradient_Conv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Residual_Gradient_Conv, self).__init__()
		self.smooth = 1e-8
		self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 3 // 2)
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 1 // 2)
		self.bn_1 = nn.BatchNorm2d(out_channels)
		self.bn_2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(True)

	def forward(self, input):
		features = self.conv3x3(input)
		gradient_x = cal_gradient(input, gradient_type = 'x')
		gradient_y = cal_gradient(input, gradient_type = 'y')
		gradient = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + self.smooth)
		gradient = self.conv1x1(gradient)
		gradient = self.bn_1(gradient)
		features = features + gradient
		features = self.bn_2(features)
		features = self.relu(features)
		return features

class Residual_Gradient_Block(nn.Module):
	def __init__(self, in_channels, multiplier = 2, first = False):
		super(Residual_Gradient_Block, self).__init__()
		self.blocks = nn.Sequential(
			Residual_Gradient_Conv(in_channels * (1 if first else 2), 64 * multiplier),
			Residual_Gradient_Conv(64 * multiplier, 96 * multiplier),
			Residual_Gradient_Conv(96 * multiplier, 64 * multiplier),
		)
		self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

	def forward(self, input):
		pre_pool_features = self.blocks(input)
		features = self.maxpool(pre_pool_features)
		return features, pre_pool_features

class Pool_Block(nn.Module):
	def __init__(self, in_channels = 128, reduc_num = 32, frame_len = 11):
		super(Pool_Block, self).__init__()
		self.reduc_num = reduc_num
		self.frame_len = frame_len
		self.conv1x1 = nn.Conv2d(in_channels, reduc_num, kernel_size = 1, padding = 1 // 2)
		self.bn = nn.BatchNorm2d(reduc_num)
		self.conv3x3 = nn.Conv2d(288, in_channels, kernel_size = 3, padding = 3 // 2)
		self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

	def forward(self, input):
		batchsize, channels, height, weight = input.shape
		batchsize = batchsize // self.frame_len
		feature = self.conv1x1(input)
		feature = self.bn(feature)
		feature_reshape = feature.view(batchsize, self.reduc_num, self.frame_len, height, weight)
		gradient_x = cal_gradient(feature, 'x')
		gradient_x = gradient_x.view(batchsize, self.reduc_num, self.frame_len, height, weight)
		gradient_y = cal_gradient(feature, 'y')
		gradient_y = gradient_y.view(batchsize, self.reduc_num, self.frame_len, height, weight)
		temporal_gradient = feature_reshape[:, :, :-1, :, :] - feature_reshape[:, :, 1:, :, :]
		pre_pool_features = input.view(batchsize, -1, self.frame_len, height, weight)
		scale = 1.0
		pool_features = torch.cat([
			pre_pool_features[:, :, :-1, :, :] * scale,
			gradient_x[:, :, :-1, :, :],
			gradient_y[:, :, :-1, :, :],
			gradient_x[:, :,  1:, :, :],
			gradient_y[:, :,  1:, :, :],
			temporal_gradient
		], dim = 1)
		pool_features_batch = pool_features.view(-1, pool_features.shape[1], height, weight)
		res_features = self.conv3x3(pool_features_batch)
		res_features = self.maxpool(res_features)
		return res_features

