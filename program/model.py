import torch.nn as nn
import torch

class Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, downsample):
		super(Conv_Block, self).__init__()
		self.conv = nn.Sequential(
						nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 2 if downsample else 1, padding = kernel_size // 2),
						nn.BatchNorm3d(out_channels),
						nn.ReLU(True),
					)

	def forward(self, input):
		return self.conv(input)

class Multi_Conv(nn.Module):
	def __init__(self, in_channels, out_channels, downsample = False):
		super(Multi_Conv, self).__init__()
		self.residual_conv = Conv_Block(in_channels, out_channels, 1, downsample)
		self.conv1 = Conv_Block(in_channels, out_channels, 1, downsample)
		self.conv3 = Conv_Block(in_channels, out_channels, 3, downsample)
		self.conv5 = Conv_Block(in_channels, out_channels, 5, downsample)
		self.conv_cat = Conv_Block(out_channels * 3, out_channels, 3, False)

	def forward(self, input):
		residual = self.residual_conv(input)
		conv1 = self.conv1(input)
		conv3 = self.conv3(input)
		conv5 = self.conv5(input)
		mix_conv = torch.cat((conv1, conv3, conv5), dim = 1)
		output = self.conv_cat(mix_conv) + residual
		return output

class Extractor(nn.Module):
	def __init__(self, channels):
		super(Extractor, self).__init__()
		self.blocks = nn.Sequential(*[Multi_Conv(channels[i], channels[i + 1], True) for i in range(len(channels) - 1)])
		self.avgpool = nn.AdaptiveAvgPool3d(3)

	def forward(self, input):
		features = self.blocks(input)
		features = self.avgpool(features)
		features = features.view(input.shape[0], -1)
		return features

class MLP(nn.Module):
	def __init__(self, in_channels = 64 * 3 * 3 * 3, hidden = 128, out_channels = 2):
		super(MLP, self).__init__()
		self.classify = nn.Sequential(
			nn.Linear(in_channels, hidden),
			nn.ReLU(True),
			nn.Linear(hidden, out_channels),
		)

	def forward(self, input):
		result = self.classify(input)
		return result

class Classifier(nn.Module):
	def __init__(self, channels = [3, 32, 64, 64]):
		super(Classifier, self).__init__()
		self.encoder = Extractor(channels)
		self.classify = MLP()
		self.__initialize_weights()

	def forward(self, input):
		features = self.encoder(input)
		predict = self.classify(features)
		return predict

	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight.data.normal_(0, 1)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 1)
				if m.bias is not None:
					m.bias.data.zero_()
