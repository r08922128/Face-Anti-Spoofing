import torch.nn as nn
import torch
from torch.autograd import Function

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
	def __init__(self, channels = [3, 32, 64, 64]):
		super(Extractor, self).__init__()
		self.blocks = nn.Sequential(*[Multi_Conv(channels[i], channels[i + 1], True) for i in range(len(channels) - 1)])
		self.avgpool = nn.AdaptiveAvgPool3d(3)
		self.__initialize_weights()

	def forward(self, input):
		features = self.blocks(input)
		features = self.avgpool(features)
		features = features.view(input.shape[0], -1)
		return features

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

class Classifier(nn.Module):
	def __init__(self, in_channels = 64 * 3 * 3 * 3, hidden = 128, out_channels = 2):
		super(Classifier, self).__init__()
		self.classify = nn.Sequential(
			nn.Linear(in_channels, hidden),
			nn.ReLU(True),
			nn.Linear(hidden, out_channels),
		)
		self.__initialize_weights()

	def forward(self, input):
		result = self.classify(input)
		return result

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

class Discriminator(nn.Module):
	def __init__(self, in_channels = 64 * 3 * 3 * 3, hidden = 128, out_channels = 2):
		super(Discriminator, self).__init__()
		self.classify = nn.Sequential(
			nn.Linear(in_channels, hidden),
			nn.ReLU(True),
			nn.Linear(hidden, out_channels),
		)
		self.__initialize_weights()

	def forward(self, input, lambda_term):
		reverse_input = Gradient_Reverse_Layer.apply(input, lambda_term)
		result = self.classify(reverse_input)
		return result

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

class Gradient_Reverse_Layer(Function):
	@staticmethod
	def forward(ctx, input, lambda_term):
		ctx.lambda_term = lambda_term
		return input.view_as(input)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.lambda_term
		return output, None