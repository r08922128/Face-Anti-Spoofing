import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import RealFakeDataloader_2D, AllDataloader_2D
from torch.utils.data import DataLoader
from networks import FaceMapNet, PoolNet, ConvLSTMNet, Classifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys, csv
import numpy as np
from utils.utils import cal_accuracy, write_csv, cal_AUC

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 100)
	parser.add_argument('--lr_step', type = int, default = 40)
	parser.add_argument('--beta', type = float, default = 0.4)
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	parser.add_argument('--test_data_path', type = str, default = '../oulu_npu_cropped/test')
	return parser.parse_args()

def get_final_features(args, features_f, rnn_features, train = True):
	alpha = 1 - args.beta
	real_final_features, fake_final_features = [], []
	if train:
		for index in range(1, len(features_f[0])):
			real_final_features.append((alpha * rnn_features[0][index - 1] + args.beta * features_f[0][index]).unsqueeze(0))
			fake_final_features.append((alpha * rnn_features[1][index - 1] + args.beta * features_f[1][index]).unsqueeze(0))
		real_final_features = torch.cat(real_final_features, dim = 0).unsqueeze(0)
		fake_final_features = torch.cat(fake_final_features, dim = 0).unsqueeze(0)
		final_features = torch.cat((real_final_features, fake_final_features), dim = 0)
	else:
		for index in range(1, len(features_f[0])):
			real_final_features.append((alpha * rnn_features[0][index - 1] + args.beta * features_f[0][index]).unsqueeze(0))
		real_final_features = torch.cat(real_final_features, dim = 0).unsqueeze(0)
		final_features = real_final_features
	return final_features

def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_real_dataloader = RealFakeDataloader_2D(args.mode, args.train_data_path, 0)
		train_real_data = DataLoader(train_real_dataloader, batch_size = 1, num_workers = 3, shuffle = True, pin_memory = True)
		train_fake_dataloader = RealFakeDataloader_2D(args.mode, args.train_data_path, 1)
		train_fake_data = DataLoader(train_fake_dataloader, batch_size = 1, num_workers = 3, shuffle = True, pin_memory = True)
		train_data_len = min(len(train_real_data), len(train_fake_data))

		valid_dataloader = AllDataloader_2D('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = False)
		valid_data_len = len(valid_data)

		print('loading model...')
		FNet = FaceMapNet()
		total_params = sum(p.numel() for p in FNet.parameters() if p.requires_grad)
		FNet.cuda().float()
		print(FNet)
		print("Total number of params = ", total_params)
		print()

		PNet = PoolNet()
		total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
		PNet.cuda().float()
		print(PNet)
		print("Total number of params = ", total_params)
		print()

		LNet = ConvLSTMNet()
		total_params = sum(p.numel() for p in LNet.parameters() if p.requires_grad)
		LNet.cuda().float()
		print(LNet)
		print("Total number of params = ", total_params)
		print()

		CNet = Classifier()
		total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
		CNet.cuda().float()
		print(CNet)
		print("Total number of params = ", total_params)
		print()

		save_path = './models/'
		os.makedirs(save_path, exist_ok = True)

		if args.load != -1:
			FNet.load_state_dict(torch.load(join(save_path, 'F_' + str(args.load) + '.ckpt')))
			PNet.load_state_dict(torch.load(join(save_path, 'P_' + str(args.load) + '.ckpt')))
			LNet.load_state_dict(torch.load(join(save_path, 'L_' + str(args.load) + '.ckpt')))
			CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))

		optimizer_SGD = optim.SGD(list(list(FNet.parameters()) + list(PNet.parameters()) + list(LNet.parameters()) + list(CNet.parameters())), lr = 1e-3, weight_decay = 5e-4, momentum = 0.9)
		scheduler_SGD = optim.lr_scheduler.StepLR(optimizer_SGD, step_size = train_data_len * args.lr_step, gamma = 0.1)

		# optimizer_Adam = optim.Adam(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-3, betas = (0.5, 0.9))		

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		best_loss = 100.0
		
		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}/{}  (lr: {})'.format(epoch, args.epochs, scheduler_SGD.get_last_lr()[0]))
			# print('epoch: {}/{}'.format(epoch, args.epochs))
			FNet.train()
			PNet.train()
			LNet.train()
			CNet.train()
			total_classification_loss = total_real_domain_loss = total_asymmetricTripletLoss = 0
			start_step = epoch * train_data_len
			total_step = args.epochs * train_data_len

			for index, ((real_images, real_spoof_labels, real_session_labels, real_folder_name), (fake_images, fake_spoof_labels, fake_session_labels, fake_folder_name)) in enumerate(tqdm(zip(train_real_data, train_fake_data), total = train_data_len, ncols = 70, desc = 'Training')):
				real_images, real_spoof_labels, real_session_labels = real_images.to(device), real_spoof_labels.to(device), real_session_labels.to(device)
				fake_images, fake_spoof_labels, fake_session_labels = fake_images.to(device), fake_spoof_labels.to(device), fake_session_labels.to(device)
				batch_size, frame_len, channels, height, weight = real_images.shape
				real_images = real_images.reshape(-1, channels, height, weight)
				batch_size, frame_len, channels, height, weight = fake_images.shape
				fake_images = fake_images.reshape(-1, channels, height, weight)
				images = torch.cat((real_images, fake_images), dim = 0)
				features_f, pre_pool_list = FNet(images)
				features_p = PNet(pre_pool_list)
				features_p = LNet(features_p)
				final_features = get_final_features(args, features_f, features_p)
				predict = CNet(final_features)
				predict = torch.mean(predict, dim = 1)
				classification_loss = CELoss(predict, torch.cat((real_spoof_labels, fake_spoof_labels), dim = 0))

				total_classification_loss += classification_loss.item()

				total_loss = classification_loss
				optimizer_SGD.zero_grad()
				total_loss.backward()
				optimizer_SGD.step()
				scheduler_SGD.step()

			avg_classification_loss = total_classification_loss / train_data_len
			print('[Train] avg_classification_loss: {:.5f}'.format(avg_classification_loss))
			print()

			torch.save(FNet.state_dict(), join(save_path + 'F_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
			torch.save(PNet.state_dict(), join(save_path + 'P_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
			torch.save(LNet.state_dict(), join(save_path + 'L_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
			torch.save(CNet.state_dict(), join(save_path + 'C_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

			if avg_classification_loss <= 0.10:
				with torch.no_grad():
					FNet.eval()
					PNet.eval()
					LNet.eval()
					CNet.eval()
					total_classification_loss = 0
					pred_list, label_list, folder_name_list = [], [], []
					for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70, desc = 'Validate')):
						images, labels = images.to(device), labels.to(device)
						batch_size, frame_len, channels, height, weight = images.shape
						images = images.reshape(-1, channels, height, weight)
						features_f, pre_pool_list = FNet(images)
						features_p = PNet(pre_pool_list)
						features_p = LNet(features_p)
						final_features = get_final_features(args, features_f, features_p, False)
						prediction = CNet(final_features)
						prediction = torch.mean(prediction, dim = 1)
						
						total_classification_loss += CELoss(prediction, labels)

						for index in range(prediction.shape[0]):
							pred_list.append(prediction[index].unsqueeze(0))
							label_list.append(labels[index].unsqueeze(0))
							folder_name_list.append(folder_name[index])

					avg_classification_loss = total_classification_loss / valid_data_len
					print('[Valid] avg_classification_loss: {:.5f}'.format(avg_classification_loss))
					print()

					cal_accuracy(pred_list, label_list)
					cal_AUC(pred_list, label_list, folder_name_list, './valid_pred.csv', True)

	elif args.mode == 'valid':
		valid_dataloader = AllDataloader_2D('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
		valid_data_len = len(valid_data)

		print('loading model...')
		FNet = FaceMapNet()
		total_params = sum(p.numel() for p in FNet.parameters() if p.requires_grad)
		FNet.cuda().float()
		print(FNet)
		print("Total number of params = ", total_params)
		print()

		PNet = PoolNet()
		total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
		PNet.cuda().float()
		print(PNet)
		print("Total number of params = ", total_params)
		print()

		LNet = ConvLSTMNet()
		total_params = sum(p.numel() for p in LNet.parameters() if p.requires_grad)
		LNet.cuda().float()
		print(LNet)
		print("Total number of params = ", total_params)
		print()

		CNet = Classifier()
		total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
		CNet.cuda().float()
		print(CNet)
		print("Total number of params = ", total_params)
		print()

		save_path = './models/'
		FNet.load_state_dict(torch.load(join(save_path, 'F_' + str(args.load) + '.ckpt')))
		PNet.load_state_dict(torch.load(join(save_path, 'P_' + str(args.load) + '.ckpt')))
		LNet.load_state_dict(torch.load(join(save_path, 'L_' + str(args.load) + '.ckpt')))
		CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		with torch.no_grad():
			FNet.eval()
			PNet.eval()
			LNet.eval()
			CNet.eval()
			total_classification_loss = 0
			pred_list, label_list, folder_name_list = [], [], []
			for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70, desc = 'Validate')):
				images, labels = images.to(device), labels.to(device)
				batch_size, frame_len, channels, height, weight = images.shape
				images = images.reshape(-1, channels, height, weight)
				features_f, pre_pool_list = FNet(images)
				features_p = PNet(pre_pool_list)
				features_p = LNet(features_p)
				final_features = get_final_features(args, features_f, features_p, False)
				prediction = CNet(final_features)
				prediction = torch.mean(prediction, dim = 1)
				total_classification_loss += CELoss(prediction, labels)

				for index in range(prediction.shape[0]):
					pred_list.append(prediction[index].unsqueeze(0))
					label_list.append(labels[index].unsqueeze(0))
					folder_name_list.append(folder_name[index])

			avg_classification_loss = total_classification_loss / valid_data_len
			print('[Valid] avg_classification_loss: {:.5f}'.format(avg_classification_loss))
			print()

			cal_accuracy(pred_list, label_list)
			cal_AUC(pred_list, label_list, folder_name_list, './valid_pred.csv', True)

def test(args, device):
	test_dataloader = AllDataloader_2D('test', args.test_data_path)
	test_data = DataLoader(test_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
	print('loading model...')
	FNet = FaceMapNet()
	total_params = sum(p.numel() for p in FNet.parameters() if p.requires_grad)
	FNet.cuda().float()
	print(FNet)
	print("Total number of params = ", total_params)
	print()

	PNet = PoolNet()
	total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
	PNet.cuda().float()
	print(PNet)
	print("Total number of params = ", total_params)
	print()

	LNet = ConvLSTMNet()
	total_params = sum(p.numel() for p in LNet.parameters() if p.requires_grad)
	LNet.cuda().float()
	print(LNet)
	print("Total number of params = ", total_params)
	print()

	CNet = Classifier()
	total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
	CNet.cuda().float()
	print(CNet)
	print("Total number of params = ", total_params)
	print()

	save_path = './models/'
	FNet.load_state_dict(torch.load(join(save_path, 'F_' + str(args.load) + '.ckpt')))
	PNet.load_state_dict(torch.load(join(save_path, 'P_' + str(args.load) + '.ckpt')))
	LNet.load_state_dict(torch.load(join(save_path, 'L_' + str(args.load) + '.ckpt')))
	CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))

	with torch.no_grad():
		FNet.eval()
		PNet.eval()
		LNet.eval()
		CNet.eval()
		pred_list, label_list, folder_name_list = [], [], []
		for index, (images, labels, folder_name) in enumerate(tqdm(test_data, ncols = 70, desc = 'Testing')):
			images, labels = images.to(device), labels.to(device)
			batch_size, frame_len, channels, height, weight = images.shape
			images = images.reshape(-1, channels, height, weight)
			features_f, pre_pool_list = FNet(images)
			features_p = PNet(pre_pool_list)
			features_p = LNet(features_p)
			final_features = get_final_features(args, features_f, features_p, False)
			prediction = CNet(final_features)
			prediction = torch.mean(prediction, dim = 1)

			for index in range(prediction.shape[0]):
				pred_list.append(prediction[index].unsqueeze(0))
				label_list.append(labels[index].unsqueeze(0))
				folder_name_list.append(folder_name[index])
				
		cal_AUC(pred_list, label_list, folder_name_list, './test_pred.csv', False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)
