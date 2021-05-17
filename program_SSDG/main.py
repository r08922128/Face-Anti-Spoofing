import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import RealFakeDataloader, SessionDataloader, AllDataloader
from torch.utils.data import DataLoader
from model import Extractor, Classifier, Discriminator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys, csv
import numpy as np
from utils import cal_accuracy, write_csv, cal_AUC
from hard_triplet_loss import HardTripletLoss

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 50)
	parser.add_argument('--lr_step', type = int, default = 20)
	parser.add_argument('--h_degree', type = int, default = 200)
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	parser.add_argument('--test_data_path', type = str, default = '../oulu_npu_cropped/test')
	return parser.parse_args()

def get_Lambda(index, start_step, total_step):
	p = float(index + start_step) / total_step
	lambda_term = 2. / (1. + np.exp(-10 * p)) - 1
	return lambda_term

def prepare_features_labels(real_features, fake_features, fake_session_labels):
	real_labels = torch.tensor([[0]] * real_features.shape[0]).cuda()
	fake_s1_labels = torch.tensor([[1]] * sum(fake_session_labels == 0)).cuda()
	fake_s2_labels = torch.tensor([[2]] * sum(fake_session_labels == 1)).cuda()
	total_labels = torch.cat((real_labels, fake_s1_labels, fake_s2_labels), dim = 0)
	total_features = torch.cat((real_features, fake_features), dim = 0)
	return total_features, total_labels

def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_real_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 0)
		train_real_data = DataLoader(train_real_dataloader, batch_size = 6, num_workers = 3, shuffle = True, pin_memory = True)
		train_fake_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 1)
		train_fake_data = DataLoader(train_fake_dataloader, batch_size = 6, num_workers = 3, shuffle = True, pin_memory = True)
		train_data_len = min(len(train_real_data), len(train_fake_data))

		valid_dataloader = AllDataloader('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 6, num_workers = 3, shuffle = False, pin_memory = False)
		valid_data_len = len(valid_data)

		print('loading model...')
		ENet = Extractor(args)
		total_params = sum(p.numel() for p in ENet.parameters() if p.requires_grad)
		ENet.cuda().float()
		print(ENet)
		print("Total number of params = ", total_params)
		print()

		CNet = Classifier()
		total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
		CNet.cuda().float()
		print(CNet)
		print("Total number of params = ", total_params)
		print()

		DNet = Discriminator()
		total_params = sum(p.numel() for p in DNet.parameters() if p.requires_grad)
		DNet.cuda().float()
		print(DNet)
		print("Total number of params = ", total_params)
		print()

		save_path = './models/'
		os.makedirs(save_path, exist_ok = True)

		if args.load != -1:
			ENet.load_state_dict(torch.load(join(save_path, 'E_' + str(args.load) + '.ckpt')))
			CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))
			DNet.load_state_dict(torch.load(join(save_path, 'D_' + str(args.load) + '.ckpt')))

		optimizer_SGD = optim.SGD(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		scheduler_SGD = optim.lr_scheduler.StepLR(optimizer_SGD, step_size = train_data_len * args.lr_step, gamma = 0.1)

		# optimizer_Adam = optim.Adam(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-3, betas = (0.5, 0.9))		

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		TripletLoss = HardTripletLoss()
		TripletLoss.cuda()
		best_loss = 100.0
		
		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}/{}  (lr: {})'.format(epoch, args.epochs, scheduler_SGD.get_last_lr()[0]))
			# print('epoch: {}/{}'.format(epoch, args.epochs))
			ENet.train()
			CNet.train()
			DNet.train()
			total_classification_loss = total_real_domain_loss = total_asymmetricTripletLoss = 0
			start_step = epoch * train_data_len
			total_step = args.epochs * train_data_len

			for index, ((real_images, real_spoof_labels, real_session_labels, real_folder_name), (fake_images, fake_spoof_labels, fake_session_labels, fake_folder_name)) in enumerate(tqdm(zip(train_real_data, train_fake_data), total = train_data_len, ncols = 70, desc = 'Training')):
				real_images, real_spoof_labels, real_session_labels = real_images.to(device), real_spoof_labels.to(device), real_session_labels.to(device)
				fake_images, fake_spoof_labels, fake_session_labels = fake_images.to(device), fake_spoof_labels.to(device), fake_session_labels.to(device)
				images = torch.cat((real_images, fake_images), dim = 0)
				features = ENet(images)
				predict = CNet(features)
				classification_loss = CELoss(predict, torch.cat((real_spoof_labels, fake_spoof_labels), dim = 0))

				real_features = features[:real_images.shape[0]]
				fake_features = features[real_images.shape[0]:]
				real_discrimination = DNet(real_features, get_Lambda(index, start_step, total_step))
				real_domain_loss = CELoss(real_discrimination, real_session_labels)

				total_features, total_labels = prepare_features_labels(real_features, fake_features, fake_session_labels)
				asymmetricTripletLoss = TripletLoss(total_features, total_labels)
				total_classification_loss += classification_loss.item()
				total_real_domain_loss += real_domain_loss.item()
				total_asymmetricTripletLoss += asymmetricTripletLoss.item()

				total_loss = classification_loss + 3 * real_domain_loss + 5 * asymmetricTripletLoss
				optimizer_SGD.zero_grad()
				total_loss.backward()
				optimizer_SGD.step()
				scheduler_SGD.step()

			avg_classification_loss, avg_real_domain_loss, avg_asymmetricTripletLoss = total_classification_loss / train_data_len, total_real_domain_loss / train_data_len, total_asymmetricTripletLoss / train_data_len
			print('[Train] avg_classification_loss: {:.5f} avg_real_domain_loss: {:.5f} avg_asymmetricTripletLoss: {:.5f}'.format(avg_classification_loss, avg_real_domain_loss, avg_asymmetricTripletLoss))
			print()

			torch.save(ENet.state_dict(), join(save_path + 'E_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
			torch.save(CNet.state_dict(), join(save_path + 'C_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
			torch.save(DNet.state_dict(), join(save_path + 'D_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

			if avg_classification_loss <= 0.13:
				with torch.no_grad():
					ENet.eval()
					CNet.eval()
					total_classification_loss = 0
					pred_list, label_list, folder_name_list = [], [], []
					for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70, desc = 'Validate')):
						images, labels = images.to(device), labels.to(device)
						features = ENet(images)
						prediction = CNet(features)
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
		valid_dataloader = AllDataloader('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 6, num_workers = 3, shuffle = False, pin_memory = True)
		valid_data_len = len(valid_data)

		print('loading model...')
		ENet = Extractor(args)
		total_params = sum(p.numel() for p in ENet.parameters() if p.requires_grad)
		ENet.cuda().float()
		print(ENet)
		print("Total number of params = ", total_params)
		print()

		CNet = Classifier()
		total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
		CNet.cuda().float()
		print(CNet)
		print("Total number of params = ", total_params)
		print()

		save_path = './models/'
		ENet.load_state_dict(torch.load(join(save_path, 'E_' + str(args.load) + '.ckpt')))
		CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		with torch.no_grad():
			ENet.eval()
			CNet.eval()
			total_classification_loss = 0
			pred_list, label_list, folder_name_list = [], [], []
			for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70, desc = 'Validate')):
				images, labels = images.to(device), labels.to(device)
				features = ENet(images)
				prediction = CNet(features)
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
	test_dataloader = AllDataloader('test', args.test_data_path)
	test_data = DataLoader(test_dataloader, batch_size = 6, num_workers = 3, shuffle = False, pin_memory = True)
	print('loading model...')
	ENet = Extractor(args)
	total_params = sum(p.numel() for p in ENet.parameters() if p.requires_grad)
	ENet.cuda().float()
	print(ENet)
	print("Total number of params = ", total_params)
	print()

	CNet = Classifier()
	total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
	CNet.cuda().float()
	print(CNet)
	print("Total number of params = ", total_params)
	print()

	save_path = './models/'
	ENet.load_state_dict(torch.load(join(save_path, 'E_' + str(args.load) + '.ckpt')))
	CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.ckpt')))

	with torch.no_grad():
		ENet.eval()
		CNet.eval()
		pred_list, label_list, folder_name_list = [], [], []
		for index, (images, labels, folder_name) in enumerate(tqdm(test_data, ncols = 70, desc = 'Testing')):
			batch_images, batch_labels = images.to(device), labels.to(device)
			features = ENet(batch_images)
			prediction = CNet(features)

			for index in range(prediction.shape[0]):
				pred_list.append(prediction[index].unsqueeze(0))
				label_list.append(labels[index].unsqueeze(0))
				folder_name_list.append(folder_name[index])
				
		cal_AUC(pred_list, label_list, folder_name_list, './test_pred.csv', False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)
