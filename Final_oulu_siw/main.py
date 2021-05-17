import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import RealFakeDataloader, SessionDataloader, AllDataloader
from torch.utils.data import DataLoader
#from model_1 import Extractor, Classifier, Discriminator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys, csv
import numpy as np
from utils import cal_accuracy, write_csv, cal_AUC
from hard_triplet_loss import HardTripletLoss
from DGFAS_norm_after import *
from config import config
def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--data', type = str,default=None)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 40)
	parser.add_argument('--lr_step', type = int, default = 10)
	parser.add_argument('--h_degree', type = int, default = 5)
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	parser.add_argument('--test_data_path', type = str, default = '../siw_test')
	return parser.parse_args()

def get_Lambda(index, start_step, total_step):
	p = float(index + start_step) / total_step
	lambda_term = 2. / (1. + np.exp(-10 * p)) - 1
	return lambda_term

def prepare_noise_features_labels(args, real_noise_features, fake_noise_features, fake_session_labels):
	features_len = real_noise_features.shape[-1]
	real_labels = torch.tensor([[0]] * real_noise_features.shape[0] * args.h_degree).cuda()
	fake_session_labels = (fake_session_labels + 1).unsqueeze(-1).expand(-1, args.h_degree).reshape(-1, 1)
	real_noise_features = real_noise_features.view(-1, features_len)
	fake_noise_features = fake_noise_features.view(-1, features_len)
	total_labels = torch.cat((real_labels, fake_session_labels), dim = 0)
	total_features = torch.cat((real_noise_features, fake_noise_features), dim = 0)
	return total_features, total_labels

def prepare_features_labels(real_features, fake_features, fake_session_labels):
	real_labels = torch.tensor([[0]] * real_features.shape[0]).cuda()
	fake_session_labels = fake_session_labels + 1
	total_labels = torch.cat((real_labels, fake_session_labels.unsqueeze(-1)), dim = 0)
	total_features = torch.cat((real_features, fake_features), dim = 0)
	total_labels=total_labels.squeeze(-1)
	return total_features, total_labels
def one_dim_expand(label,h_degree):
	batch_size=label.size(0)
	label=label.unsqueeze(1)
	label=label.expand(-1,h_degree+1)
	label=label.reshape(-1)
	return label
def two_dim_expand(feature,h_degree):
	batch_size=feature.size(0)
	feature_size=feature.size(1)
	feature=feature.unsqueeze(2)
	feature=feature.expand(-1,-1,h_degree+1)
	feature=feature.reshape(-1,feature_size)
	return feature
def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_real_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 0)
		train_real_data = DataLoader(train_real_dataloader, batch_size = 24, num_workers = 4, shuffle = True)
		train_fake_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 1)
		train_fake_data = DataLoader(train_fake_dataloader, batch_size = 24, num_workers = 4, shuffle = True)
		train_data_len = min(len(train_real_data), len(train_fake_data))

		valid_dataloader = AllDataloader('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 11, num_workers = 0, shuffle = False)
		valid_data_len = len(valid_data)

		print('loading model...')
		ENet = DG_model('resnet18',args)
		total_params = sum(p.numel() for p in ENet.parameters() if p.requires_grad)
		ENet.cuda().float()
		# print(ENet)
		# print("Total number of params = ", total_params)
		# print()

		# CNet = Classifier()
		# total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
		# CNet.cuda().float()
		# print(CNet)
		# print("Total number of params = ", total_params)
		# print()

		DNet = Discriminator()
		total_params = sum(p.numel() for p in DNet.parameters() if p.requires_grad)
		DNet.cuda().float()
		# print(DNet)
		# print("Total number of params = ", total_params)
		# print()

		save_path = './models/'
		os.makedirs(save_path, exist_ok = True)

		if args.load != -1:
			ENet.load_state_dict(torch.load(join(save_path, 'E_SSDG_new_100.0_1.0_27.pth')))
			DNet.load_state_dict(torch.load(join(save_path, 'D_SSDG_new_100.0_1.0_27.pth')))

		optimizer_SGD = optim.SGD(list(list(ENet.parameters()) + list(DNet.parameters())), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		scheduler_SGD = optim.lr_scheduler.StepLR(optimizer_SGD, step_size = 10, gamma = 0.1)

		# optimizer_Adam = optim.Adam(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-3, betas = (0.5, 0.9))		

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		TripletLoss = HardTripletLoss()
		TripletLoss.cuda()
		MSELoss = nn.MSELoss()
		MSELoss.cuda()
		best_loss = 100.0
		
		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}/{}  (lr: {})'.format(epoch, args.epochs, scheduler_SGD.get_last_lr()[0]))
			# print('epoch: {}'.format(epoch))
			ENet.train()
			# CNet.train()
			DNet.train()
			total_classification_loss = total_real_domain_loss = total_gen_asymmetricTripletLoss = total_aug_asymmetricTripletLoss = total_reconLoss = 0
			start_step = epoch * train_data_len
			total_step = args.epochs * train_data_len
			total_DNet_output_session0=0.0
			total_DNet_output_session1=0.0
			for index, ((real_images, real_spoof_labels, real_session_labels, real_folder_name), (fake_images, fake_spoof_labels, fake_session_labels, fake_folder_name)) in enumerate(tqdm(zip(train_real_data, train_fake_data), total = train_data_len, ncols = 70, desc = 'Training')):

				real_images, real_spoof_labels, real_session_labels = real_images.to(device), real_spoof_labels.to(device), real_session_labels.to(device)
				fake_images, fake_spoof_labels, fake_session_labels = fake_images.to(device), fake_spoof_labels.to(device), fake_session_labels.to(device)
				images = torch.cat((real_images, fake_images), dim = 0)
				aug_predict, aug_features, clean_predict, clean_features= ENet(images, True)

				#classfication
				real_spoof_labels=one_dim_expand(real_spoof_labels,args.h_degree)
				fake_spoof_labels=one_dim_expand(fake_spoof_labels,args.h_degree)
				classification_loss = CELoss(aug_predict, torch.cat((real_spoof_labels, fake_spoof_labels), dim = 0))

				#reconstruction
				clean_features=two_dim_expand(clean_features,args.h_degree)
				reconLoss = MSELoss(aug_features, clean_features)

				#DNet
				real_features = aug_features[:real_images.shape[0]*(1+args.h_degree)]
				fake_features = aug_features[real_images.shape[0]*(1+args.h_degree):]
				real_discrimination = DNet(real_features,get_Lambda(index, start_step, total_step))
				real_session_labels=one_dim_expand(real_session_labels,args.h_degree)
				real_domain_loss = CELoss(real_discrimination, real_session_labels)
				total_DNet_output_session0+=(torch.argmax(real_discrimination,dim=1)==0).sum().item()
				total_DNet_output_session1+=(torch.argmax(real_discrimination,dim=1)==1).sum().item()

				##TripletLoss
				fake_session_labels=one_dim_expand(fake_session_labels,args.h_degree)
				total_features, total_labels = prepare_features_labels(real_features, fake_features, fake_session_labels)
				gen_asymmetricTripletLoss = TripletLoss(total_features, total_labels.squeeze())


				total_classification_loss += classification_loss.item()
				total_real_domain_loss += real_domain_loss.item()
				total_gen_asymmetricTripletLoss += gen_asymmetricTripletLoss.item()
				total_reconLoss+=reconLoss

				total_loss = classification_loss + 0.1 * real_domain_loss + 2 * gen_asymmetricTripletLoss + reconLoss
				optimizer_SGD.zero_grad()
				total_loss.backward()
				optimizer_SGD.step()
			scheduler_SGD.step()

			avg_classification_loss, avg_real_domain_loss, avg_gen_asymmetricTripletLoss, avg_aug_asymmetricTripletLoss, avg_reconLoss = total_classification_loss / train_data_len, total_real_domain_loss / train_data_len, total_gen_asymmetricTripletLoss / train_data_len, total_aug_asymmetricTripletLoss / train_data_len, total_reconLoss / train_data_len
			print('[Train]\navg_classification_loss: {:.5f} avg_real_domain_loss: {:.5f}\navg_gen_tripletLoss: {:.5f} avg_aug_tripletLoss: {:.5f} avg_reconLoss: {:.5f} session_0: {:.1f} session_1: {:.1f}'.format(avg_classification_loss, avg_real_domain_loss, avg_gen_asymmetricTripletLoss, avg_aug_asymmetricTripletLoss, avg_reconLoss,total_DNet_output_session0,total_DNet_output_session1))
			print()

			
			if epoch>25:
				with torch.no_grad():
					ENet.eval()
					#CNet.eval()
					DNet.eval()
					total_classification_loss = total_domain_loss = 0
					pred_list, label_list, folder_name_list = [], [], []
					total_DNet_output_session0=0.0
					total_DNet_output_session1=0.0
					for index, (images, labels, sessions, folder_name) in enumerate(tqdm(valid_data, ncols = 70, desc = 'Validate')):
						images, labels, sessions = images.to(device), labels.to(device), sessions.to(device)
						aug_predict, aug_features, clean_predict, clean_features= ENet(images, True)
						real_discrimination = DNet(clean_features, 1)

						total_DNet_output_session0+=(torch.argmax(real_discrimination,dim=1)==0).sum().item()
						total_DNet_output_session1+=(torch.argmax(real_discrimination,dim=1)==1).sum().item()

						total_classification_loss += CELoss(clean_predict, labels)
						total_domain_loss += CELoss(real_discrimination, sessions)

						pred_list.append(clean_predict[0].view(1,-1))
						label_list.append(labels[0].view(1,-1))
						folder_name_list.append(folder_name[0])

					avg_classification_loss = total_classification_loss / valid_data_len
					avg_domain_loss = total_domain_loss / valid_data_len
					print('[Valid]\navg_classification_loss: {:.5f} avg_domain_loss: {:.5f} session_0: {:.1f} session_1: {:.1f}'.format(avg_classification_loss, avg_domain_loss,total_DNet_output_session0,total_DNet_output_session1))
					print()

					acc=cal_accuracy(pred_list, label_list)
					AUC=cal_AUC(pred_list, label_list, folder_name_list, './valid_pred.csv', True)
					torch.save(ENet.state_dict(), join(save_path + 'E_SSDG_new_{}_{}_{}.pth'.format(acc,AUC,epoch)))
					torch.save(DNet.state_dict(), join(save_path + 'D_SSDG_new_{}_{}_{}.pth'.format(acc,AUC,epoch)))

def test(args, device):
	if args.data=='oulu':
		test_data_path=args.test_data_path
		batch_size=11
		csv_name='./test_pred_oulu.csv'
	elif args.data=='siw':
		test_data_path=args.test_data_path
		batch_size=10
		csv_name='./test_pred_siw.csv'
	test_dataloader = AllDataloader('test', test_data_path)
	test_data = DataLoader(test_dataloader, batch_size = batch_size, num_workers = 0, shuffle = False)
	print('loading model...')
	ENet = DG_model('resnet18',args)
	total_params = sum(p.numel() for p in ENet.parameters() if p.requires_grad)
	ENet.cuda().float()
	# print(ENet)
	# print("Total number of params = ", total_params)
	# print()

	# CNet = Classifier()
	# total_params = sum(p.numel() for p in CNet.parameters() if p.requires_grad)
	# CNet.cuda().float()
	# print(CNet)
	# print("Total number of params = ", total_params)
	# print()

	save_path = './models/'
	ENet.load_state_dict(torch.load(join(save_path, 'E_SSDG_new_100.0_1.0_19.pth')))
	# CNet.load_state_dict(torch.load(join(save_path, 'C_' + str(args.load) + '.pth')))

	with torch.no_grad():
		ENet.eval()
		# CNet.eval()
		pred_list, label_list, folder_name_list = [], [], []
		for index, (images, labels, sessions, folder_name) in enumerate(tqdm(test_data, ncols = 70)):
			images, labels = images.to(device), labels.to(device)
			aug_predict, aug_features, clean_predict, clean_features= ENet(images, True)

			pred_list.append(clean_predict[0].view(1,-1))
			label_list.append(labels[0].view(1,-1))
			folder_name_list.append(folder_name[0])

		cal_AUC(pred_list, label_list, folder_name_list, csv_name, False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)