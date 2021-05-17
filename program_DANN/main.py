import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import SessionDataloader, AllDataloader
from torch.utils.data import DataLoader
from model import Extractor, Classifier, Discriminator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import sys, csv
import numpy as np
from sklearn.metrics import roc_auc_score

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 20)
	parser.add_argument('--lr_step', type = int, default = 10)
	parser.add_argument('--source', type = int)
	parser.add_argument('--target', type = int)
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	parser.add_argument('--test_data_path', type = str, default = '../oulu_npu_cropped/test')
	return parser.parse_args()

def cal_accuracy(pred_list, label_list, session):
	hit = 0
	for index in range(len(label_list)):
		hit += torch.sum(torch.argmax(pred_list[index]) == label_list[index]).item()
	total = len(label_list)
	print('s{}_Accuracy: {:.2f} %'.format(session, hit / total * 100))

def write_csv(folder_name_list, pred_prob_real, output_csv):
	folder_name_list = np.array(folder_name_list)
	df = pd.DataFrame([], columns = ['video_id','label'])
	df['video_id'] = folder_name_list
	df['label'] = pred_prob_real
	df = df.set_index('video_id')
	df.to_csv(output_csv)

def cal_AUC(pred_list, label_list, folder_name_list, session, output_csv, flag):
	pred_list = torch.cat(pred_list).detach().cpu()
	label_list = torch.cat(label_list).detach().cpu()

	label_list = np.array(label_list)
	label_list[label_list > 0] = -1
	label_list[label_list == 0] = 1
	label_list[label_list == -1] = 0

	pred_prob_real = nn.Softmax(dim = 1)(pred_list)[:, 0]
	pred_prob_real = np.array(pred_prob_real)
	write_csv(folder_name_list, pred_prob_real, output_csv)

	csv_reader = pd.read_csv(output_csv)['label']
	pred_prob_real = csv_reader.values.tolist()
	if flag == True:
		AUC_score = roc_auc_score(label_list, pred_prob_real)
		print('s{}_AUC: {:.3f}'.format(session, AUC_score))
		print()

def get_Lambda(index, start_step, total_step):
	p = float(index + start_step) / total_step
	lambda_term = 2. / (1. + np.exp(-10 * p)) - 1
	return lambda_term

def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_source_dataloader = SessionDataloader(args.mode, args.train_data_path, args.source)
		train_source_data = DataLoader(train_source_dataloader, batch_size = 4, num_workers = 3, shuffle = True, pin_memory = True)
		train_target_dataloader = SessionDataloader(args.mode, args.train_data_path, args.target)
		train_target_data = DataLoader(train_target_dataloader, batch_size = 4, num_workers = 3, shuffle = True, pin_memory = True)
		train_data_len = min(len(train_source_data), len(train_target_data))

		valid_source_dataloader = SessionDataloader('valid', args.valid_data_path, args.source)
		valid_source_data = DataLoader(valid_source_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
		valid_target_dataloader = SessionDataloader('valid', args.valid_data_path, args.target)
		valid_target_data = DataLoader(valid_target_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
		valid_data_len = min(len(valid_source_data), len(valid_target_data))

		print('loading model...')
		ENet = Extractor()
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

		optimizer_SGD = optim.SGD(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-4, weight_decay = 0.012, momentum = 0.9)
		scheduler_SGD = optim.lr_scheduler.StepLR(optimizer_SGD, step_size = train_data_len * args.lr_step, gamma = 0.1)

		# optimizer_Adam = optim.Adam(list(list(ENet.parameters()) + list(CNet.parameters()) + list(DNet.parameters())), lr = 1e-3, betas = (0.5, 0.9))		

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		best_loss = 100.0
		
		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}  (lr: {})'.format(epoch, scheduler_SGD.get_last_lr()[0]))
			# print('epoch: {}'.format(epoch))
			ENet.train()
			CNet.train()
			DNet.train()
			total_E_loss = total_C_loss = total_D_loss = 0
			start_step = epoch * train_data_len
			total_step = args.epochs * train_data_len

			for index, ((source_images, source_spoof_labels, source_folder_name), (target_images, target_spoof_labels, target_folder_name)) in enumerate(tqdm(zip(train_source_data, train_target_data), total = train_data_len, ncols = 70, desc = 'Training')):
				source_images, source_spoof_labels, target_images, target_spoof_labels = source_images.to(device), source_spoof_labels.to(device), target_images.to(device), target_spoof_labels.to(device)
				source_features = ENet(source_images)
				target_features = ENet(target_images)
				spoof_classification = CNet(source_features)
				classification_loss = CELoss(spoof_classification, source_spoof_labels)
				total_E_loss += classification_loss.item()
				total_C_loss += classification_loss.item()

				source_domain_labels = torch.zeros(source_spoof_labels.shape[0], dtype = torch.long).cuda()
				target_domain_labels = torch.ones(target_spoof_labels.shape[0], dtype = torch.long).cuda()

				source_discrimination = DNet(source_features, get_Lambda(index, start_step, total_step))
				target_discrimination = DNet(target_features, get_Lambda(index, start_step, total_step))
				source_domain_loss = CELoss(source_discrimination, source_domain_labels)
				target_domain_loss = CELoss(target_discrimination, target_domain_labels)
				total_E_loss += (source_domain_loss + target_domain_loss).item()
				total_D_loss += (source_domain_loss + target_domain_loss).item()
				total_loss = classification_loss + source_domain_loss + target_domain_loss
				optimizer_SGD.zero_grad()
				total_loss.backward()
				optimizer_SGD.step()
				scheduler_SGD.step()

			avg_E_loss, avg_C_loss, avg_D_loss = total_E_loss / train_data_len, total_C_loss / train_data_len, total_D_loss / train_data_len
			print('[Train] avg_E_loss: {:.5f} avg_C_loss: {:.5f} avg_D_loss: {:.5f}'.format(avg_E_loss, avg_C_loss, avg_D_loss))
			print()

			with torch.no_grad():
				ENet.eval()
				CNet.eval()
				total_source_loss = total_target_loss = 0
				source_pred_list, source_label_list, source_folder_name_list = [], [], []
				target_pred_list, target_label_list, target_folder_name_list = [], [], []
				for index, ((source_images, source_spoof_labels, source_folder_name), (target_images, target_spoof_labels, target_folder_name)) in enumerate(tqdm(zip(valid_source_data, valid_target_data), total = valid_data_len, ncols = 70, desc = 'Validate')):
					source_images, source_spoof_labels, target_images, target_spoof_labels = source_images.to(device), source_spoof_labels.to(device), target_images.to(device), target_spoof_labels.to(device)
					source_features = ENet(source_images)
					target_features = ENet(target_images)
					source_spoof_classification = CNet(source_features)
					target_spoof_classification = CNet(target_features)
					source_classification_loss = CELoss(source_spoof_classification, source_spoof_labels)
					target_classification_loss = CELoss(target_spoof_classification, target_spoof_labels)
					total_source_loss += source_classification_loss.item()
					total_target_loss += target_classification_loss.item()

					source_pred_list.append(source_spoof_classification)
					source_label_list.append(source_spoof_labels)
					source_folder_name_list.append(source_folder_name[0])

					target_pred_list.append(target_spoof_classification)
					target_label_list.append(target_spoof_labels)
					target_folder_name_list.append(target_folder_name[0])

				avg_source_loss = total_source_loss / valid_data_len
				avg_target_loss = total_target_loss / valid_data_len
				print('[Valid] avg_source_loss: {:.5f} avg_target_loss: {:.5f}'.format(avg_source_loss, avg_target_loss))
				print()

				cal_accuracy(source_pred_list, source_label_list, args.source)
				cal_AUC(source_pred_list, source_label_list, source_folder_name_list, args.source, './source_val_pred.csv', True)

				cal_accuracy(target_pred_list, target_label_list, args.target)
				cal_AUC(target_pred_list, target_label_list, target_folder_name_list, args.target, './target_val_pred.csv', True)

				torch.save(ENet.state_dict(), join(save_path + 'E_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
				torch.save(CNet.state_dict(), join(save_path + 'C_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
				torch.save(DNet.state_dict(), join(save_path + 'D_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

	elif args.mode == 'valid':
		valid_source_dataloader = SessionDataloader('valid', args.valid_data_path, args.source)
		valid_source_data = DataLoader(valid_source_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
		valid_target_dataloader = SessionDataloader('valid', args.valid_data_path, args.target)
		valid_target_data = DataLoader(valid_target_dataloader, batch_size = 1, num_workers = 3, shuffle = False, pin_memory = True)
		valid_data_len = min(len(valid_source_data), len(valid_target_data))

		print('loading model...')
		ENet = Extractor()
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
			total_source_loss = total_target_loss = 0
			source_pred_list, source_label_list, source_folder_name_list = [], [], []
			target_pred_list, target_label_list, target_folder_name_list = [], [], []
			for index, ((source_images, source_spoof_labels, source_folder_name), (target_images, target_spoof_labels, target_folder_name)) in enumerate(tqdm(zip(valid_source_data, valid_target_data), total = valid_data_len, ncols = 70, desc = 'Validation')):
				source_images, source_spoof_labels, target_images, target_spoof_labels = source_images.to(device), source_spoof_labels.to(device), target_images.to(device), target_spoof_labels.to(device)
				source_features = ENet(source_images)
				target_features = ENet(target_images)
				source_spoof_classification = CNet(source_features)
				target_spoof_classification = CNet(target_features)
				source_classification_loss = CELoss(source_spoof_classification, source_spoof_labels)
				target_classification_loss = CELoss(target_spoof_classification, target_spoof_labels)
				total_source_loss += source_classification_loss.item()
				total_target_loss += target_classification_loss.item()

				source_pred_list.append(source_spoof_classification)
				source_label_list.append(source_spoof_labels)
				source_folder_name_list.append(source_folder_name[0])

				target_pred_list.append(target_spoof_classification)
				target_label_list.append(target_spoof_labels)
				target_folder_name_list.append(target_folder_name[0])

			avg_source_loss = total_source_loss / valid_data_len
			avg_target_loss = total_target_loss / valid_data_len
			print('[Valid] avg_source_loss: {:.5f} avg_target_loss: {:.5f}'.format(avg_source_loss, avg_target_loss))
			print()

			cal_accuracy(source_pred_list, source_label_list, args.source)
			cal_AUC(source_pred_list, source_label_list, source_folder_name_list, args.source, './source_val_pred.csv', True)

			cal_accuracy(target_pred_list, target_label_list, args.target)
			cal_AUC(target_pred_list, target_label_list, target_folder_name_list, args.target, './target_val_pred.csv', True)

def test(args, device):
	test_dataloader = AllDataloader('test', args.test_data_path)
	test_data = DataLoader(test_dataloader, batch_size = 1, num_workers = 0, shuffle = False, pin_memory = True)
	print('loading model...')
	ENet = Extractor()
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
		for index, (images, labels, folder_name) in enumerate(tqdm(test_data, ncols = 70)):
			batch_images, batch_labels = images.to(device), labels.to(device)
			features = ENet(batch_images)
			predict = CNet(features)
			pred_list.append(predict)
			label_list.append(batch_labels)
			folder_name_list.append(folder_name[0])
		cal_AUC(pred_list, label_list, folder_name_list, 3, './test_pred.csv', False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)
