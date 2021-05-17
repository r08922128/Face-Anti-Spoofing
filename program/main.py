import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Classifier
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
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	parser.add_argument('--test_data_path', type = str, default = '../oulu_npu_cropped/test')
	return parser.parse_args()

def cal_accuracy(pred_list, label_list):
	hit = 0
	for index in range(len(label_list)):
		hit += torch.sum(torch.argmax(pred_list[index]) == label_list[index]).item()
	total = len(label_list)
	print('2_class_Accuracy: {:.2f} %'.format(hit / total * 100))
	print()

def write_csv(folder_name_list, pred_prob_real, output_csv):
	folder_name_list = np.array(folder_name_list)
	df = pd.DataFrame([], columns = ['video_id','label'])
	df['video_id'] = folder_name_list
	df['label'] = pred_prob_real
	df = df.set_index('video_id')
	df.to_csv(output_csv)

def cal_AUC(pred_list, label_list, folder_name_list, output_csv, flag):
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
		print('AUC: {:.3f}'.format(AUC_score))
		print()

def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_dataloader = dataloader(args.mode, args.train_data_path)
		train_data = DataLoader(train_dataloader, batch_size = 6, num_workers = 5, shuffle = True, pin_memory = True)
		valid_dataloader = dataloader('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
		print('loading model...')
		Net = Classifier()
		print(Net)
		total_params = sum(p.numel() for p in Net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		Net.cuda().float()

		save_path = './models/'
		os.makedirs(save_path, exist_ok = True)

		if args.load != -1:
			Net.load_state_dict(torch.load(join(save_path, str(args.load) + '.ckpt')))

		optimizer = optim.SGD(filter(lambda param : param.requires_grad, list(Net.parameters())), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = len(train_data) * args.lr_step, gamma = 0.1)
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()
		best_loss = 100.0
		
		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}  (lr: {})'.format(epoch, scheduler.get_last_lr()[0]))
			total_loss = 0
			for index, (images, labels, folder_name) in enumerate(tqdm(train_data, ncols = 70)):
				batch_images, batch_labels = images.to(device), labels.to(device)
				predict = Net(batch_images)
				loss = loss_func(predict, batch_labels)
				total_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()

			avg_loss = total_loss / len(train_data)
			print('train_avg_loss:', avg_loss)

			total_loss = 0
			with torch.no_grad():
				Net.eval()
				pred_list, label_list, folder_name_list = [], [], []
				for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_images, batch_labels = images.to(device), labels.to(device)
					predict = Net(batch_images)
					pred_list.append(predict)
					label_list.append(batch_labels)
					folder_name_list.append(folder_name[0])
					loss = loss_func(predict, batch_labels)
					total_loss += loss.item()

				avg_loss = total_loss / len(valid_data)
				print('valid_avg_loss:', avg_loss)
				print()
				cal_accuracy(pred_list, label_list)
				cal_AUC(pred_list, label_list, folder_name_list, './val_pred.csv', True)
				if avg_loss < best_loss:
					best_loss = avg_loss
					torch.save(Net.state_dict(), join(save_path, 'best.ckpt'), _use_new_zipfile_serialization = False)
				torch.save(Net.state_dict(), join(save_path + '{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

	elif args.mode == 'valid':
		valid_dataloader = dataloader('valid', args.valid_data_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
		print('loading model...')
		Net = Classifier()
		print(Net)
		total_params = sum(p.numel() for p in Net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		Net.cuda().float()

		save_path = './models/'
		Net.load_state_dict(torch.load(join(save_path, str(args.load) + '.ckpt')))

		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()
		total_loss = 0
		with torch.no_grad():
			Net.eval()
			pred_list, label_list, folder_name_list = [], [], []
			for index, (images, labels, folder_name) in enumerate(tqdm(valid_data, ncols = 70)):
				batch_images, batch_labels = images.to(device), labels.to(device)
				predict = Net(batch_images)
				pred_list.append(predict)
				label_list.append(batch_labels)
				folder_name_list.append(folder_name[0])
				loss = loss_func(predict, batch_labels)
				total_loss += loss.item()
			avg_loss = total_loss / len(valid_data)
			print('valid_avg_loss:', avg_loss)
			print()
			cal_accuracy(pred_list, label_list)
			cal_AUC(pred_list, label_list, folder_name_list, './val_pred.csv', True)

def test(args, device):
	test_dataloader = dataloader('test', args.test_data_path)
	test_data = DataLoader(test_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
	print('loading model...')
	Net = Classifier()
	print(Net)
	total_params = sum(p.numel() for p in Net.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	Net.cuda().float()

	save_path = './models/'
	Net.load_state_dict(torch.load(join(save_path, str(args.load) + '.ckpt')))

	with torch.no_grad():
		Net.eval()
		pred_list, label_list, folder_name_list = [], [], []
		for index, (images, labels, folder_name) in enumerate(tqdm(test_data, ncols = 70)):
			batch_images, batch_labels = images.to(device), labels.to(device)
			predict = Net(batch_images)
			pred_list.append(predict)
			label_list.append(batch_labels)
			folder_name_list.append(folder_name[0])
		cal_AUC(pred_list, label_list, folder_name_list, './test_pred.csv', False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)
