import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def cal_accuracy(pred_list, label_list):
	hit = 0
	real_pred = print_pred = replay_pred = 0
	for index in range(len(label_list)):
		hit += torch.sum(torch.argmax(pred_list[index]) == label_list[index]).item()
		real_pred += torch.sum(torch.argmax(pred_list[index]) == 0).item()
		print_pred += torch.sum(torch.argmax(pred_list[index]) == 1).item()
		replay_pred += torch.sum(torch.argmax(pred_list[index]) == 2).item()
	total = len(label_list)
	acc = hit / total * 100
	print('Accuracy: {:.2f} %'.format(acc))
	print('real: ', real_pred)
	print('print: ', print_pred)
	print('replay: ', replay_pred)
	return acc

def write_csv(folder_name_list, pred_prob_real, output_csv):
	folder_name_list = np.array(folder_name_list)
	df = pd.DataFrame([], columns = ['video_id','label'])
	df['video_id'] = folder_name_list
	df['label'] = pred_prob_real
	df = df.set_index('video_id')
	df.to_csv(output_csv)

def cal_AUC(pred_list, label_list, folder_name_list, output_csv, flag):
	pred_list = torch.cat(pred_list, dim=0).detach().cpu()
	label_list = torch.cat(label_list, dim=0).detach().cpu()

	label_list = np.array(label_list)
	label_list[label_list > 0] = -1
	label_list[label_list == 0] = 1
	label_list[label_list == -1] = 0

	pred_prob_real = torch.argmax(nn.Softmax(dim = 1)(pred_list), dim = 1)
	pred_prob_real = np.array(pred_prob_real)
	write_csv(folder_name_list, pred_prob_real, output_csv)
	print()
