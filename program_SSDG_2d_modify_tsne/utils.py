import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def cal_accuracy(pred_list, label_list):
	hit = 0
	for index in range(len(label_list)):
		hit += torch.sum(torch.argmax(pred_list[index]) == label_list[index]).item()
	total = len(label_list)
	print('Accuracy: {:.2f} %'.format(hit / total * 100))
	return hit / total * 100
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

	pred_prob_real = nn.Softmax(dim = 1)(pred_list)[:, 0]
	pred_prob_real = np.array(pred_prob_real)
	write_csv(folder_name_list, pred_prob_real, output_csv)

	csv_reader = pd.read_csv(output_csv)['label']
	pred_prob_real = csv_reader.values
	for i in range(11):
		print("Threshold: ", i / 10, "\tPositive number: ", (pred_prob_real > i / 10).sum().item())
	if flag == True:
		AUC_score = roc_auc_score(label_list, pred_prob_real.tolist())
		print('AUC: {:.3f}'.format(AUC_score))
		return AUC_score
	else:
		return 0
