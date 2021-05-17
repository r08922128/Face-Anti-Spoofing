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
	parser.add_argument('--load', type = int, default = 1)
	parser.add_argument('--epochs', type = int, default = 30)
	parser.add_argument('--lr_step', type = int, default = 10)
	parser.add_argument('--h_degree', type = int, default = 1)
	parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
	parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
	#parser.add_argument('--test_data_path', type = str, default = '../siw_test')
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

    train_real_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 0)
    train_real_data = DataLoader(train_real_dataloader, batch_size = 16, num_workers = 4, shuffle = True)
    train_fake_dataloader = RealFakeDataloader(args.mode, args.train_data_path, 1)
    train_fake_data = DataLoader(train_fake_dataloader, batch_size = 16, num_workers = 4, shuffle = True)
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

    ENet.load_state_dict(torch.load(join(save_path, 'E_SSDG_new_99.8888888888889_0.9999845679012346_28.pth')))
    #DNet.load_state_dict(torch.load(join(save_path, 'D_SSDG_new_99.33333333333333_0.9997067901234569_29.pth')))

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

    with torch.no_grad():
        epoch=1
        ENet.eval()
        # CNet.train()
        DNet.eval()
        total_classification_loss = total_real_domain_loss = total_gen_asymmetricTripletLoss = total_aug_asymmetricTripletLoss = total_reconLoss = 0
        start_step = epoch * train_data_len
        total_step = args.epochs * train_data_len
        total_DNet_output_session0=0.0
        total_DNet_output_session1=0.0
        all_features=[]
        all_realfake_label=[]
        all_session_label=[]
        for index, ((real_images, real_spoof_labels, real_session_labels, real_folder_name), (fake_images, fake_spoof_labels, fake_session_labels, fake_folder_name)) in enumerate(tqdm(zip(train_real_data, train_fake_data), total = train_data_len, ncols = 70, desc = 'Training')):
            if index>30:
                break
            real_images, real_spoof_labels, real_session_labels = real_images.to(device), real_spoof_labels.to(device), real_session_labels.to(device)
            fake_images, fake_spoof_labels, fake_session_labels = fake_images.to(device), fake_spoof_labels.to(device), fake_session_labels.to(device)
            images = torch.cat((real_images, fake_images), dim = 0)
            aug_predict, aug_features, clean_predict, clean_features= ENet(images, True)

            real_spoof_labels=one_dim_expand(real_spoof_labels,args.h_degree)
            fake_spoof_labels=one_dim_expand(fake_spoof_labels,args.h_degree)
            real_session_labels=one_dim_expand(real_session_labels,args.h_degree)
            fake_session_labels=one_dim_expand(fake_session_labels,args.h_degree)
            all_features.append(aug_features)
            all_realfake_label.append(real_spoof_labels)
            all_realfake_label.append(fake_spoof_labels)
            all_session_label.append(real_session_labels)
            all_session_label.append(fake_session_labels)

        ##tsne
        all_features=torch.cat(all_features)
        all_realfake_label=torch.cat(all_realfake_label)
        all_session_label=torch.cat(all_session_label)

    #plot 2d
    realfake_session_label=[]
    for i in range(all_realfake_label.size(0)):
        if all_realfake_label[i].item()==0 and all_session_label[i].item()==0:
            realfake_session_label.append(0)
        elif all_realfake_label[i].item()==0 and all_session_label[i].item()==1:
            realfake_session_label.append(1)
        elif all_realfake_label[i].item()==1 and all_session_label[i].item()==0:
            realfake_session_label.append(2)
        else:
            realfake_session_label.append(3)

    label=realfake_session_label
    all_features=all_features.detach().cpu()
    tsne = TSNE(n_components=2, init='random', random_state=5, verbose=1) 
    X_tsne = tsne.fit_transform(all_features)
    print(X_tsne.shape)

    data=X_tsne
    # x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    # data = (data- x_min) / (x_max - x_min)
    x=data[:,0]
    y=data[:,1]


    ax = plt.figure().add_subplot(111) 
    #nsamples=50
    #colors={0:'b',1:'r',2:'r',3:'c',4:'m',5:'y',6:'k'}
    #c=[colors[i] for i in np.round(np.random.uniform(0,6,nsamples),0)]
    c=['#52C0D4','#F6C75C','#1D99A6','#DDA230']
    for i in range(data.shape[0]):
        if i%(args.h_degree+1)==0: 
            temp=ax.scatter(x[i],y[i],color=c[int(label[i])], marker='.')
            if int(label[i])==0:
                p0=temp
            elif int(label[i])==1:
                p1=temp
            elif int(label[i])==2:
                p2=temp
            else:
                p3=temp
        else:
            temp=ax.scatter(x[i],y[i],color=c[int(label[i])], marker='x')
            if int(label[i])==0:
                p4=temp
            elif int(label[i])==1:
                p5=temp
            elif int(label[i])==2:
                p6=temp
            else:
                p7=temp
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
    ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.legend([p0,p1,p2,p3,p4,p5,p6,p7],['real session1','real session2','fake session1','fake session2','hallu real s1','hallu real s2','hallu fake s1','hallu fake s2'],scatterpoints=1)

    plt.show()


if __name__ == '__main__':
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args, device)