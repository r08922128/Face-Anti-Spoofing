from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os.path import join
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from models.CDCNs import Conv2d_cd, CDCNpp


from Loadtemporal_BinaryMask_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root
# image_dir = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/'          


# train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_train.txt'
# val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_dev_res.txt'

#train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_train.txt'
#val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_dev_res.txt'

#train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_train.txt'
#val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_dev_res.txt'


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss

def cal_accuracy(pred_list, label_list):
	hit = 0
	for index in range(len(label_list)):
		hit += torch.sum(torch.argmax(pred_list[index]) == label_list[index]).item()
	total = len(label_list)
	print('2_class_Accuracy: {:.2f} %'.format(hit / total * 100))
	print()
	return hit / total * 100

def cal_AUC(pred_list,label_list,flag):

	pred_list=torch.cat(pred_list).detach().cpu()
	label_list=torch.cat(label_list).detach().cpu()

	label_list=np.array(label_list)
	# label_list[label_list>0]=-1
	# label_list[label_list==0]=1
	# label_list[label_list==-1]=0

	pred_prob_real=[]
	softmax=nn.Softmax(dim=1)
	pred_prob_real=softmax(pred_list)[:,1]
	pred_prob_real=np.array(pred_prob_real)

	AUC_score=0
	if flag==True:
		AUC_score=roc_auc_score(label_list,pred_prob_real)
	print('AUC: {:.3f}'.format(AUC_score))
	print()
	return AUC_score, pred_prob_real


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')


    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         

        #model = CDCNpp( basic_conv=Conv2d_cd, theta=0.7)
        model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
        model.load_state_dict(torch.load('./CDCNpp_BinaryMask/CDCNpp_BinaryMask_1.pkl'))

        model = model.cuda()

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    


    ACER_save = 1.0
    
    ## for train dataloader
    data_path=args.train_data_path
    image_path=[]
    image_label=[]
    folder_name=os.listdir(data_path)
    folder_path = [join(data_path, folder_name_) for folder_name_ in folder_name]
    for i in range(len(folder_path)):
        slices_name = sorted(os.listdir(folder_path[i]))
        for j in range(len(slices_name)):
            image_path.append(join(folder_path[i],slices_name[j]))
            image_label.append(1 if int(folder_name[i][-1]) == 1 else 0)
    image_path=np.array(image_path)
    image_label=np.array(image_label)


    best_val_acc=0.0
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(image_path, image_label, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(tqdm(dataloader_train)):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 

            optimizer.zero_grad()

            #print(inputs.size())#orch.Size([9, 3, 256, 256])
            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
            #print(map_x.size())#torch.Size([9, 32, 32])

            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            
            loss =  absolute_loss + contrastive_loss
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            
            
        

            # if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
            #     # visualization
            #     #FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

            #     # log written
            #     print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
        
            #break            
            
        # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
            
        epoch_test = 1            
        with torch.no_grad():
            model.eval()
            ###########################################
            '''                val             '''
            ###########################################
            # val for threshold
            # valid_dataloader = dataloader_3d('valid', args.valid_data_path)
            # valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
            val_data = Spoofing_valtest('valid', args.valid_data_path, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
            
            map_score_list = []
            pred_list, label_list = [], []
            for i, sample_batched in enumerate(tqdm(dataloader_val)):
                # get the inputs
                inputs = sample_batched['image_x'].cuda()
                string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
    
                optimizer.zero_grad()
                
                
                map_score = 0.0
                for frame_t in range(inputs.shape[1]):
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                    #print(map_x.size())#torch.Size([1, 32, 32])
                    # print(torch.sum(map_x),'  ',torch.sum(binary_mask[:,frame_t,:,:]))
                    score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                    map_score += score_norm
                map_score = map_score/inputs.shape[1]
                #print(map_score)
                if map_score>1:
                    map_score = 1.0

                map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
                pred_list.append(torch.tensor([1-map_score,map_score]).unsqueeze(0))
                label_list.append(string_name.unsqueeze(0))
            map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_%d.txt'% (epoch + 1)
            with open(map_score_val_filename, 'w') as file:
                file.writelines(map_score_list)                
            acc=cal_accuracy(pred_list, label_list)
            AUC_scores,pred_prob_real=cal_AUC(pred_list,label_list,True)
        
        if acc > best_val_acc:
            best_val_acc = acc
            print('saving best model')
            print('best_acc: ',acc)
            print('best_AUC: ',AUC_scores)
            torch.save(model.state_dict(),'best_CDCN_'+str(round(acc,1))+'_'+str(round(AUC_scores*100,1))+'.pth', _use_new_zipfile_serialization = False)
        # save the model until the next improvement     
        #torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=15, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--train_data_path', type = str, default = '../oulu_npu_cropped/train')
    parser.add_argument('--valid_data_path', type = str, default = '../oulu_npu_cropped/val')
    args = parser.parse_args()
    train_test()
