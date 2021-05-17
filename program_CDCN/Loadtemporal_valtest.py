from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
from os.path import join

#frames_total = 11    # each video 8 uniform samples


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, string_name, folder_name = sample['image_x'],sample['binary_mask'],sample['string_name'],sample['folder_name']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'string_name': string_name, 'folder_name':folder_name}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, string_name, folder_name = sample['image_x'],sample['binary_mask'],sample['string_name'],sample['folder_name']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
                        
        binary_mask = np.array(binary_mask)
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': string_name, 'folder_name':folder_name} 



class Spoofing_valtest(Dataset):

    def __init__(self, mode, data_path,  transform=None,frames_total = 11):
        self.mode = mode
        self.data_path = data_path
        self.folder_name = os.listdir(data_path)
        #self.folder_path = [join(data_path, folder_name) for folder_name in self.folder_name]
        # self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        # self.root_dir = root_dir
        self.transform = transform
        self.frames_total=frames_total
    def __len__(self):
        return len(self.folder_name)

    
    def __getitem__(self, index):
        
        # videoname = str(self.landmarks_frame.iloc[idx, 0])
        # image_path = os.path.join(self.root_dir, videoname)
        # image_path = os.path.join(image_path, 'profile')
        folder_name = self.folder_name[index]
        if self.mode != 'test':
            spoofing_label = 1 if int(folder_name[-1]) == 1 else 0
        else:
            spoofing_label = 0

        # slices_name = sorted(os.listdir(self.folder_path[index]))
        # slices = torch.cat([self.transform(join(self.folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0).permute(1, 0, 2, 3)
        folder_path=join(self.data_path,folder_name)
        image_x, binary_mask = self.get_single_image_x(folder_path)
		    
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'string_name': spoofing_label, 'folder_name': folder_name}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, folder_path):

        # files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        # interval = files_total//frames_total
        image_id_total=os.listdir(folder_path)

        image_x = np.zeros((self.frames_total, 256, 256, 3))
        
        binary_mask = np.zeros((self.frames_total, 32, 32))
        
        
        
        # random choose 1 frame
        for ii in range(len(image_id_total)):
            # image_id = ii*interval + 1 
            
            # s = "%04d.jpg" % image_id            
            
            # RGB
            image_path2 = os.path.join(folder_path, image_id_total[ii])
            image_x_temp = cv2.imread(image_path2)
            
            image_x_temp_gray = cv2.imread(image_path2, 0)
            image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))

            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            
            #print(image_path2)
            
            for i in range(32):
                for j in range(32):
                    if image_x_temp_gray[i,j]>0:
                        binary_mask[ii, i, j]=1.0
                    else:
                        binary_mask[ii, i, j]=0.0
            

        
        return image_x, binary_mask







if __name__ == '__main__':
    # usage
    # MAHNOB
    root_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/cropped_frm/'
    trainval_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/ImageSet_5fold/trainval_zitong_fold1.txt'
    

    BioVid_train = BioVid(trainval_list, root_list, transform=transforms.Compose([Normaliztion(), Rescale((133,108)),RandomCrop((125,100)),RandomHorizontalFlip(),  ToTensor()]))
    
    dataloader = DataLoader(BioVid_train, batch_size=1, shuffle=True, num_workers=8)
    
    # print first batch for evaluation
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_x'].size(), sample_batched['video_label'].size())
        print(i_batch, sample_batched['image_x'], sample_batched['pain_label'], sample_batched['ecg'])
        pdb.set_trace()
        break

            
 


