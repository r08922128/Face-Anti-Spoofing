import os
from os import listdir
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import random
import torchvision.utils as utils 

size = 512

class RealFakeDataloader(Dataset):
    def __init__(self, mode, data_path, realFake):
        super(RealFakeDataloader, self).__init__()
        self.mode = mode
        self.folder_name = os.listdir(data_path)
        # realFake = {0, 1} real, fake
        self.realFake = realFake
        if realFake == 0:
            self.realFake_folder_name = [folder_name for folder_name in self.folder_name if int(folder_name[-1]) - 1 == realFake]
        else:
            self.realFake_folder_name = [folder_name for folder_name in self.folder_name if int(folder_name[-1]) != realFake]

        self.realFake_folder_path = [join(data_path, folder_name) for folder_name in self.realFake_folder_name]
        self.transform = transforms.Compose([
                            transforms.Lambda(self.openImage),
                            transforms.Resize((size, size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])
        self.all_slices_realFake_path = []
        # self.all_slices_realFake_name = []
        self.sessions = []
        self.folder_names = []
        for folder_name in self.realFake_folder_name:
            path = join(data_path, folder_name)
            for slices in os.listdir(path):
                self.all_slices_realFake_path.append(join(path, slices))
                self.folder_names.append(folder_name)
                # self.all_slices_realFake_name.append(slices)
                self.sessions.append(int(folder_name[folder_name.index('_') + 1]) - 1)

        # print(self.sessions)
        
    def __len__(self):
        return len(self.all_slices_realFake_path)
    
    def openImage(self, x):
        return Image.open(x)

    def randomFlip(self, x):
        flipper = transforms.RandomHorizontalFlip(p = 0.0)
        return flipper(x)

    def __getitem__(self, index):
        slices = self.transform(self.all_slices_realFake_path[index])
        session = self.sessions[index]
        folder_name = self.folder_names[index]
        if self.realFake == 0:
            label = 0
        elif 2 <= int(folder_name[-1]) <= 3:
            label = 1
        else:
            label = 2 
        return slices, label, session, folder_name


class SessionDataloader(Dataset):
    def __init__(self, mode, data_path, session):
        super(SessionDataloader, self).__init__()
        self.mode = mode
        self.folder_name = os.listdir(data_path)
        self.session_folder_name = [folder_name for folder_name in self.folder_name if int(folder_name[folder_name.index('_') + 1]) == session]
        self.session_folder_path = [join(data_path, folder_name) for folder_name in self.session_folder_name]
        self.transform = transforms.Compose([
                            transforms.Lambda(self.openImage),
                            transforms.Resize((size, size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])

    def __len__(self):
        return len(self.session_folder_name)
    
    def openImage(self, x):
        return Image.open(x)

    def randomFlip(self, x):
        flipper = transforms.RandomHorizontalFlip(p = 0.0)
        return flipper(x)

    def __getitem__(self, index):
        folder_name = self.session_folder_name[index]
        slices_name = sorted(os.listdir(self.session_folder_path[index]))
        slices = torch.cat([self.transform(join(self.session_folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0).permute(1, 0, 2, 3)
        access = 0
        if self.mode != 'test':
            slices = self.randomFlip(slices)
            access = 0 if int(folder_name[-1]) == 1 else 1
        return slices, access, folder_name



class AllDataloader(Dataset):
    def __init__(self, mode, data_path):
        super(AllDataloader, self).__init__()
        self.mode = mode
        self.data_path = data_path
        self.folder_name = os.listdir(data_path)
        self.folder_path = [join(data_path, folder_name) for folder_name in self.folder_name]
        self.transform = transforms.Compose([
                        transforms.Lambda(self.openImage),
                        transforms.Resize((size, size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        self.slice_path = []
        self.folder_labels = []
        self.access = []
        self.sessions = []

        for folder_name in self.folder_name:
            path = join(data_path, folder_name)
            for slic in os.listdir(path):
                self.slice_path.append(join(path,slic))
                self.folder_labels.append(folder_name)

                if int(folder_name[-1]) == 1:
                    self.access.append(0)
                elif 2 <= int(folder_name[-1]) <= 3:
                    self.access.append(1)
                else:
                    self.access.append(2)
                if self.mode != 'test':
                    self.sessions.append(int(folder_name[folder_name.index('_') + 1]) - 1)

    def __len__(self):
        return len(self.slice_path)
    
    def openImage(self, x):
        return Image.open(x)

    def randomFlip(self, x):
        flipper = transforms.RandomHorizontalFlip(p = 0.0)
        return flipper(x)

    def __getitem__(self, index):
        slice_ = self.slice_path[index]
        slice_img = self.transform(slice_)
        access = self.access[index]
        session = self.sessions[index] if self.mode != 'test' else 0
        folder_name = self.folder_labels[index]
        return slice_img, access, session, folder_name

if __name__ == '__main__':
    test = RealFakeDataloader('train', '../oulu_npu_cropped/val', 1)
    test_data = DataLoader(test, batch_size = 4, shuffle = True)
    for index, (image, access, session, folder_name) in enumerate(test_data):
        print(index, image.shape, access, session, folder_name)
        break