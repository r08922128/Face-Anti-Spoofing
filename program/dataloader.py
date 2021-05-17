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

class dataloader(Dataset):
	def __init__(self, mode, data_path):
		super(dataloader, self).__init__()
		self.mode = mode
		self.data_path = data_path
		self.folder_name = os.listdir(data_path)
		self.folder_path = [join(data_path, folder_name) for folder_name in self.folder_name]
		self.transform = transforms.Compose([
							transforms.Lambda(self.openImage),
							transforms.Resize((512, 512)),
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
						 ])

	def __len__(self):
		return len(self.folder_path)
	
	def openImage(self, x):
		return Image.open(x)

	def randomFlip(self, x):
		flipper = transforms.RandomHorizontalFlip(p = 0.3)
		return flipper(x)

	def __getitem__(self, index):
		folder_name = self.folder_name[index]
		slices_name = sorted(os.listdir(self.folder_path[index]))
		slices = torch.cat([self.transform(join(self.folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0).permute(1, 0, 2, 3)
		access = 0
		if self.mode != 'test':
			slices = self.randomFlip(slices)
			access = 0 if int(folder_name[-1]) == 1 else 1
		return slices, access, folder_name

if __name__ == '__main__':
	test = dataloader('train', '../oulu_npu_cropped/train')
	test_data = DataLoader(test, batch_size = 2, shuffle = False)
	for index, (image, access, folder_name) in enumerate(test_data):
		print(index, image.shape, access, folder_name)
		break