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

size = 128

class AllDataloader_2D(Dataset):
	def __init__(self, mode, data_path):
		super(AllDataloader_2D, self).__init__()
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

	def __len__(self):
		return len(self.folder_path)
	
	def openImage(self, x):
		return Image.open(x)

	def randomFlip(self, x):
		flipper = transforms.RandomHorizontalFlip(p = 0.0)
		return flipper(x)

	def __getitem__(self, index):
		folder_name = self.folder_name[index]
		slices_name = sorted(os.listdir(self.folder_path[index]))
		slices = torch.cat([self.transform(join(self.folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0)
		access = 0
		if self.mode != 'test':
			slices = self.randomFlip(slices)
			access = 0 if int(folder_name[-1]) == 1 else 1
		return slices, access, folder_name

class RealFakeDataloader_2D(Dataset):
	def __init__(self, mode, data_path, realFake):
		super(RealFakeDataloader_2D, self).__init__()
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
	def __len__(self):
		return len(self.realFake_folder_name)
	
	def openImage(self, x):
		return Image.open(x)

	def randomFlip(self, x):
		flipper = transforms.RandomHorizontalFlip(p = 0.0)
		return flipper(x)

	def __getitem__(self, index):
		folder_name = self.realFake_folder_name[index]
		slices_name = sorted(os.listdir(self.realFake_folder_path[index]))
		slices = torch.cat([self.transform(join(self.realFake_folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0)
		if self.mode != 'test':
			slices = self.randomFlip(slices)
			session = int(folder_name[folder_name.index('_') + 1]) - 1
		return slices, self.realFake, session, folder_name


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
	def __len__(self):
		return len(self.realFake_folder_name)
	
	def openImage(self, x):
		return Image.open(x)

	def randomFlip(self, x):
		flipper = transforms.RandomHorizontalFlip(p = 0.0)
		return flipper(x)

	def __getitem__(self, index):
		folder_name = self.realFake_folder_name[index]
		slices_name = sorted(os.listdir(self.realFake_folder_path[index]))
		slices = torch.cat([self.transform(join(self.realFake_folder_path[index], slice_name)).unsqueeze(0) for slice_name in slices_name], dim = 0).permute(1, 0, 2, 3)
		if self.mode != 'test':
			slices = self.randomFlip(slices)
			session = int(folder_name[folder_name.index('_') + 1]) - 1
		return slices, self.realFake, session, folder_name


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

	def __len__(self):
		return len(self.folder_path)
	
	def openImage(self, x):
		return Image.open(x)

	def randomFlip(self, x):
		flipper = transforms.RandomHorizontalFlip(p = 0.0)
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
	test = RealFakeDataloader('train', '../oulu_npu_cropped/train', 1)
	test_data = DataLoader(test, batch_size = 4, shuffle = True)
	for index, (image, access, session, folder_name) in enumerate(test_data):
		print(index, image.shape, access, session, folder_name)
		break