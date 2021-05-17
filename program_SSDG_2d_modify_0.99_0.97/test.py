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
from DGFAS import *
from config import config

a=torch.tensor([1,2,3,4,5,6,7,8])
print(a.size())
a=a.unsqueeze(1)
print(a.size())
print(a)
a=a.expand(-1,2)
print(a)
a=a.reshape(-1)
print(a)
print(a.size())
