#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable

logsm = nn.LogSoftmax()
