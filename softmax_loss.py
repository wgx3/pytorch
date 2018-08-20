#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable

logsm = nn.LogSoftmax()
loss = nn.NLLLoss()

input = Variable(torch.randn(3,5),requires_grad=True)
logsm_out = logsm(input)

target = Variable(torch.LongTensor([1,0,4]))
l = loss((logsm_out), target)
l.backward()

print(input.size(), target.size(), l.size())