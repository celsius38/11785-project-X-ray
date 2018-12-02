import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence,pad_sequence,pack_padded_sequence,pack_sequence
import sys

from preprocess import INT_TO_CHAR, CHAR_TO_INT
from model_config import *


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5

data = np.load('data/data.npy')
findings = np.load('data/findings.npy')
indications = np.load('data/indications.npy')
impressions = np.load('data/impressions.npy')

idx = np.arange(len(data))
np.random.shuffle(idx)
train_idx, dev_idx, test_idx = idx[:6000],idx[6000:6000+735],idx[6000+735:] # 8 : 1 : 1

# use findings 
train_x = data[train_idx]
train_y = findings[train_idx]

dev_x = data[dev_idx]
dev_y = findings[dev_idx]

test_x = data[test_idx]
test_y = findings[test_idx]

train_set = CustomDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, collate_fn=collate_lines)

cnn = ResNet() # image 
lstm = XrayNet(args["vocab_size"], args["lstm_hidden_size"]) # report
optimizer = torch.optim.Adam([{'params':cnn.parameters()}, {'params':lstm.parameters()}],lr = 1e-4)
criterion = nn.CrossEntropyLoss().to(DEVICE)



for epoch in range(NUM_EPOCHS):
    loss, cnn, lstm = train(train_loader, cnn, lstm, optimizer, criterion, DEVICE)
    print('='*100)
    print(epoch," training loss:", loss)
    torch.save(cnn.state_dict(),'./cnn_'+str(epoch) + '.pt')
    torch.save(lstm.state_dict(),'./lstm_'+str(epoch) + '.pt')
    
    if epoch%1 == 0: 
        distance = validation(dev_loader, cnn, lstm, DEVICE)
        print(epoch," validation distance:", distance)
 




