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
from preprocess import INT_TO_CHAR, CHAR_TO_INT
import sys
from Levenshtein import distance

args = {}
args["vocab_size"] = 58
args["lstm_hidden_size"] = 512
args["batch_size"] = 32
args["epochs"] = 15
args["num_workers"] = 4
args["image_embed_size"] = 2048
args["gpu"] = True
if (not torch.cuda.is_available()): args["gpu"] = False
args["label_cutoff"] = 0.2 # minimum probability of a softmax output for a valid label
args["k"] = 4 # select top k softmax outputs as labels
args["cnn_output_size"] = 512
args["char_embed_size"] = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, data, label = None):
        self._data = data
        label = [np.append(y,0) for y in label]
        self._label = label

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        d =  self._data[index]
        d = torch.from_numpy(d).float()
        l = torch.tensor([0])
        if self._label is not None:
            l = torch.from_numpy(self._label[index]).long()
        d, l = d.to(DEVICE), l.to(DEVICE)
        return (d, l)
    
def collate_lines(batch):
    batch = sorted(batch, key = lambda x: len(x[0]), reverse = True)
    data = [b[0].unsqueeze(0) for b in batch] # B of (1, W, H)
    target = [b[1] for b in batch] # B of (L, )
    return torch.cat(data, dim = 0).unsqueeze(1), target #(B, 1, W, H), (B, )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.elu(out)

        return out

class ResNet(nn.Module):
    """
    tunable hyper parameters: embeddings
    """
    def __init__(self):
        global args 
        super(ResNet, self).__init__()
        self.network = nn.Sequential(
                nn.Conv2d(1,32,kernel_size = 5,padding = 0,stride = 2,bias = False),
                nn.ELU(inplace=True),
                BasicBlock(32,32), 
                nn.Conv2d(32,64,kernel_size = 5,padding = 0,stride = 2,bias = False),
                nn.ELU(inplace=True),
                BasicBlock(64,64),  
                nn.Conv2d(64,128,kernel_size = 5,padding = 0,stride = 2,bias = False),
                nn.ELU(inplace=True),
                BasicBlock(128,128), 
                nn.Conv2d(128,512,kernel_size = 5,padding = 0,stride = 2,bias = False),
                nn.ELU(inplace=True),
                BasicBlock(512,512),
                nn.AdaptiveAvgPool2d((2,2))
        )
        self.fc = nn.Linear(args["image_embed_size"], args['cnn_output_size'], bias = False) 
#         self.sm = torch.nn.Softmax(dim = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def l2_normalization(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        norm = torch.sqrt(torch.sum(buffer, 1).add_(1e-10))
        temp = torch.div(x, norm.view(-1, 1).expand_as(x))
        x_l2 = temp.view(input_size)
        return x_l2


    def forward(self, x):
        out = self.network(x)
        out = out.view(out.size(0), -1) # flatten to N x E
        out = self.l2_normalization(out)
        out = self.fc(out) 
#         out = torch.sigmoid(out)
        return out


class XrayNet(nn.Module):
    """
    """
    def __init__(self, vocab_size, hidden_size, max_len = 250):
        global args 
        super(XrayNet, self).__init__()
        self.vocab_size = vocab_size
        self.char_embed_size = args['char_embed_size'] 
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.embedding = nn.Embedding(self.vocab_size,self.char_embed_size) # Embedding layer    
        self.softmax = nn.LogSoftmax(dim = 1)
        self.lstmcell = nn.LSTMCell(input_size = self.char_embed_size, hidden_size = hidden_size)
        self.lstmcell2 = nn.LSTMCell(input_size = hidden_size, hidden_size = hidden_size)
        self.character_distribution = nn.Linear(hidden_size, vocab_size) # Projection layer
     
    # Stepwise operation of each sequence
    def forward_step(self, input_step, hidden_cell_state, hidden_cell_state2): 
        
        embed = self.embedding(input_step)
        
        hidden_state, cell_state = self.lstmcell(embed, hidden_cell_state) # s_i   
        
        hidden_state2, cell_state2 = self.lstmcell2(hidden_state, hidden_cell_state2) # s_i

        raw_pred = self.softmax(self.character_distribution(hidden_state2))
        
        return  raw_pred, (hidden_state, cell_state), (hidden_state2, cell_state2)
    
    def forward(self, cnn_output, mode = "train", ground_truth = None, ground_truth_len = None, teacher_force = 1): 
        if ground_truth is None:
            step_size = self.max_len
            
        else:
            ground_truth_len = torch.tensor([len(g) for g in ground_truth])
            ground_truth_pad = rnn.pad_sequence(ground_truth, batch_first = True) # B * L
            step_size = ground_truth_pad.size(1)

        raw_pred_seq = []
        output_seq = []
        score = 0
        batch_size = cnn_output.size(0)
        cell_state = cnn_output
        hidden_state = torch.zeros(cell_state.shape).to(DEVICE)
        hidden_cell_state = (hidden_state, cell_state) # B x hidden
        hidden_cell_state2 = None
        
        if ground_truth is not None:
            input_step = ground_truth_pad[:,0]
        else:
#             input_step = torch.zeros(batch_size)  # (B, )
            input_step = torch.LongTensor([0 for i in range(batch_size)]).to(DEVICE)
        
        score = 0
        for step in range(step_size-1):
            
            raw_pred, hidden_cell_state, hidden_cell_state2 = self.forward_step(input_step, hidden_cell_state, hidden_cell_state2)
            
            # if train
            if mode == "train":
                raw_pred_seq.append(raw_pred.unsqueeze(1))

            elif mode == "dev":
                output = raw_pred.max(dim = 1)[1]
                raw_pred_seq.append(output.unsqueeze(1)) #(B, 1)
                if output.item() == 0:
                    break
            else:
                ######## greedy ############
#                 output = raw_pred.max(dim = 1)[1]
#                 raw_pred_seq.append(output.unsqueeze(1)) #(B, 1)
#                 if output.item() == 1:
#                     break
                #############################
  
                ######### random #############
                
                dist = torch.distributions.Categorical(logits = raw_pred) #(B, ttl_char)
                output = dist.sample() # (B, )
                score += raw_pred[0][output.item()]
                raw_pred_seq.append(output.unsqueeze(1)) #(B, 1)
                if output.item() == 0:
                    break
                ##############################
            
            if mode == "train" and np.random.rand() < teacher_force:
                input_step = ground_truth_pad[:,step+1]
            else:
                input_step = raw_pred.max(dim = 1)[1]
            
        pred_seq = torch.cat(raw_pred_seq,dim=1)  # matrix
        if mode == "train":
            pred_seq = torch.cat([pred_seq[i,:ground_truth_len[i]-1,:] for i in range(batch_size)],dim=0)
        elif mode == "dev":
            pred_seq = torch.cat([pred_seq[i,:ground_truth_len[i]-1] for i in range(batch_size)],dim=0)
        return pred_seq, score/len(pred_seq)


def train(train_loader, cnn, lstm, optimizer, criterion, DEVICE):
    global args
    cnn, lstm = cnn.train(), lstm.train()
    total_loss = 0
    for batch_id,(inputs,targets) in tqdm(enumerate(train_loader)): # lists, presorted, preloaded on GPU    # Load data
        if len(targets) == 0:
            continue
        inputs.to(DEVICE)
#         targets.to(DEVICE)      
        optimizer.zero_grad()
        # Input shape: B x C x H x W = 32 x 1 x 512 x 512
        cnn_out = cnn(inputs)
         # Output shape: B x Hidden_size = 32 x 512
        pred_y, _ = lstm(cnn_out, mode = "train", ground_truth = targets) #, teacher_force = 0.9)
        true_y = torch.cat([y[1:] for y in targets])
        loss = criterion(pred_y,true_y)
        loss.backward()
        optimizer.step()
        
        if batch_id % 500 == 0:
            lpw = loss.item() 
            print("At batch",batch_id)
            print("Training loss per word:",lpw)
            print("Training perplexity :",np.exp(lpw))
        total_loss += lpw 
    return total_loss, cnn, lstm

def validation(dev_loader, cnn, lstm, DEVICE):
    cnn, lstm = cnn.eval(), lstm.eval()
    total_dist = 0
    with torch.no_grad():
        for batch_id, (inputs,targets) in tqdm(enumerate(dev_loader)):
            if len(targets) == 0:
                continue
            inputs.to(DEVICE)
#             targets.to(DEVICE)
            cnn_out = cnn(inputs)
            pred_y, _ = lstm(cnn_out, mode = "train", ground_truth = targets) #, teacher_force = 0.9)
            pred_y = pred_y.detach().cpu().numpy()
            pred_y = np.argmax(pred_y, axis = 1)
            targets = targets[0].detach().cpu().numpy()
            targets_seq = []
            pred_seq = []
            targets_seq = int_to_str(targets)
            pred_seq = int_to_str(pred_y)
            dist =  distance(targets_seq, pred_seq)
            if batch_id % 1 == 0:
                print("At batch",batch_id)
                print("Validation Distance:",dist)
            total_dist += dist
            print("[Validation] pred sample: {}, target: {}".format(pred_seq, targets_seq))
    return total_dist/ (batch_id + 1)
