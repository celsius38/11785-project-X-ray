import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from tqdm import tqdm
from torch.nn.utils.rnn import (pad_packed_sequence,
                                pad_sequence,
                                pack_padded_sequence,
                                pack_sequence)
from preprocess import INT_TO_CHAR, CHAR_TO_INT, int_to_str, str_to_int
import sys
import Levenshtein

SOS = 0
EOS = 0

args = {}
args["train_subsample"]     = 4
args["val_subsample"]       = -1
args["batch_size"]          = 2
args["lr"]                  = 1e-4
args["max_step"]            = 250
args["random_sample"]       = 20
args["epochs"]              = 15

# rather fixed
args["num_workers"]         = 4
args["device"]              = "cuda" if torch.cuda.is_available() else "cpu"
args["vocab_size"]          = 58

# model hyper parameters
args["image_embed_size"]    = 2048
args["cnn_output_size"]     = 512
args["lstm_hidden_size"]    = 512
args["char_embed_size"]     = 256

class CustomDataset(Dataset):
    def __init__(self, data, label = None):
        self._data = data
        # append <eos> for each label
        label = [np.append(y,0) for y in label]
        self._label = label

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        global args
        d =  self._data[index]
        d = torch.from_numpy(d).float()
        l = torch.tensor([0])
        if self._label is not None:
            l = torch.from_numpy(self._label[index]).long()
        return (d, l)
    
def collate_lines(batch):
    """
    @Param:
        batch: list of tensor tuple: (data, label) of len B
    @Return 
        data        : (B, 1, W, H) (channel is 1) 
        target      : padded_seq (B, L)
        target_len  : tensor (B, )
    """
    batch   = sorted(batch, key = lambda x: len(x[0]), reverse = True)
    data    = [b[0].unsqueeze(0) for b in batch] # B of (1, W, H)
    data    = torch.cat(data, dim = 0).unsqueeze(1) #(B, 1, W, H)
    target  = [b[1] for b in batch] # B of (L, )
    target_len = torch.tensor([len(t) for t in target])
    target  = pad_sequence(target, batch_first = True)
    return data, target, target_len

def load_data():
    global args
    data        = np.load('data/data.npy')
    findings    = np.load('data/findings.npy')
    indications = np.load('data/indications.npy')
    impressions = np.load('data/impressions.npy')

    none_index = []
    for i, f in enumerate(findings):
        if len(f) == 0:
            none_index.append(i)
    full_index = set(np.arange(len(findings)))

    # train, dev, test split
    idx = list(full_index - set(none_index))
    np.random.shuffle(idx)
    total = len(idx)
    train_idx, dev_idx, test_idx = idx[:int(total*0.8)],idx[int(total*0.8):int(total*0.9)],idx[int(total*0.9):] # 8 : 1 : 1

    # TODO: use findings 
    train_x, train_y= (data[train_idx][:args["train_subsample"]], 
                       findings[train_idx][:args["train_subsample"]])
    dev_x, dev_y    = (data[dev_idx][:args["val_subsample"]],
                       findings[dev_idx][:args["val_subsample"]])
    test_x, test_y = data[test_idx], findings[test_idx]

    train_set   = CustomDataset(train_x, train_y)
    val_set     = CustomDataset(dev_x, dev_y)
    test_set    = CustomDataset(test_x, test_y)
    return (DataLoader( dataset=train_set, 
                        batch_size=args["batch_size"],
                        shuffle=True,
                        collate_fn=collate_lines), 
            DataLoader( dataset=val_set,
                        batch_size=args["batch_size"],
                        shuffle=False,
                        collate_fn=collate_lines),
            DataLoader( dataset=test_set,
                        batch_size=args["batch_size"],
                        shuffle=False,
                        collate_fn=collate_lines))

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
        return out


class XrayNet(nn.Module):
    def __init__(self):
        global args 
        super(XrayNet, self).__init__()
        vocab_size          = args["vocab_size"]
        embed_size          = args["char_embed_size"]
        hidden_size         = args["lstm_hidden_size"]
        self.embedding      = nn.Embedding(vocab_size, embed_size)
        self.logsoftmax     = nn.LogSoftmax(dim = 1)
        self.lstmcell       = nn.LSTMCell(embed_size, hidden_size)
        self.lstmcell2      = nn.LSTMCell(hidden_size, hidden_size)
        self.character_distribution = nn.Linear(hidden_size, vocab_size) # Projection layer
     
    def forward_step(self, input_step, hidden_cell_state, hidden_cell_state2):    
        """
        @Param: 
            input_step: (B, ) input chars
            hidden_cell_state: (hidden_state,  cell_state), both (B, lstm_hidden_size)
            hidden_cell_state2:(hidden_state2, cell_state2), both (B, lstm_hidden_size)
        @Return:
            raw_pred:  (B, vocab_size)
            hidden_state, cell_state    (B, lstm_hidden_size):
            hidden_state2, cell_state2  (B, lstm_hidden_size):
        """
        embed = self.embedding(input_step)
        hidden_state, cell_state = self.lstmcell(embed, hidden_cell_state)              #(B, H)
        hidden_state2, cell_state2 = self.lstmcell2(hidden_state, hidden_cell_state2)   #(B, H)
        raw_pred = self.logsoftmax(self.character_distribution(hidden_state2))             #(B, V)
        return raw_pred, (hidden_state, cell_state), (hidden_state2, cell_state2)
    
    def forward(self, cnn_output, mode, 
                ground_truth = None, teacher_force = 0.9): 
        """
        @Param:
            cnn_output: (B, H)
            mode:  "train", "test" or "val"
            ground_truth        : padded seq of (B, L)
        @Return:
            raw_pred_seq    : prob dist of each char (B, L, vocab_size) 
            output_seq      : list of B of (L, ), string for each batch
            ttl_score       : list of B of scalar, cumulative score for each batch
        """
        if mode == "train":
            max_step = ground_truth.size(1)
        else:
            max_step = args["max_step"]
        batch_size = cnn_output.size(0)

        raw_pred_seq  = []
        output_seq    = []
        all_score = []

        # initialize first hidden state with cnn output
        cell_state = cnn_output
        hidden_state = torch.zeros(cell_state.shape).to(args["device"])
        hidden_cell_state = (hidden_state, cell_state) # B x hidden
        hidden_cell_state2 = None
        
        # initialize first char as <sos>
        input_step = torch.tensor([SOS] * batch_size).to(args["device"])

        for step in range(max_step):
            raw_pred, hidden_cell_state, hidden_cell_state2 = (
                self.forward_step(input_step, hidden_cell_state, hidden_cell_state2))

            if mode == "train":
                raw_pred_seq.append(raw_pred.unsqueeze(1)) #(B, 1, vocab_size)

            # generate output
            # greedy
            if mode == "train" or mode == "val":
                output = raw_pred.max(dim = 1)[1] # argmax (B, )
                if mode == "val":
                    output_seq.append(output.unsqueeze(1).cpu().detach())
                    all_score.append(torch.gather(raw_pred, 1, output.view(-1, 1))) #(B, 1)
            # random
            else:
                dist = torch.distributions.Categorical(logits=raw_pred)
                output = dist.sample() #(B, )
                output_seq.append(output.unsqueeze(1).cpu().detach()) #(B, 1)
                all_score.append(torch.gather(raw_pred, 1, output.view(-1, 1))) #(B, 1)
            
            if mode == "train" and np.random.rand() < teacher_force:
                input_step = ground_truth[:,step] 
            else:
                input_step = output

        if mode == "train":
            raw_pred_seq    = torch.cat(raw_pred_seq, dim=1) #(B, L, vocab_size)
        if mode == "val" or mode == "test": # calculate loss and each output length
            output_seq  = torch.cat(output_seq, dim=1) #(B, L)
            all_score   = torch.cat(all_score, dim=1)  #(B, L)
            output_fixed = []
            score_fixed  = []
            for output, score in zip(output_seq, all_score):
                idx = (output == EOS).nonzero()
                if len(idx) == 0: # no <EOS> contained, until final
                    output_fixed.append(output)
                    score_fixed.append(score.sum().item()/max_step)
                else: #
                    output_fixed.append(output[:idx[0] + 1])
                    score_fixed.append(score[:idx[0] + 1].mean())

        if mode == "train":
            return raw_pred_seq, None, None
        else:
            return None, output_fixed, score_fixed

def train(epoch, cnn, lstm, train_loader, optimizer, criterion):
    global args
    cnn, lstm = cnn.train(), lstm.train()
    ttl_perplexity = 0
    ttl_loss = 0
    for batch_id, (inputs, targets, target_len) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.to(args["device"]), targets.to(args["device"])
        
        # Input shape: (B, C, H, W) = (B, 1, 512, 512)
        cnn_out = cnn(inputs) # (B, cnn_output_size)
        # TODO:
        raw_pred_seq, _, _  = lstm(cnn_out, mode = "train", ground_truth = targets) #, teacher_force = 0.9)

        # mask the padding part of generated seq to be -1 and ignore for loss
        comp_range      = torch.arange(target_len.max().item()).unsqueeze(0)
        transript_mask  = target_len.unsqueeze(1)
        transript_mask  = (transript_mask <= comp_range).to(args["device"])
        targets_masked  = targets.clone()
        targets_masked[transript_mask] = -1
        
        # backward pass
        loss = criterion(raw_pred_seq.view(-1, args["vocab_size"]),
                         targets_masked.view(-1))
        perplexity = (loss / ((1 - transript_mask).sum().item())).exp()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ttl_loss = ttl_loss + loss.item()/args["batch_size"] # per utter loss
        ttl_perplexity = ttl_perplexity + perplexity # exp loss per char
        if batch_id > 0 and batch_id % 50 == 0:
            print("[Epoch{} batch {}] loss: {} perplexity: {}".format(
                    epoch, batch_id, 
                    ttl_loss/(batch_id +1), ttl_perplexity/(batch_id + 1)))
    return ttl_loss/(batch_id + 1), ttl_perplexity/(batch_id + 1)

def validation(cnn, lstm, dev_loader):
    global args
    cnn, lstm   = cnn.eval(), lstm.eval()
    ttl_dist    = 0
    with torch.no_grad():
        for batch_id, (inputs, targets, _) in tqdm(enumerate(dev_loader)):
            inputs = inputs.to(args["device"])
            cnn_out = cnn(inputs)
            _, output_seq, _ = lstm(cnn_out, mode = "val")

            # translate to string
            output_seq = [int_to_str(out.numpy()) for out in output_seq]
            targets_seq = [int_to_str(tar.numpy()) for tar in targets]
            # comp distance
            dist = [Levenshtein.distance(out, tar) for out, tar in zip(output_seq, targets_seq)]
            ttl_dist += np.mean(dist)
    print("[Validation] pred sample: {}, target: {}".format(output_seq, targets_seq))
    return ttl_dist/ (batch_id + 1)

def test(cnn, lstm, test_loader):
    global args
    cnn, lstm = cnn.eval(), lstm.eval()
    predictions = []

    with torch.no_grad():
        for batch_id, (inputs, targets, _) in tqdm(enumerate(test_loader)):
            inputs.to(args["device"])
            cnn_out = cnn(inputs)

            # use greedy search as best for now
            _, best_outs, best_scores = lstm(cnn_out, mode = "val")

            # use random search
            for i in range(args["random_sample"]):
                _, outs, scores = lstm(cnn_out, mode = "test")
                for idx, (out, score) in enumerate(zip(outs, scores)):
                    if score > best_scores[idx]:
                        best_outs[idx]      = out
                        best_scores[idx]    = score
            predictions.extend(list(map(int_to_str, predictions)))
    return predictions

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    cnn = ResNet().to(args["device"])
    lstm = XrayNet().to(args["device"])
    optimizer = torch.optim.Adam([{'params':cnn.parameters()}, 
                                    {'params':lstm.parameters()}],
                                    lr = args["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    mode="min", factor=0.1, patience=5,
                    min_lr=1e-6)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=-1).to(args["device"])
    best_dist = float("inf")

    for epoch in range(args["epochs"]):
        # train
        epoch_loss, epoch_perplexity = (
                train(epoch, cnn, lstm, train_loader, optimizer, criterion))
        print("[Epoch {}] loss: {}, perplexity: {}".format(
                        epoch, epoch_loss, epoch_perplexity))

        # val
        dist = validation(cnn, lstm, val_loader)
        print("[Validation] Levenshtein distance:", dist)
            
        # step lr
        scheduler.step(dist)
        if dist < best_dist:
            print("crt: {}, best: {}, saving...".format(dist, best_dist))
            best_dist = dist
            torch.save(cnn, "cnn_{}.pt".format(epoch))
            torch.save(lstm, "lstm_{}.pt".format(epoch))
    test(cnn, lstm, test_loader)
