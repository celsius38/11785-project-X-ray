import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from preprocess import get_traindata, get_valdata, get_testdata

args = {}

# one hot encoding mapping
OHE_MAPPING = ['Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Hernia',
        'Infiltration',
        'Mass',
        'No Finding',
        'Nodule',
        'Pleural_Thickening',
        'Pneumonia',
        'Pneumothorax']

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

class XrayNet(nn.Module):
    """
    tunable hyper parameters: embeddings
    """
    def __init__(self):
        global args
        # self.blocks = None
        # self.pool = AdaptivePool2D((2,2)) # 
        # self.fc = nn.Linear(args["embed_size"], 15)

        self.network = Sequential(
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
                nn.AdaptiveAvgPool2d((2,2)))
        self.fc = nn.Linear(args["embed_size"], 15, bias = False) 
        self.sm = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        out = self.network(self.x) 
        out = out.view(out.size(0), -1) # flatten to N x E
        out = self.fc(out) 
        out = self.sm(out) # softmax for prob interpretation
        return out

def iou(pred, target):
    """
    Compute IoU of two tensors with same dimensions
    @Param:
        pred: a 2d tensor
        target: a 2d tensor
    """
    pred, target = pred.float(), target.float()
    intersection_array = (pred * target)
    union_array = pred + target - intersection_array
    intersection = intersection_array.sum(dim = 1)
    union = union_array.sum(dim = 1)
    iou_tensor = intersection/union # a tensor array of iou's
    batch_iou = float(iou_tensor.sum()) # total iou over a batch 

    return batch_iou

def remove_null(one_hot, idx_null):
    """
    @Param:
        one_hot: 1d one hot numpy array  
        idx_null: index of "No Finding" class
    """
    if pred_row[idx_null] == 1 and pred_row.sum()>1:
        pred_row[idx_null] = 0
    

def train(net, epoch_id, train_set, criterion, optimizer):
    global args
    net = net.train()
    ttl_loss = 0
    for batch_index, (batch_data, batch_label) in enumerate(train_set):
        if args["gpu"]:
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()

        out = net(batch_data)
        optimizer.zero_grad()
        loss = criterion(out, batch_label)
        loss.backward()
        optimizer.step()

        ttl_loss += loss.item()
        if(batch_index % 100 == 0):
            print("[Epoch {}] Loss: {}".format(epoch_id, ttl_loss/((batch_index + 1) * args["batch_size"])))
    return ttl_loss/((batch_index + 1) * args["batch_size"])

def val(net, val_set):
    global args
    net = net.eval()
    with torch.no_grad():
        total_iou = 0
        total_sample = 0
        for batch_index, (batch_data, batch_label) in enumerate(val_set):
            if args["gpu"]:
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            out = net(batch_data)
            out_k,label_pred_k = torch.topk(out, k=args["k"], dim=1) # top k max classes
            pred = []
            for idx in len(out_k):
                pred_sample = [label for label in label_pred_k[idx] if out_k[idx][label] >= args["label_cutoff"]] # valid labels have softmax >= cutoff 
                pred.append(pred_sample)
            # convert pred to one hot
            pred_one_hot = torch.zeros(out.size()).int()
            for i in range(len(pred)):
                for j in pred[i]:
                    pred_one_hot[i][j] = 1
            pred_one_hot = pred_one_hot.numpy()
            # remove No Finding from pred if necessary
            np.apply_along_axis(remove_null, 1, pred_one_hot, ONE_MAPPING.index("No Finding"))
            pred_one_hot = torch.from_numpy(pred_one_hot)
            batch_iou = iou(pred_one_hot, batch_label) # average iou score over a batch
            total_iou += batch_iou
            total_sample += batch_label.size(0)
        acc = total_iou/total_sample # average iou
        print("Acc: {}".format(acc))
        return acc

def test(net, test_set):
    global args
    net = net.eval()

    with torch.no_grad():
        pred_label = []
        for batch_index, (batch_data, _) in enumerate(test_set):
            if args["gpu"]:
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            out = net(batch_data)
            out_k,label_pred_k = torch.topk(out, k=args["k"], dim=1) # top k max classes
            pred = []
            for idx in len(out_k):
                pred_sample = [label for label in label_pred_k[idx] if out_k[idx][label] >= args["label_cutoff"]] # valid labels have softmax >= cutoff 
                pred.append(pred_sample)
            # convert pred to one hot
            pred_one_hot = torch.zeros(out.size()).int()
            for i in range(len(pred)):
                for j in pred[i]:
                    pred_one_hot[i][j] = 1
            pred_one_hot = torch.zeros(out.size()).int()
            pred_one_hot = pred_one_hot.numpy()
            # remove No Finding from pred if necessary
            np.apply_along_axis(remove_null, 1, pred_one_hot, ONE_MAPPING.index("No Finding"))
            pred_one_hot = torch.from_numpy(pred_one_hot)
            batch_iou = iou(pred_one_hot, batch_label) # average iou score over a batch
            total_iou += batch_iou
            total_sample += batch_label.size(0)
        acc = total_iou/total_sample # average iou
        print("Acc: {}".format(acc))
        return acc


def train_val(net, train_set, val_set):
    global args
    # criterion = CustomLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999),
            weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode = "max", factor = 0.1, patience = 1) #reduce lr once acc stop increasing


    best_acc = -float("inf")
    if args["gpu"]:
        net = net.cuda()
        criterion = criterion.cuda()

    for epoch in args["epochs"]:
        # train
        loss = train(net, epoch, train_set, criterion, optimizer)

        # validation
        acc = val(net, val_set)

        # step learning rate
        scheduler.step(acc)

        # save model if best
        if acc > best_acc:
            print("crt: {}, best: {}, saving...".format(dist, best_dist))
            best_acc = acc
            torch.save(net, "epoch{}".format(epoch))


def main():
    args["batch_size"] = 20
    args["epochs"] = 20
    args["num_workers"] = 4
    args["embed_size"] = 2048
    args["gpu"] = True
    if (not torch.cuda.is_available()): args["gpu"] = False
    args["null_percent"] = 0.2 # specify the percentage of No finding in the dataset
    args["label_cutoff"] = 0.3 # minimum probability of a softmax output for a valid label
    args["k"] = 3 # select top k softmax outputs as labels

    # load train and val set
    train_set = get_traindata()
    val_set = get_valdata()

    # get net and train
    # net = torch.load("")
    net = XrayNet()
    train_loss, train_acc, val_err = train_val(net, train_set, val_set)

    # test
    test_set = get_testdata()
    test(net, test_set)

if __name__ == "__main__":
    main()
    
