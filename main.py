import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np
import preprocess
from tqdm import tqdm

args = {}
args["batch_size"] = 30
args["epochs"] = 15
args["num_workers"] = 4
args["embed_size"] = 2048
args["gpu"] = True
if (not torch.cuda.is_available()): args["gpu"] = False
args["label_cutoff"] = 0.2 # minimum probability of a softmax output for a valid label
args["k"] = 4 # select top k softmax outputs as labels

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
NOFINDING_IDX = OHE_MAPPING.index("No Finding")

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
        super(XrayNet, self).__init__()
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
        self.fc = nn.Linear(args["embed_size"], 15)
        # self.sm = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.network(x)
        out = out.view(out.size(0), -1) # flatten to N x E
        out = self.fc(out) 
        out = torch.sigmoid(out)
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

def remove_null(row):
    """
    @Param:
        one_hot: 1d one hot numpy array  
    """ 
    if row[NOFINDING_IDX] == 1 and row.sum() > 1:
        row[NOFINDING_IDX] = 0

def train(net, epoch_id, train_set, criterion, optimizer):
    global args
    net = net.train()
    ttl_loss = 0
    for batch_index, (batch_data, batch_label) in enumerate(tqdm(train_set)):
        if args["gpu"]:
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        optimizer.zero_grad()
        out = net(batch_data) 
        loss = criterion(out, batch_label)
        loss.backward()
        optimizer.step()

        ttl_loss += loss.item()
        if(batch_index % 100 == 0):
            print("[Epoch {} batch {}] Loss: {}".format(epoch_id, 
                    batch_index,
                    ttl_loss/((batch_index + 1) * args["batch_size"])))
    return ttl_loss/((batch_index + 1) * args["batch_size"])

def val_test(net, val_set):
    global args
    net = net.eval()
    with torch.no_grad():
        total_iou = 0
        for batch_index, (batch_data, batch_label) in enumerate(tqdm(val_set)):
            if args["gpu"]:
                batch_data = batch_data.cuda()
            out = net(batch_data)
            out = out.detach().cpu()
            top_k, indices = torch.topk(out, k=args["k"], dim=1) # top k max classes
            out.zero_()
            out.scatter_(1, indices, top_k)     # clip non-top-k to be zero
            out[out < args["label_cutoff"]] = 0 # clip those non-exceeding threshold to be zero
            out = out.numpy()
            # remove No Finding from pred if necessary
            np.apply_along_axis(remove_null, 1, out)
            pred_one_hot = torch.from_numpy(out)
            batch_iou = iou(pred_one_hot, batch_label) # average iou score over a batch
            total_iou += batch_iou
        acc = total_iou/((batch_index+1)*args["batch_size"])# average iou
        print("Acc: {}".format(acc))
        return acc

def train_val(net, train_set, val_set):
    global args
    # criterion = CustomLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode = "max", factor = 0.1, patience = 1) #reduce lr once acc stop increasing
    if args["gpu"]:
        net = net.cuda()
        criterion = criterion.cuda()

    best_acc = -float("inf")
    train_loss = []; val_acc = []
    for epoch in range(args["epochs"]):
        # train
        loss = train(net, epoch, train_set, criterion, optimizer)
        train_loss.append(loss)

        # validation
        acc = val_test(net, val_set)
        val_acc.append(acc)

        # step learning rate
        scheduler.step(acc)

        # save model if best
        if acc > best_acc:
            print("crt: {}, best: {}, saving...".format(acc, best_acc))
            best_acc = acc
            torch.save(net, "epoch{}".format(epoch))
    return train_loss, val_acc

def main():

    # load train and val set
    train_set = preprocess.get_traindata(args["batch_size"], args["num_workers"])
    val_set = preprocess.get_valdata(args["batch_size"], args["num_workers"])

    # get net and train
    # net = torch.load("")
    net = XrayNet()
    train_loss, val_acc = train_val(net, train_set, val_set)

    # test
    test_set = preprocess.get_testdata(args["batch_size"], args["num_workers"])
    val_test(net, test_set)

if __name__ == "__main__":
    main()
    
