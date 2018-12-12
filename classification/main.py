import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import roc_curve, auc
import numpy as np
import preprocess
from tqdm import tqdm

args = {}
args["train_subsample"] = -1
args["val_subsample"]   = -1
args["test_subsample"]  = -1
args["batch_size"]      = 8
args["epochs"]          = 2

args["label_cutoff"]    = 0.5 # arbitrary label cutoff prob threshold
args["num_workers"]     = 4
args["embed_size"]      = 512
args["num_classes"]     = 15
args["device"]          = "cuda" if torch.cuda.is_available() else "cpu"

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
                nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(args["embed_size"], args["num_classes"], bias = False)
        self.bn = nn.BatchNorm1d(args["num_classes"])
        self.sigmoid = nn.Sigmoid()

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
        # out = self.l2_normalization(out)
        # alpha = 16
        # out = out * alpha
        out = self.fc(out) 
        out = self.bn(out)
        out = self.sigmoid(out)
        return out

def calc_iou(pred, target):
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
    batch_iou = iou_tensor.mean().cpu().item() # total iou over a batch 
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
        batch_data = batch_data.to(args["device"])
        batch_label = batch_label.to(args["device"])
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

def validation(net, val_set):
    global args
    net = net.eval()
    with torch.no_grad():
        total_iou = 0
        for batch_index, (batch_data, batch_label) in enumerate(tqdm(val_set)):
            batch_data = batch_data.to(args["device"])
            out = net(batch_data)
            out = out.detach().cpu().numpy()

            # select only top k as predicted class
            # top_k, indices = torch.topk(out, k=args["k"], dim=1) # top k max classes
            # out.zero_()
            # out.scatter_(1, indices, top_k)     # clip non-top-k to be zer

            out[out < args["label_cutoff"]] = 0     # only select those exceeding the threshold as label
            out[out >= args["label_cutoff"]] = 1

            # remove No Finding from pred if necessary
            # np.apply_along_axis(remove_null, 1, out)

            pred_one_hot = torch.from_numpy(out)
            batch_iou = calc_iou(pred_one_hot, batch_label) # average iou score over a batch
            total_iou += batch_iou
    acc = total_iou/(batch_index + 1) # average iou
    return acc

def train_val(net, train_set, val_set):
    global args
    # criterion = CustomLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode = "max", factor = 0.1, patience = 1) #reduce lr once acc stop increasing
    net = net.to(args["device"])
    criterion = criterion.to(args["device"])

    best_iou = -float("inf")
    train_loss, val_iou = [], []
    for epoch in range(args["epochs"]):
        # train
        loss = train(net, epoch, train_set, criterion, optimizer)
        train_loss.append(loss)
        print("[Epoch {}] Train loss: {}".format(epoch, loss))

        # validation
        epoch_iou = validation(net, val_set)
        val_iou.append(epoch_iou)
        print("[Epoch {}] Val iou: {}".format(epoch, epoch_iou))

        # step learning rate
        scheduler.step(epoch_iou)

        # save model if best
        if epoch_iou > best_iou:
            print("crt: {}, best: {}, saving...".format(epoch_iou, best_iou))
            best_iou = epoch_iou
            torch.save(net, "saved_models/epoch{}".format(epoch))
    return train_loss, val_iou

def roc_auc(net):
    test_data, test_label = get_testdata()
    # get scores
    scores = []
    with torch.no_grad():
        for batch_id, (batch_data, _) in tqdm(enumerate(test_data)):
            out = net(batch_data)
            scores.append(out.detach().cpu().numpy())
    scores = np.concatenate(scores, axis = 0) #(N, num_classes)

    # compute roc_curve
    fpr, tpr = dict(), dict()
    roc_auc = dict()
    for i, dis in enumerate(OHE_MAPPING):
        fpr[dis], tpr[dis] = roc_curve(test_label[:, i], scores[:, i])
        roc_auc[dis] = auc(fpr[dis], tpr[dis])
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plots
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=8)
    colors = ["crimson", "red", "salmon", "coral", "wheat", "gold", "orange", "olive", "lime", "aqua", "azure", "blue", "navy", "violet", "purple"]
    for dis, color in zip(OHE_MAPPING, colors):
        plt.plot(fpr[dis], tpr[dis], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(dis, roc_auc[dis]))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve of various disease')
    ax.legend(loc="lower right")
    plt.savefig("figures/fig.png")

def main():
    print("Running with {}".format(args))

    # load train and val set
    train_set = preprocess.get_traindata(args["batch_size"], args["num_workers"], args["train_subsample"])
    val_set = preprocess.get_valdata(args["batch_size"], args["num_workers"], args["val_subsample"])

    # get net and train
    # net = torch.load("")
    net = XrayNet()
    train_loss, val_acc = train_val(net, train_set, val_set)

    # test
    test_set = preprocess.get_testdata(args["batch_size"], args["num_workers"], args["test_subsample"])
    validation(net, test_set)

if __name__ == "__main__":
    main()
    
