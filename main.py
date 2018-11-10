from torch.utils.data.dataset import Dataset
import torch.nn as nn
args = {}


class CustomDataset(Dataset):
    def __init__(self, data, label = None):
        pass
    def __len__(self):
        pass
    def __getitem(self, index):
        pass
  

def get_traindata(i):
    """
    @Param:
      int i: index of train set
    @Return:
      DataLoader of train data with index i
    """
    DataLoader(CustomDataset(train_data, train_label), 
                        batch_size = args["batch_size"], 
                        shuffle = True, 
                        drop_last = True,
                        num_workers = args["num_workers"],
                        collate_fn = collate_fn)
  
def get_valdata():
    """
    @Return:
      DataLoader of validation data
    """
    pass
  
def get_testdata():
    """
    @Return:
      DataLoader of test data
    """
    pass






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

class SpeechNetwork(nn.Module):
  def __init__(self): 
    super(SpeechNetwork, self).__init__()        
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
      nn.AdaptiveAvgPool2d((2,2))
      Flatten()
      nn.Linear(2048,15,bias = False)

    )

  def forward(self, x):
    final_classes = self.network(x)
    return final_classes


  
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
      nn.AdaptiveAvgPool2d((2,2))
      Flatten()
      nn.Linear(args["embed_size"], 15 ,bias = False)

    )
        
    def forward(self, x):
      #TODO: output size of embedding
      final_classes = self.network(self.x)
      return final_classes

class CustomLoss(nn.Module):
    # TODO:
    pass

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
    # TODO: validation returning some accuracy measure
    return acc

def test(net, test_set):
    global args
    net = net.eval()
    with torch.no_grad():
        pass
    
def train_val(net):
    global args
    # criterion = CustomLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999),
                                    weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                    mode = "max", factor = 0.1, patience = 1) #reduce lr once acc stop increasing
    val_set = get_valdata()
    best_acc = -float("inf")
    if args["gpu"]:
        net = net.cuda()
        criterion = criterion.cuda()
    
    for epoch in args["epochs"]:
        # train on every set of train data
        for i in range(args["ttl_trainsets"]):
          train_set = get_traindata(i)
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
    
    test_set = get_testdata()
    test(net, test_set)
        
if __name__ == "__main__":
    global args
    args["batch_size"] = 40
    args["epochs"] = 20
    args["embed_size"] = 2048,
    args["gpu"] = True
    args["ttl_trainsets"] = 3
    args["null_percent"] = 0.2 # specify the percentage of No finding in the dataset
    if (not torch.cuda.is_available()): args["gpu"] = False
    
