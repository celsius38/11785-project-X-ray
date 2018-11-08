from torch.utils.data.dataset import Dataset
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
  
class XrayNet(nn.Module):
  	"""
	tunable hyper parameters: embeddings
    """
	def __init__(self):
      	global args
		self.blocks = None
      	self.pool = AdaptivePool2D((1,1)) # 
        self.fc = nn.Linear(xxx, args["embed_size"])
        
  	def forward():
      	#TODO: output size of embedding
      	pass

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
    criterion = CustomLoss()
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
    args["embed_size"] = 1024,
    args["gpu"] = True
    args["ttl_trainsets"] = 3
    args["null_percent"] = 0.2 # specify the percentage of No finding in the dataset
    if (not torch.cuda.is_available()): args["gpu"] = False
  	
