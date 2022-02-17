import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def accuracy(outputs,label):
    _, pred = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(pred==label).item()/len(pred))

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)
        

class Image_Classifcation_Base(nn.Module):

    def training_step(self,batch):
        img, label = batch
        out = self(img)
        loss = F.cross_entropy(input=out,target=label)
        return loss

    def validation_step(self,batch):
        img, label = batch
        out = self(img)
        loss = F.cross_entropy(input=out,target=label)
        acc = accuracy(outputs=out,label=label)
        return {'val_loss':loss.detach(),'val_acc':acc}

    def validation_epoch_end(self,output):
        batch_loss = [x['val_loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in output]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item()}

    def epoch_end(self,epoch_no,result):
        print('Epoch[{}], Train Loss :- {:4f}, Val Loss :- {:4f}, Val Accuracy :- {:4f}'
        .format(epoch_no,result['train_loss'],result['val_loss'],result['val_acc']))


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def save_model(model,epoch):
    torch.save(model.state_dict(),'models/cifar10-cnn-{}-model.pth'.format(epoch))
