#user.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
from config import device


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (
            x - self.mean.type_as(x)[None,:,None,None]
        ) / self.std.type_as(x)[None,:,None,None]
    
norm = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


class User:
    def __init__(self, user_id, dataloader):
        self.user_id = user_id
        self.dataloader = dataloader
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = self.model.to(device)
        self.model.fc = nn.Linear(512, 10).to(device)
        self.optim = optim.SGD(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()
        #self.eps = 0.3        

    def train(self):
        # put model in training mode
        self.model.train()
        losses = []
        clean_correct = 0 
        total = 0

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > 20:
                break

            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self.model(norm(images))
            #track correct clean predictions
            
            

            pred_labels = pred.argmax(dim=1)

            clean_correct += (pred_labels == labels).sum().item() 
            loss = self.loss_fn(pred, labels)
            total += labels.size(0)

            self.optim.zero_grad()
            loss.backward() 
            self.optim.step()
            losses.append(loss.item())

        clean_acc = round((clean_correct/total), 2)

        avg_loss = round((sum(losses)/len(losses)), 2)

        return avg_loss, clean_acc
    
    def get_weight(self):
        return self.model.state_dict()
    
    def set_weight(self, new_weights):
        #store a copy of the global model
        self.global_weights = {
            k: v.clone()
            for k, v in new_weights.items()
        }
        
        #load the weights into the local model
        self.model.load_state_dict(new_weights)
