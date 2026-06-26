#user.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#from art.estimators.classification import PyTorchClassifier
# https://github.com/DukeOrji/federated-learning-security.git

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
            if batch_idx > 33:
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
        self.model.load_state_dict(new_weights)

class LabelPoison(User):
    def train(self):
        losses = []
        self.model.train()

        # track how often model predicts the poisoned labels correctly
        correct_pred = 0
        # track total samples processed
        total = 0

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > 3:
                break
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            # create a copy so original labels remain unchanged
            poisoned_labels = labels.clone()

            # label poisoning attack:
            # change every class 0 label into class 1
            poisoned_labels[labels==0] = 1

            pred = self.model(norm(images))
            loss = self.loss_fn(pred, poisoned_labels)
            pred_labels = pred.argmax(dim=1)

            # measure how well the model learns the poisoned mapping
            correct_pred += (pred_labels==poisoned_labels).sum().item()
            total += labels.size(0)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())

        p_acc = round((correct_pred/total), 2)
        avg_loss = round(sum(losses)/len(losses), 2)

        return avg_loss, p_acc
    
class WeightMan(User): 
    def get_weight(self):
        weights = {
            k: v.clone() #confirm only a copy is being modified rather than the local model itself
            for k, v in self.model.state_dict().items()
        }
        for key in weights:
            #confirm only learned parameters are beign manipulated
            if weights[key].dtype == torch.float32:
                weights[key]*=2
        return weights
    
class SignFlip(User):
    def get_weight(self):
        weights = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
        }
        for key in weights:
            if weights[key].dtype == torch.float32:
                weights[key]*= -1
        return weights

