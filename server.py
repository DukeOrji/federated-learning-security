#server.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
from user import norm

class Server:
    def __init__(self):
        self.weights = ResNet18_Weights.DEFAULT
        self.global_model = resnet18(weights=None)
        self.global_model.fc = nn.Linear(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()

        
    def broadcast_weights(self):
        return self.global_model.state_dict()
        
    def aggregate(self, user_weights):
        avg_weights = {}
        for key in user_weights[0].keys():
            if user_weights[0][key].dtype == torch.float32:
                avg_weights[key] = torch.stack(
                    [user[key] for user in user_weights],
                    dim=0
                ).mean(dim=0)
                
            else:
                avg_weights[key] = user_weights[0][key]
        self.global_model.load_state_dict(avg_weights)
        print("\nAggregation complete...")   


    def label_poison_evaluate(self, dataloader):
        self.global_model.eval()

        correct= 0
        zero_to_one= 0
        class0_total= 0

        with torch.no_grad():
            for images, labels in dataloader:
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)
                mask = labels == 0

                #check for changes in the global model parameters
                correct += (pred_labels[mask] == labels[mask]).sum().item() 

                zero_to_one += (pred_labels[mask] == 1).sum().item()
                class0_total += mask.sum().item()

        #How often does the global model classify class 0 as class 1?
        poison_rate = round(zero_to_one/class0_total, 2)
        #How often does the global model correctly classify class 0?
        class_acc = round(correct/class0_total, 2)

        return poison_rate, class_acc
    
    def weight_man_evaluate(self, dataloader):
        losses=[]
        self.global_model.eval()

        correct= 0
        total= 0

        with torch.no_grad():
            for images, labels in dataloader:
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
                losses.append(loss.item())

        avg_loss = round((sum(losses)/len(losses)), 2)
        global_acc = round(correct/total, 2)
            

        return avg_loss, global_acc