
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from art.estimators.classification import PyTorchClassifier

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
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 10)
        self.optim = optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.eps = 0.3        

    def train(self):
        # put model in training mode
        self.model.train()
        losses = []

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > 3:
                break

            pred = self.model(norm(images))
            #track correct clean predictions
            clean_correct = 0 
            total = 0
            

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

class LabelPoisonUser(User): 
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


class Server:
    def __init__(self):
        self.weights = ResNet18_Weights.DEFAULT
        self.global_model = resnet18(weights=None)
        self.global_model.fc = nn.Linear(512, 10)

        
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


    def evaluate(self, dataloader):
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



server = Server()
conversion = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor()
])



#initialize CIFAR10 dataset
cifar_train = datasets.CIFAR10(
    root = "./data",
    train=True,
    download=True,
    transform=conversion
)
#modify the length of dataset
small_dataset = Subset(
    cifar_train,
    range(900)
)

#classes = cifar_train.classes

#split dataset for users
user_datasets = random_split(
    small_dataset,
    [300, 300, 300]
)

#each user loads a different dataset
user_dataloader = [DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
) for dataset in user_datasets]

#create a test dataset
cifar_test = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=conversion
)

test_loader = DataLoader(
    cifar_test,
    batch_size=64,
    shuffle=False
)

#create clients
users = [
    User(0, user_dataloader[0]),
    User(1, user_dataloader[1]),
    LabelPoisonUser(2, user_dataloader[2])
]


#initialize multiple rounds - improve accuracy
for epoch in range(10):
    print(f"\nRound {epoch+1}")
    #broadcast global model
    global_weights = server.broadcast_weights()

    #send weights to users
    for user in users:
        user.set_weight(global_weights)
        loss, acc = user.train()

        if user.user_id == 2:
            user_name = "malicious"
        else:
            user_name = user.user_id

        print(f"\nUser: {user_name} \nLoss: {loss} \nAcc: {acc}") 

    user_weights = [
        user.get_weight()
        for user in users
    ]

    server.aggregate(user_weights)
    pr, class_acc = server.evaluate(test_loader)
    print(f"\nPoisoning rate: {pr}  Class0 Acc: {class_acc}")

    new_global_weights = server.broadcast_weights()
    for user in users:
        user.set_weight(new_global_weights)

    print("Next round...")
    




