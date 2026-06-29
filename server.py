#server.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
from user import norm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Server:
    def __init__(self):
        self.weights = ResNet18_Weights.DEFAULT
        self.global_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.global_model = self.global_model.to(device)
        self.global_model.fc = nn.Linear(512, 10).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.trust_scores = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 1.0,
            8: 1.0,
            9: 1.0
        }
        
    def broadcast_weights(self):
        return self.global_model.state_dict()
        
    def aggregate(self, user_weights, threshold=100):
        clip_threshold = 50
        # =========================
        # STEP 1: Update trust scores
        # =========================
        for idx, weights in enumerate(user_weights):

            total_dist = 0

            for key in weights.keys():

                # Ignore non-learnable parameters
                if weights[key].dtype == torch.float32:

                    
                    total_dist += torch.norm(weights[key] - self.global_model.state_dict()[key]).item()
                    

            # Update trust based on total distance
            if total_dist > threshold:
                self.trust_scores[idx] *= 0.9

            else:
                self.trust_scores[idx] = min(
                    self.trust_scores[idx] + 0.02,
                    1.0
                )

            print(
                f"Client {idx}: Distance={total_dist:.2f}, "
                f"Trust={self.trust_scores[idx]:.3f}"
            )

        # =========================
        # STEP 2: Weighted aggregation
        # =========================
        avg_weights = {}

        for key in user_weights[0].keys():

            if user_weights[0][key].dtype == torch.float32:

                weighted_sum = torch.zeros_like(user_weights[0][key])
                total_trust = 0

                for idx, weights in enumerate(user_weights):

                    trust = self.trust_scores[idx]
                    #compute update
                    update = weights[key] - self.global_model.state_dict()[key]

                    norm = torch.norm(update)
                    #clip update
                    if norm > clip_threshold:
                        update = update * (clip_threshold/norm)

                    #aggregate clipped update
                    weighted_sum += (self.global_model.state_dict()[key] + update) * trust #prevents malicious clients from dominating purely through magnitude.
                    total_trust += trust

                avg_weights[key] = weighted_sum / max(total_trust, 1e-8)

            else:
                # Copy non-trainable parameters directly
                avg_weights[key] = user_weights[0][key]

        print("\nTrust Scores:", self.trust_scores)
        print("\nAggregation complete...")

        self.global_model.load_state_dict(avg_weights)


    def label_poison_evaluate(self, dataloader):
        self.global_model.eval()

        correct= 0
        zero_to_one= 0
        class0_total= 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)#send to gpu
                labels = labels.to(device)
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
                images = images.to(device)#send to gpu
                labels = labels.to(device)
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
                losses.append(loss.item())

        avg_loss = round((sum(losses)/len(losses)), 2)
        global_acc = round(correct/total, 2)
            

        return avg_loss, global_acc, self.trust_scores
