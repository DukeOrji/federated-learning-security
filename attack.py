from user import User
import torch
import torch.nn as nn
import torch.optim as optim
from config import device
from user import norm

class InputNoise(User):
    pass

class LabelFlip(User):
    def train(self):
        losses = []
        self.model.train()

        # track how often model predicts the poisoned labels correctly
        correct_pred = 0
        # track total samples processed
        total = 0

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > 2:
                break
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            # create a copy so original labels remain unchanged
            poisoned_labels = labels.clone()

            # label poisoning attack:
            # Shift every label to the next class
            poisoned_labels[:] = (labels + 1) % 10

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
                update = (weights[key] - self.global_weights[key])
                mal_update = -3 * update
                weights[key] = (self.global_weights[key] + mal_update)

        return weights