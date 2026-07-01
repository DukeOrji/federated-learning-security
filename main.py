#main.py
import torch
from ds import load_cifar10
from user import User
from attack import LabelFlip, WeightMan, SignFlip
from server import Server
import pandas as pd
from server import device



client_results = []
results = []



num_client = 10
server = Server(num_client)
user_dataloader, test_loader = load_cifar10(num_client)

#create clients
attack_type = {
    2: "LF",
    3: "LF"
}

users = [
    User(0, user_dataloader[0]),
    User(1, user_dataloader[1]),
    LabelFlip(2, user_dataloader[2]),
    LabelFlip(3, user_dataloader[3]),
    User(4, user_dataloader[4]),
    User(5, user_dataloader[5]),
    User(6, user_dataloader[6]),
    User(7, user_dataloader[7]),
    User(8, user_dataloader[8]),
    User(9, user_dataloader[9])
]

rng_num = 20
#initialize multiple rounds - improve accuracy
print(next(server.global_model.parameters()).device) #print gpu or cpu
for epoch in range(rng_num):
    print(f"\nRound {epoch+1}")
    #broadcast global model
    global_weights = server.broadcast_weights()

    #send weights to users
    for user in users:
        user.set_weight(global_weights)
        loss, acc = user.train()
        
        #get the attack type if this user ID exists in the dictionary, otherwise return the user ID.
        user_name = attack_type.get(user.user_id, user.user_id)
        print(f"\nUser: {user_name} Loss: {loss} Acc: {acc}") 

        #save client updates
        client_results.append({
            "Round": epoch + 1,
            "User": user.user_id,
            "client_loss": loss,
            "client_acc": acc,
            "malicious": user.user_id in [2,3]
        })

    user_weights = [
        user.get_weight()
        for user in users
    ]

    server.aggregate(user_weights)
    

    pr, global_acc, trust_scores = server.label_poison_evaluate(test_loader)
    print(f"\nPoison Rate: {pr}  Global Acc: {global_acc}")

    for client_id, trust in trust_scores.items():
        results.append({
            "Round": epoch + 1,
            "User": client_id,
            "Trust score": trust,
            "global loss": pr,
            "global accuracy": global_acc
        })

    new_global_weights = server.broadcast_weights()
    for user in users:
        user.set_weight(new_global_weights)

    if epoch < rng_num-1:
        print("Next round...")
    else:
        print("Experiment Complete.")

#save results
global_df = pd.DataFrame(results)
client_df = pd.DataFrame(client_results)

client_df.to_csv(
    "results/client_updates.csv",
    index=False
)

global_df.to_csv(
    "results/trust_scoring.csv",
    index=False
)
