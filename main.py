#main.py
import torch
from ds import load_cifar10
from user import User, LabelPoison, WeightMan, SignFlip
from server import Server
import pandas as pd
from server import device



#saves results
df = pd.DataFrame()
results = []

"""df.to_csv(
    "results/trust_scoring.csv",
    index=False
)"""


server = Server()
num_client = 10
user_dataloader, test_loader = load_cifar10(num_client)

#create clients
users = [
    User(0, user_dataloader[0]),
    User(1, user_dataloader[1]),
    User(2, user_dataloader[2]),
    User(3, user_dataloader[3]),
    User(4, user_dataloader[4]),
    WeightMan(5, user_dataloader[5]),
    User(6, user_dataloader[6]),
    User(7, user_dataloader[7]),
    User(8, user_dataloader[8]),
    User(9, user_dataloader[9]),
]

rng_num = 10
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

        """if user.user_id == 4:
            user_name = "WeightManipulation" """
        
        if user.user_id == 5:
            user_name = "WeightMan"
        else:
            user_name = user.user_id

        print(f"\nUser: {user_name} Loss: {loss} Acc: {acc}") 

    user_weights = [
        user.get_weight()
        for user in users
    ]

    server.aggregate(user_weights)
    global_loss, global_acc, trust_scores = server.weight_man_evaluate(test_loader)
    print(f"\nGlobal Loss: {global_loss}  Global Acc: {global_acc}")

    new_global_weights = server.broadcast_weights()
    for user in users:
        user.set_weight(new_global_weights)

    if epoch < rng_num-1:
        print("Next round...")
    else:
        print("Experiment Complete.")

        print(trust_scores, sep="\n")