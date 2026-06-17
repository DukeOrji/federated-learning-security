#main.py
from ds import load_cifar10
from user import User, LabelPoison, WeightMan, SignFlip
from server import Server

server = Server()
num_client = 6
user_dataloader, test_loader = load_cifar10(num_client)

#create clients
users = [
    User(0, user_dataloader[0]),
    User(1, user_dataloader[1]),
    User(2, user_dataloader[2]),
    User(3, user_dataloader[3]),
    User(4, user_dataloader[4]),
    SignFlip(5, user_dataloader[5])
]

rng_num = 7
#initialize multiple rounds - improve accuracy
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
            user_name = "SignFlip"
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
        print(trust_scores)