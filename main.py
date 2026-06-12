#main.py
from ds import load_cifar10
from user import User, LabelPoison, WeightMan
from server import Server

server = Server()
user_dataloader, test_loader = load_cifar10()
#create clients
users = [
    User(0, user_dataloader[0]),
    User(1, user_dataloader[1]),
    WeightMan(2, user_dataloader[2])
]

rng_num = 3
#initialize multiple rounds - improve accuracy
for epoch in range(rng_num):
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
    global_loss, global_acc = server.weight_man_evaluate(test_loader)
    print(f"\nGlobal Loss: {global_loss}  Global Acc: {global_acc}")

    new_global_weights = server.broadcast_weights()
    for user in users:
        user.set_weight(new_global_weights)

    if epoch < rng_num-1:
        print("Next round...")
    else:
        print("Experiment Over.")
        break