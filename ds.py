#datasets.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder

conversion = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor()
])

def load_cifar10(num_client):
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
        range(3000)
    )

    #classes = cifar_train.classes

    #split dataset for users
    ud = len(small_dataset)//num_client
    user_datasets = random_split(
        small_dataset,
        [ud for _ in range(num_client)] #add to users per item
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

    return user_dataloader, test_loader

#def load_cinic10(num_client):  #not active (memory issues)
    cinic_train = ImageFolder(
        root="./cinic-10/train",
        transform=conversion
    )

    small_dataset = Subset(
        cinic_train,
        range(3000)
    )

   #split dataset for users
    ud = len(small_dataset)//num_client
    user_datasets = random_split(
        small_dataset,
        [ud for _ in range(num_client)] #add to users per item
    )

    user_dataloader = [DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    ) for dataset in user_datasets]

    cinic_test = ImageFolder(
        root="./cinic-10/test",
        transform=conversion
    )

    test_loader = DataLoader(
        cinic_test,
        batch_size=64,
        shuffle=False
    )

    return user_dataloader, test_loader