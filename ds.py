#datasets.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


conversion = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor()
])

def load_cifar10():
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

    return user_dataloader, test_loader

def load_cinic10():
    pass