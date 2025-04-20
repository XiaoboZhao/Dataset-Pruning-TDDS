import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset

########################################################################################################################
# Load Data
########################################################################################################################

def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(args)
    elif args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(args)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_loader, test_loader

def load_cifar10(args):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

class CustomMNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.target_index = [[dataset.targets[i], i] for i in range(len(dataset.targets))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]  # Get the original image and target
        target_with_index = self.target_index[index]  # Add the index to the target
        return img, target_with_index

def load_mnist(args):
    """
    Load MNIST dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading MNIST... ', end='')
    time_start = time.time()
    
    train_transform = transforms.Compose([
        transforms.resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = dset.MNIST(args.data_path, train=True, transform=train_transform, download=True)
    # target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    # train_data.targets = target_index
    train_data = CustomMNISTDataset(train_data)  # Wrap the dataset with the custom class
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_data = dset.MNIST(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader
