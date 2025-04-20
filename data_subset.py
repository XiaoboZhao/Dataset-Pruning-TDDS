import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset

########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, data_mask, sorted_score):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100_sub(args, data_mask, sorted_score):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

class CustomMNISTDataset(Dataset):
    def __init__(self, dataset, data_mask, sorted_score):
        self.dataset = dataset
        score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
        self.targets_with_score = [[dataset.targets[i], score[np.where(data_mask == i)]] for i in range(len(dataset.targets))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]  # Get the original image and target
        target_with_score = self.targets_with_score[index]
        return img, target_with_score

def load_mnist_sub(args, data_mask, sorted_score):
    """
    Load MNIST dataset with specified transformations and subset selection.
    """
    print('Loading MNIST... ', end='')
    time_start = time.time()
    
    mean = (0.1307,)
    std = (0.3081,)
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.MNIST(args.data_path, train=True, transform=train_transform, download=True)
    train_data = CustomMNISTDataset(train_data, data_mask, sorted_score)

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.MNIST(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader
