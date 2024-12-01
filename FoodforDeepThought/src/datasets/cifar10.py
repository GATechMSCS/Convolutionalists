import os
import torchvision
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Cifar10:
    def __init__(self, random_seed = 101, batch_size = 128):
        self.root = os.path.join("data", "cifar10")
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.transformer = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def getTrainingAndValidationData(self):
        data = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=self.transformer)
        generator = torch.Generator().manual_seed(self.random_seed)
        return random_split(data, [0.8, 0.2], generator=generator)

    def getTrainingData(self):
        training, validation = self.getTrainingAndValidationData()
        return DataLoader(training, batch_size=self.batch_size, shuffle=True)

    def getValidationData(self):
        training, validation = self.getTrainingAndValidationData()
        return DataLoader(validation, batch_size=self.batch_size, shuffle=True)

    def getTestData(self):
        test_data = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.transformer)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
