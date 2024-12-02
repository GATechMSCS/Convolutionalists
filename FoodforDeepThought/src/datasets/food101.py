import os
import torchvision
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
import torchvision.transforms as transforms


class food101:
    def __init__(self, random_seed = 101, batch_size = 128, perc_keep = 1.0):
        self.data_dir = os.path.join("data", "food101") # Directory in which dataset resides
        self.random_seed = random_seed
        self.batch_size = batch_size
        
        self.perc_keep = perc_keep # Percentage of dataset to be kept (number between 0 and 1)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((250, 250)),
                transforms.ToTensor(),
                # transforms.Normalize
            ]
        )

        # Placeholder variables for the training, validation, and test sets:
        self.train = None
        self.val = None
        self.test = None
    

    def load_data(self):

        """ This function strives to download the Food101 dataset to the local data directory if
        it has not already been downloaded previously. This function also splits the datasets into training,
        validation, and testing sets, assigning them as class variables. """
        
        # If the dataset does not exist locally, then download:
        if os.path.exists(self.data_dir):
            download_param = False
        else:
            download_param = True

        # Loading/downloading datasets:
        train_raw = Food101(root=self.data_dir, split="train", download=download_param, transform=self.transforms)
        test_raw = Food101(root=self.data_dir, split="test", download=download_param, transform=self.transforms)

        # Seed generator:
        generator = torch.Generator().manual_seed(self.random_seed)
        
        # Calculating the limited sizes of the datasets to keep:
        train_size = int(len(train_raw) * self.perc_keep)
        test_size = int(len(test_raw) * self.perc_keep)

        # Decreasing the size of the datasets using random_split:
        train_raw = random_split(train_raw, [train_size, (len(train_raw)-train_size)])
        self.test = random_split(test_raw, [test_size, (len(test_raw)-test_size)])

        # Splitting the training set further into training and validation sets:
        self.train, self.val = random_split(train_raw, [int(0.8 * len(train_raw)), (len(train_raw) - int(0.8 * len(train_raw)))], generator=generator)        
        
            
    # def getTrainingAndValidationData(self):
    #     data = Food101(root=self.data_dir, split='train', download=True, transform=self.transformer)
    #     generator = torch.Generator().manual_seed(self.random_seed)
    #     return random_split(data, [0.8, 0.2], generator=generator)

    # def getTrainingData(self):
    #     training, validation = self.getTrainingAndValidationData()
    #     return DataLoader(training, batch_size=self.batch_size, shuffle=True)

    # def getValidationData(self):
    #     training, validation = self.getTrainingAndValidationData()
    #     return DataLoader(validation, batch_size=self.batch_size, shuffle=True)

    # def getTestData(self):
    #     test_data = torchvision.datasets.Food101(root=self.data_dir, split='test', download=True, transform=self.transformer)
    #     return DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
