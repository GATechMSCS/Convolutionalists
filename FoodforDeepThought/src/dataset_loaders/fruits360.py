import os
import torchvision
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class Fruits360Loader:
    def __init__(self, random_seed = 101, batch_size = 128, perc_keep = 1.0):
        self.train_dir = os.path.join("data", "fruits-360", "Training") # Directory in which training dataset resides
        self.test_dir = os.path.join("data", "fruits-360", "Test") # Directory in which testing dataset resides 
        self.random_seed = random_seed
        self.batch_size = batch_size
        
        self.perc_keep = perc_keep # Percentage of dataset to be kept (number between 0 and 1)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet's normalization statistics
            ]
        )
    

    def load_data(self):

        """ This function strives to download the Food101 dataset to the local data directory if
        it has not already been downloaded previously. This function also splits the datasets into training,
        validation, and testing sets, assigning them as class variables. """

        # Note - this assumes the Fruits 360 dataset has already been downloaded to their respective directories:.
        # If the dataset has not been downloaded, then please manually download it and place it in the directories
        # as described in the class initialization:
        train_raw = ImageFolder(self.train_dir, transform = self.transforms)
        test_raw = ImageFolder(self.test_dir, transform = self.transforms)

        # Seed generator:
        generator = torch.Generator().manual_seed(self.random_seed)

        if self.perc_keep != 1.00:
            # Calculating the limited sizes of the datasets to keep:
            train_size = int(len(train_raw) * self.perc_keep)
            test_size = int(len(test_raw) * self.perc_keep)

            # Decreasing the size of the datasets using random_split:
            train_raw, _ = random_split(train_raw, [train_size, (len(train_raw)-train_size)])
            test_raw, _ = random_split(test_raw, [test_size, (len(test_raw)-test_size)])
        
        test_set = DataLoader(test_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
        
        # Splitting the training set further into training and validation sets:
        train_set, val_set = random_split(train_raw, [int(0.8 * len(train_raw)), (len(train_raw) - int(0.8 * len(train_raw)))], generator=generator)        

        # Applying DataLoaders to the training and validation sets:
        train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        val_set = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_set, val_set, test_set
            
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
