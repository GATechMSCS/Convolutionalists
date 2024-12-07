import os
import torch
from torch import nn, optim
from ..dataset_loaders.download_openimages import OpenImagesLoader
from ..models.resnet18 import ResNet18
from ..model_managers.standard_model_manager import StandardModelManager


def run():
    dataset_loader = OpenImagesLoader()
    dataset_loader.download_data()
    dataset_loader.split_data(keep_class_dirs=False)
    print([x.lower() for x in dataset_loader.classes])

if __name__ == "__main__":
    run()
