import os
import torch
from torch import nn, optim
from ..dataset_loaders.openimages import OpenImagesLoader
from ..models.resnet18 import ResNet18
from ..model_managers.standard_model_manager import StandardModelManager


def run():
    dataset_loader = OpenImagesLoader()

    dataset_loader.download_data()


if __name__ == "__main__":
    run()
