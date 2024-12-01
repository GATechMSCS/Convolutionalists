import torch
from torch import nn, optim
from ..datasets.cifar10 import Cifar10
from ..models.resnet18 import ResNet18
from ..model_managers.standard_model_manager import StandardMondelManager


def run():
    model = ResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_manager = StandardMondelManager(model, criterion, optimizer)

    dataset = Cifar10()
    training_data = dataset.getTrainingData()

    model_manager.train(training_data, epochs=2)

if __name__ == "__main__":
    run()
