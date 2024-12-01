import torch
from torch import nn, optim
from ..datasets.cifar10 import Cifar10
from ..models.resnet18 import ResNet18
from ..solvers.standard_solver import StandardSolver


def run():
    model = ResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    solver = StandardSolver(model, criterion, optimizer)

    dataset = Cifar10()
    training_data = dataset.getTrainingData()

    solver.train(training_data, epochs=2)

if __name__ == "__main__":
    run()
