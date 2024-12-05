import os
import torch
from torch import nn, optim
from ..dataset_loaders.cifar10 import Cifar10Loader
from ..models.densenet121 import DenseNet121
from ..model_managers.standard_model_manager import StandardModelManager


def run():
    model = DenseNet121(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_manager = StandardModelManager(model, criterion, optimizer)

    dataset = Cifar10Loader()
    training_data = dataset.getTrainingData()
    validation_data = dataset.getValidationData()

    model_manager.train(training_data, validation_data, epochs=20)

    data, target = next(iter(validation_data))

    pred1, _ = model_manager.predict(data)

    model_manager.save(os.path.join('src', 'model_saves', 'densenet121_densenet_experiment.pth'))


if __name__ == "__main__":
    run()
