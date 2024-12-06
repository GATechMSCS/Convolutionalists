import os
import torch
from torch import nn, optim
from ..dataset_loaders.cifar10 import Cifar10Loader
from ..models.resnet18 import ResNet18
from ..model_managers.standard_model_manager import StandardModelManager


def run():
    model = ResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_manager = StandardModelManager(model, criterion, optimizer)

    dataset = Cifar10Loader()
    training_data = dataset.getTrainingData()
    validation_data = dataset.getValidationData()

    model_manager.train(training_data, validation_data, epochs=2)

    data, target = next(iter(validation_data))

    pred1, _ = model_manager.predict(data)

    model_manager.save(os.path.join('src', 'model_saves', 'resnet18_example_experiment.pth'))

    model_manager.load(os.path.join('src', 'model_saves', 'resnet18_example_experiment.pth'))

    pred2, _ = model_manager.predict(data)

    print(f'Predictions are equal after saving and loading the model: {torch.equal(pred1, pred2)}')


if __name__ == "__main__":
    run()
