import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from torchvision import datasets

from ..models.fasterrcnn import FasterRCnn


class RandomDataset(Dataset):
    def __init__(self, images, targets):
        super().__init__()
        self.images = copy.deepcopy(images)
        self.targets = [target['boxes'] for target in targets]
        self.labels = [target['labels'] for target in targets]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        label = self.labels[idx]
        # # Open and process the file
        # with open(file_path, 'r') as f:
        #     data = f.read()
        #     # Process the data as needed
        return image, target, label


def run():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = FasterRCnn(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # For training
    images, boxes = torch.rand(8, 3, 256, 256).to(device), torch.randint(1, 100, (8, 11, 4)).to(device)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.randint(1, 4, (8, 11)).to(device)
    images = list(image for image in images)
    targets = []

    for i in range(len(images)):
        d = {'boxes': boxes[i, ...], 'labels': labels[i, ...]}
        targets.append(d)

    dataset = RandomDataset(images, targets)

    data_loader = DataLoader(dataset=dataset, batch_size=4)

    # output = model(images, targets)

    training_accs = []  # List of training accuracy values
    val_accs = []  # List of validation accuracy values
    epochs = 2
    for epoch in tqdm(range(epochs)):
        display_epoch = epoch + 1
        for images, boxes, labels in data_loader:
            # Build targets
            targets = []
            for i in range(labels.size(dim=0)):
                targets.append({'boxes': boxes[i, ...], 'labels': labels[i, ...]})

            # Train Batch
            optimizer.zero_grad()
            losses = model.forward(images, targets)
            loss = torch.sum(torch.stack(list(losses.values())))
            loss.backward()
            optimizer.step()


    # For inference
    model.eval()
    x = torch.rand(2, 3, 256, 256).to(device)
    predictions = model.forward(x)
    print(predictions)


if __name__ == "__main__":
    run()