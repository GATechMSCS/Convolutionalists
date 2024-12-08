import copy
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from ..dataset_loaders.download_openimages import OpenImagesLoader
from ..models.fasterrcnn import FasterRCnn


class PascalDataset(Dataset):
    def __init__(self, targets):
        super().__init__()
        self.filenames = [target['filename'] for target in targets]
        self.targets = [target['boxes'] for target in targets]
        self.labels = [target['labels'] for target in targets]
        self.transforms = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # ImageNet's normalization statistics
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target = self.targets[idx]
        label = self.labels[idx]

        image = read_image(self.filenames[idx])  # Extracting image from file path

        # If the image has an extra channel:
        if image.shape[0] > 3:
            image = image[:3, :, :]  # Removing the extra channel from the image

        # If it's a grayscale image, then repeat the row and column values three times:
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = to_pil_image(image)  # Convert the image to PIL format
        image = self.transforms(image)  # Applying transforms to current image

        return image, target, label


def run():
    dataset_loader = OpenImagesLoader()
    targets = dataset_loader.getTargets()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = FasterRCnn(num_classes=4).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset = PascalDataset(targets)

    data_loader = DataLoader(dataset=dataset, batch_size=4)

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