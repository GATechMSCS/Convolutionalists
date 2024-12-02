import torch


class StandardModelManager:
    def __init__(self, model, criterion, optimizer):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            display_epoch = epoch + 1
            for idx, (data, target) in enumerate(data_loader):
                # Train Batch
                target = target.to(self.device)
                output = self.model.forward(data.to(self.device))
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Check Accuracy
                batch_size = target.shape[0]
                _, pred = torch.max(output, dim=-1)
                correct = pred.eq(target).sum() * 1.0
                acc = correct / batch_size

                if idx % 10 == 0:
                    print(f'Epoch {display_epoch} Batch Training Accuracy: {acc:.4f}')

    def predict(self, data):
        with torch.no_grad():
            output = self.model.forward(data.to(self.device))
            _, pred = torch.max(output, dim=-1)

        return pred, output

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, weights_only=True))

