import copy
import torch


class StandardModelManager:
    def __init__(self, model, criterion, optimizer):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.best_accuracy = 0.0
        self.best_model_state_dict = None

    def train(self, training_data_loader, validation_data_loader = None, epochs=10):
        for epoch in range(epochs):
            display_epoch = epoch + 1
            for idx, (data, target) in enumerate(training_data_loader):
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

            if validation_data_loader is not None:
                num_correct = 0
                num_samples = 0
                for idx, (data, target) in enumerate(validation_data_loader):
                    target = target.to(self.device)
                    with torch.no_grad():
                        output = self.model.forward(data.to(self.device))
                        loss = self.criterion(output, target)

                    # Check Accuracy
                    batch_size = target.shape[0]
                    _, pred = torch.max(output, dim=-1)
                    correct = pred.eq(target).sum() * 1.0
                    acc = correct / batch_size
                    num_correct += correct
                    num_samples += batch_size

                acc = num_correct / num_samples

                print(f'Epoch {display_epoch} Batch Validation Accuracy: {acc:.4f}')
                print('===========================================================')

                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

    def predict(self, data):
        with torch.no_grad():
            output = self.model.forward(data.to(self.device))
            _, pred = torch.max(output, dim=-1)

        return pred, output

    def save(self, filepath):
        if self.best_model_state_dict is None:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.best_model_state_dict, filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, weights_only=True))

