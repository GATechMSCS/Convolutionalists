import copy
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class StandardModelManager:
    def __init__(self, model, criterion, optimizer, device=None):

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.best_accuracy = 0.0
        self.best_model_state_dict = None
        self.training_accs = None
        self.val_accs = None

    def train(self, training_data_loader, validation_data_loader = None, epochs=10):

        training_accs = [] # List of training accuracy values
        val_accs = [] # List of validation accuracy values
        
        for epoch in tqdm(range(epochs)):
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

            # Appending the training accuracy at the end of the current epoch to the list of training accuracy values:
            training_accs.append(acc)
            
            if validation_data_loader:
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

            # Appending the validation accuracy for the current epoch to the list of validation accuracy values:
            val_accs.append(acc)
        
        # Load best state after training for use
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)

        # Setting the training and validation accuracy lists to their respective class variables:
        self.training_accs = training_accs
        self.val_accs = val_accs
    
    def predict(self, data):
        with torch.no_grad():
            output = self.model.forward(data.to(self.device))
            _, pred = torch.max(output, dim=-1)

        return pred, output

    def test(self, test_data_loader):
        """ This function applies the trained model to the given test data. 
            It prints and returns the test accuracy.
        """
        
        for idx, (data, target) in enumerate(test_data_loader):
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

        print(f'Test Accuracy: {acc:.4f}')
        print('===========================================================')

        return acc

    def plot_learning_curve(self, model_name):
        """
        This function plots the learning curve from the most recent training period of this model manager.
        
        Inputs:
        model_name (str) - Name of the model
        """

        title = model_name + " Learning Curve"
        filename = model_name + "_learning_curve.png"

        # Moving tensors to CPU:
        for i, values in enumerate(zip(self.training_accs, self.val_accs)):
            self.training_accs[i] = values[0].to('cpu')
            self.val_accs[i]=values[1].to('cpu')

        # Plotting training and validation accuracy values:
        plt.plot(self.training_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig(filename, dpi=600)
        plt.show()
        
    
    def save(self, filepath):
        if self.best_model_state_dict is None:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.best_model_state_dict, filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, weights_only=True))

