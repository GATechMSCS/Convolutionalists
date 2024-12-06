from torch import nn
import torchvision.models as models


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        kwargs = {'num_classes': num_classes}
        self.model = models.densenet121(**kwargs)

    def forward(self, x):
        return self.model(x)
