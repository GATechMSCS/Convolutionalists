from torch import nn
import torchvision.models as models


class FasterRCnn(nn.Module):
    def __init__(self, num_classes=10):
        super(FasterRCnn, self).__init__()
        kwargs = {'num_classes': num_classes}
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(**kwargs)

    def forward(self, images, targets=None):
        return self.model(images, targets)
