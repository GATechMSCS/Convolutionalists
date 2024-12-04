import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import ViTForImageClassification, ViTFeatureExtractor

class ObjectDetection:
    def __init__():
        pass

    def yolo():

        model = YOLO('yolo11n.pt')
        
        results = model.train(data=filepath, epochs=100, imgsz=640)

        results = model('path/to/your_image.jpg')

        results.show()

    def faster_rcnn():
        
        def get_model(num_classes):
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model

        # Create the model
        model = get_model(num_classes=138)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        for epoch in range(num_epochs):
            for images, targets in data_loader:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(image)[0]

    def ViT():

        # Load the pre-trained model and feature extractor
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def main():
    od = ObjectDetection()
    od.yolo()
    od.faster_rcnn()
    od.ViT()

if __name__ == "__main__":
    main()