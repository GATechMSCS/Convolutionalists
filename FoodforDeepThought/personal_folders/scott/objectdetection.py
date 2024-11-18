import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class ObjectDetection:
    def __init__():
        pass

    def yolo():

        # Load a pretrained YOLOv11 model
        # or 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt' depending on your needs
        model = YOLO('yolo11n.pt')

        # Fine-tune the model on your dataset
        results = model.train(data='path/to/your_dataset.yaml', epochs=100, imgsz=640)

        # # Fine-tune the model on Food-101 or Food2K dataset
        # model.train(data='path/to/food_dataset.yaml', epochs=100, imgsz=640)

        # Perform inference
        results = model('path/to/your_image.jpg')

        # Display results
        results.show()

    def faster_rcnn():
        
        def get_model(num_classes):
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model

        # Create the model
        model = get_model(num_classes=101)  # 101 for Food-101, 2000 for Food2K

        # Train the model (simplified)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        for epoch in range(num_epochs):
            for images, targets in data_loader:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

        # Perform inference
        model.eval()
        with torch.no_grad():
            prediction = model(image)[0]

def main():
    od = ObjectDetection()
    od.yolo()
    od.faster_rcnn()

if __name__ == "__main__":
    main()