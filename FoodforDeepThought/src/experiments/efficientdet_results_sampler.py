import json
import os
import random
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from torchvision import ops

from ..dataset_loaders.download_openimages import OpenImagesLoader


def run():
    dataset_loader = OpenImagesLoader()
    confidence_score = 0.1
    filename = 'efficientdet_test_results.json'
    with open(os.path.join('src', 'efficientdet-pytorch', filename)) as f:
        results = json.load(f)

    test_image_files = os.listdir(os.path.join('data', 'openimages', 'test', 'images'))

    fig, axs = plt.subplots(2, 2)

    for ax in axs.flatten().tolist():
        random_index = random.randint(1, len(test_image_files)) - 1
        image_file = test_image_files[random_index]
        image_id = image_file.split('.')[0]
        print(f'Image Selected: {image_id}')

        image_results = [result for result in results if result['image_id'] == image_id and result['score'] > confidence_score]
        print(f'Number of bounding boxes: {len(image_results)}')
        scores_index = np.argsort(np.array([result['score'] for result in image_results])).tolist()
        scores_index.reverse()
        print(f'Scores Index: {scores_index}')
        sorted_results = [image_results[idx] for idx in scores_index]

        iou_boxes = None
        iou_results = []
        for result in sorted_results:
            x = result['bbox'][0]
            y = result['bbox'][1]
            w = result['bbox'][2]
            h = result['bbox'][3]
            if iou_boxes is None:
                iou_boxes = torch.reshape(torch.tensor([x, y, x + w, y + h]), (1, -1))
                iou_results.append(result)
            else:
                box = torch.reshape(torch.tensor([x, y, x + w, y + h]), (1, -1))
                iou1 = ops.box_iou(iou_boxes, box)
                iou2 = ops.box_iou(box, iou_boxes)
                if torch.all(iou1 < 0.10) and torch.all(iou2 < 0.10):
                    iou_boxes = torch.cat((iou_boxes, box), 0)
                    iou_results.append(result)

        image = cv2.imread(os.path.join('data', 'openimages', 'test', 'images', image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.set_axis_off()

        for result in iou_results:
            class_name = dataset_loader.classes[result['category_id'] - 1]
            x = result['bbox'][0]
            y = result['bbox'][1]
            w = result['bbox'][2]
            h = result['bbox'][3]
            box = patches.Rectangle((x, y), w, h, facecolor='none', edgecolor='red', linewidth=1, label=class_name)
            ax.add_patch(box)
            ax.text(x+10, y-18, class_name, color='white', backgroundcolor='red', fontsize=6)

    plt.savefig(fname='sampler.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    run()
