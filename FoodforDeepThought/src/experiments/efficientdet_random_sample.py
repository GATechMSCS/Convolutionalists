import json
import os
import random
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches

from ..dataset_loaders.download_openimages import OpenImagesLoader


def run():
    dataset_loader = OpenImagesLoader()
    confidence_score = 0.1
    num_ranks = 5
    ranks = []
    filename = 'joes_open_images_results.json'
    with open(os.path.join('src', 'efficientdet-pytorch', filename)) as f:
        results = json.load(f)

    test_image_files = os.listdir(os.path.join('data', 'openimages', 'val', 'images'))
    random_index = random.randint(1, len(test_image_files)) - 1
    image_file = test_image_files[random_index]
    image_id = image_file.split('.')[0]
    print(f'Image Selected: {image_id}')

    image_results = [result for result in results if result['image_id'] == image_id and result['score'] > confidence_score]
    print(f'Number of bounding boxes: {len(image_results)}')

    max_score = 0
    rank_1 = None
    for result in image_results:
        if result['score'] > max_score:
            rank_1 = result
            max_score = result['score']


    class_name = dataset_loader.classes[rank_1['category_id'] - 1]
    x = rank_1['bbox'][0]
    y = rank_1['bbox'][1]
    w = rank_1['bbox'][2]
    h = rank_1['bbox'][3]
    print(f'Class Name: {class_name}')
    print(rank_1)

    image = cv2.imread(os.path.join('data', 'openimages', 'val', 'images', image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_axis_off()

    box = patches.Rectangle((x, y), w, h, facecolor='none', edgecolor='red', linewidth=1, label=class_name)
    ax.add_patch(box)
    ax.text(x+10, y-18, class_name, color='white', backgroundcolor='red')

    plt.show()


if __name__ == '__main__':
    run()
