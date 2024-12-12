Team Convolutionalists will perform object detection, visualization, and NLP to detect foods and get recipes and health benefits

### Code Structure

The source code (`FoodforDeepThought/src`) is currently set up to run experiments 
with the following directory structure:

- **experiments** - This is where we implement and run experiments that we want to analyze, see example_experiment.py
- **datasets** - This directory is for dataset classes used to download and provide training, validation, and test data
- **models** - This directory is for model classes to provide models for our experiments
- **model_managers** - This directory is for model manager classes that train, save, and load a model and 
use the model to make predictions
- **model_saves** - This directory is for saving the models we use in our experiments so they can be used in 
other experiments

To run an experiment, simply type something like the following in a terminal 
from within the `FoodforDeepThought` directory.
```
python -m src.experiments.example_experiment
```

### EfficientDet Module

This source code contains a copy of the [efficientdet-pytorch repo](https://github.com/rwightman/efficientdet-pytorch)
copied from a fork of the repo where I made modifications to run a dataset that uses our Open Images dataset with 
Pascal annotations.

To run this program, please be sure the environment meets the requirements in 
`src/efficientdet-pytorch/requirements.txt`.

Here are some example commands that I have run to train a model on our Open Images dataset of food images.

To prepare the data, simply run the following
```
python -m src.experiments.example_experiment
```

Then copy `train/`, `val/`, and `test/` directories from `data/openimages/` to `src/efficientdet-pytorch/data/OpenImages/`


To train the model:
```
https://github.com/rwightman/efficientdet-pytorch
```

To predict results using the validation set:
```
python validate.py data/OpenImages/ -b 14 --model efficientdet_q0 --pretrained --use-ema --dataset pascalDefault --results open_images_results.json --split val --checkpoint output/train/20241210-150600-efficientdet_q0/model_best.pth.tar
```

To predict results using the test set:
```
python validate.py data/OpenImages/ -b 14 --model efficientdet_q0 --pretrained --use-ema --dataset pascalDefault --results open_images_results.json --split test --checkpoint output/train/20241210-150600-efficientdet_q0/model_best.pth.tar
```


### DERT Module
Ensure to clone Meta Research's DETR repository (https://github.com/facebookresearch/detr) to this level in the project structure. Within the build() method in the DETR repo, hard-code the number of classes to 65.
