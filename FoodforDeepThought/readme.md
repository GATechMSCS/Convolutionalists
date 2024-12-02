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
